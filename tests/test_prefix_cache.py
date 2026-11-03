"""Tests for prefix-cache scenarios: counters, hit rate, cache reset, warmup and summaries."""

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

import furiosa_perf.benchmark.vllm as vllm_mod
from furiosa_perf.benchmark.vllm import VllmPerformanceBenchmark
from furiosa_perf.reporting.schemas import BenchmarkMetricLoader
from furiosa_perf.runner.monitor import HardwareMonitor, PrefixCacheCounters
from furiosa_perf.utils.config import LLMScenarioConfig

VLLM_METRICS = """\
# HELP vllm:prefix_cache_queries Prefix cache queries, in terms of number of queried tokens.
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries_total{engine="0",model_name="m"} 4096.0
vllm:prefix_cache_queries_total{engine="1",model_name="m"} 1024.0
vllm:prefix_cache_queries_created{engine="0",model_name="m"} 1.7e9
vllm:prefix_cache_hits_total{engine="0",model_name="m"} 3072.0
vllm:external_prefix_cache_hits_total{engine="0"} 999.0
"""

# As served by furiosa-llm 2026.4.0 (no `_total` suffix), plus the hit-rate gauge.
FURIOSA_METRICS = """\
# HELP furiosa_llm_prefix_cache_queries Prefix cache queries, in terms of number of queried tokens.
# TYPE furiosa_llm_prefix_cache_queries counter
furiosa_llm_prefix_cache_queries{model_name="furiosa-ai/Mistral-Nemo-Instruct-2407-FP8",engine="0"} 8192
# HELP furiosa_llm_prefix_cache_hits Prefix cache hits, in terms of number of cached tokens.
# TYPE furiosa_llm_prefix_cache_hits counter
furiosa_llm_prefix_cache_hits{model_name="furiosa-ai/Mistral-Nemo-Instruct-2407-FP8",engine="0"} 6144
furiosa_llm_prefix_cache_hit_rate{engine="0"} 0.75
furiosa_llm_cache_config_info{engine="0",block_size="1",num_gpu_blocks="967518",enable_prefix_caching="true"} 1
"""

VLLM_STDOUT = """\
Output token throughput (tok/s):         100.00
Total token throughput (tok/s):          500.00
Mean TTFT (ms):                          200.00
Mean TPOT (ms):                          10.00
"""


def make_bench(tmp_path: Path) -> VllmPerformanceBenchmark:
    """A benchmark object with the attributes the tested methods use, skipping venv setup."""
    bench = object.__new__(VllmPerformanceBenchmark)
    bench.name = "t"
    bench.model = "org/model"
    bench.device_name = "RNGD"
    bench.used_device_num = 2
    bench.task = "offline"
    bench.backend = "furiosa-llm"
    bench.host = "127.0.0.1"
    bench.port = 8000
    bench.base_dir = tmp_path
    bench.total_results = {}
    bench._reset_unavailable = False
    bench._cache_reset_ok = False
    bench.BASE_COMMAND = [arg.replace("{exe}", "python") for arg in bench.COMMAND]
    return bench


def test_parse_vllm_counters_sums_engines_and_skips_created_and_external() -> None:
    counters = HardwareMonitor.parse_prefix_cache_counters(VLLM_METRICS)
    assert counters == PrefixCacheCounters(queries=5120.0, hits=3072.0)


def test_parse_furiosa_counters_ignores_hit_rate_gauge() -> None:
    counters = HardwareMonitor.parse_prefix_cache_counters(FURIOSA_METRICS)
    assert counters == PrefixCacheCounters(queries=8192.0, hits=6144.0)


def test_parse_returns_none_without_counters() -> None:
    assert HardwareMonitor.parse_prefix_cache_counters("# nothing\nfoo_total 1\n") is None


def test_hit_rate_uses_the_delta_and_leaves_out_the_warmup(tmp_path: Path) -> None:
    bench = make_bench(tmp_path)
    bench._cache_reset_ok = True
    scenario = LLMScenarioConfig(input_tokens=1024, random_prefix_len=3072, output_tokens=64)
    # A 4096-token warmup (4095 lookups, 0 hits), then one timed request that hits the prefix
    # (3072 of 4095). As measured on furiosa-llm 2026.4.0.
    summary = bench._prefix_cache_summary(
        scenario, PrefixCacheCounters(queries=1000, hits=500), PrefixCacheCounters(queries=9190, hits=3572)
    )
    assert summary == {"Prefix Tokens": 3072, "Prefix Cache Hit Rate(%)": 75.02}


def test_plain_hit_rate_keeps_every_lookup(tmp_path: Path) -> None:
    bench = make_bench(tmp_path)
    scenario = LLMScenarioConfig(input_tokens=4096, output_tokens=64)
    summary = bench._prefix_cache_summary(
        scenario, PrefixCacheCounters(queries=0, hits=0), PrefixCacheCounters(queries=8190, hits=4095)
    )
    assert summary == {"Prefix Tokens": 0, "Prefix Cache Hit Rate(%)": 50.0}


@pytest.mark.parametrize(
    ("prefix", "concurrency", "expected"),
    [(0, 1, 3), (0, 8, 24), (3072, 1, 10), (3072, 2, 10), (3072, 8, 24)],
)
def test_prefix_scenarios_get_at_least_10_prompts(prefix: int, concurrency: int, expected: int) -> None:
    scenario = LLMScenarioConfig(random_prefix_len=prefix, max_concurrency=concurrency)
    assert scenario.num_prompts == expected


@pytest.mark.parametrize(
    ("before", "after"),
    [(None, PrefixCacheCounters(1, 1)), (PrefixCacheCounters(10, 0), PrefixCacheCounters(10, 0))],
)
def test_hit_rate_is_na_without_counters_or_lookups(
    tmp_path: Path, before: PrefixCacheCounters | None, after: PrefixCacheCounters
) -> None:
    bench = make_bench(tmp_path)
    summary = bench._prefix_cache_summary(LLMScenarioConfig(), before, after)
    assert summary == {"Prefix Tokens": 0, "Prefix Cache Hit Rate(%)": "N/A"}


def test_negative_prefix_is_rejected() -> None:
    with pytest.raises(ValueError, match="random_prefix_len"):
        LLMScenarioConfig(random_prefix_len=-1)


def test_prefix_scenarios_warm_up_once_and_plain_ones_do_not(tmp_path: Path) -> None:
    bench = make_bench(tmp_path)
    plain = bench._get_command_for_scenario(LLMScenarioConfig(input_tokens=4096, output_tokens=64))
    prefixed = bench._get_command_for_scenario(
        LLMScenarioConfig(input_tokens=1024, random_prefix_len=3072, output_tokens=64)
    )
    assert "--num-warmups=0" in plain
    assert "--num-warmups=1" in prefixed
    assert "--random-prefix-len=3072" in prefixed
    assert bench._scenario_result_name(LLMScenarioConfig(input_tokens=1024, random_prefix_len=3072)) == (
        "1024.1024.1.p3072"
    )


class FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = ""


    def raise_for_status(self) -> None:
        if not self.ok:
            raise requests.HTTPError(str(self.status_code))


def test_reset_posts_to_the_server_then_sends_a_tiny_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_post(url: str, **_: Any) -> FakeResponse:
        calls.append(url)
        return FakeResponse(200)

    monkeypatch.setattr(vllm_mod.requests, "post", fake_post)
    bench = make_bench(tmp_path)
    bench._reset_prefix_cache()
    bench._reset_prefix_cache()
    assert calls == [
        "http://127.0.0.1:8000/reset_prefix_cache",
        "http://127.0.0.1:8000/v1/completions",
    ] * 2


def test_reset_gives_up_after_a_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_post(url: str, **_: Any) -> FakeResponse:
        calls.append(url)
        return FakeResponse(404)

    monkeypatch.setattr(vllm_mod.requests, "post", fake_post)
    bench = make_bench(tmp_path)
    bench._reset_prefix_cache()
    bench._reset_prefix_cache()
    assert len(calls) == 1


def test_reset_survives_a_connection_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, **_: Any) -> FakeResponse:
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(vllm_mod.requests, "post", fake_post)
    make_bench(tmp_path)._reset_prefix_cache()


def result_row(bench: VllmPerformanceBenchmark, scenario: LLMScenarioConfig, hit_rate: float | str) -> dict[str, Any]:
    """Build a result dict the way ``_run_scenario`` does, with fake numbers."""
    result = bench._parse_results(VLLM_STDOUT, scenario)
    result.update({"mean_power": 100.0, "p95_power": 110.0, "p99_power": 120.0})
    result.update({"Prefix Tokens": bench._scenario_prefix_tokens(scenario), "Prefix Cache Hit Rate(%)": hit_rate})
    return result


def test_summaries_keep_prefix_runs_apart_and_report_reads_them(tmp_path: Path) -> None:
    bench = make_bench(tmp_path)
    plain = LLMScenarioConfig(input_tokens=4096, output_tokens=64, max_concurrency=8)
    prefixed = LLMScenarioConfig(input_tokens=1024, random_prefix_len=3072, output_tokens=64, max_concurrency=8)
    bench.total_results = {"results": [result_row(bench, plain, "N/A"), result_row(bench, prefixed, 74.5)]}

    bench.finish_info()

    root = bench.get_vllm_result_dir("")
    assert (root / "summary_4k_64.csv").exists()
    assert (root / "summary_4k_64_p3072.csv").exists()

    df = BenchmarkMetricLoader.load_offline_benchmark_metric(root / "summary.csv")
    assert list(df["ISL"]) == [4096, 4096]
    assert list(df["Prefix"]) == [0, 3072]
    assert pd.isna(df["PrefixHit(%)"].iloc[0])
    assert df["PrefixHit(%)"].iloc[1] == 74.5


def test_power_summary_survives_a_missing_monitor_csv(tmp_path: Path) -> None:
    summary = HardwareMonitor.get_benchmark_power_summary(
        csv_file_path=tmp_path / "missing.csv", start_dt=None, end_dt=None, target_csv_file_path=None
    )
    assert summary == {"mean_power": 0, "p95_power": 0, "p99_power": 0}


def test_temperature_defaults_to_0_and_follows_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = make_bench(tmp_path)
    monkeypatch.delenv("FURIOSA_PERF_TEMPERATURE", raising=False)
    assert "--temperature=0" in bench._get_command_for_scenario(LLMScenarioConfig())
    monkeypatch.setenv("FURIOSA_PERF_TEMPERATURE", "1")
    assert "--temperature=1" in bench._get_command_for_scenario(LLMScenarioConfig())


def warmup_counters() -> tuple[PrefixCacheCounters, PrefixCacheCounters]:
    """A 4096-token warmup (all miss) and one timed request hitting 3072 tokens."""
    return PrefixCacheCounters(queries=0, hits=0), PrefixCacheCounters(queries=8190, hits=3072)


def test_warmup_stays_in_the_hit_rate_without_a_reset(tmp_path: Path) -> None:
    bench = make_bench(tmp_path)
    scenario = LLMScenarioConfig(input_tokens=1024, random_prefix_len=3072)
    summary = bench._prefix_cache_summary(scenario, *warmup_counters())
    assert summary["Prefix Cache Hit Rate(%)"] == 37.51


def test_warmup_stays_in_the_hit_rate_for_vision_prompts(tmp_path: Path) -> None:
    bench = make_bench(tmp_path)
    bench.task = "vl-offline"
    bench._cache_reset_ok = True
    scenario = LLMScenarioConfig(input_tokens=1024, random_prefix_len=3072)
    summary = bench._prefix_cache_summary(scenario, *warmup_counters())
    assert summary["Prefix Cache Hit Rate(%)"] == 37.51


@pytest.mark.parametrize(("status", "reset_ok"), [(200, True), (404, False), (500, False)])
def test_reset_records_whether_it_worked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int, reset_ok: bool
) -> None:
    monkeypatch.setattr(vllm_mod.requests, "post", lambda url, **_: FakeResponse(status))
    bench = make_bench(tmp_path)
    bench._cache_reset_ok = True
    bench._reset_prefix_cache()
    assert bench._cache_reset_ok is reset_ok


def test_vllm_bench_is_pinned(tmp_path: Path) -> None:
    install = dict(VllmPerformanceBenchmark.ENV["SETUP_COMMANDS"])["install_package"]
    assert "vllm==0.32.0" in install
