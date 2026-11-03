"""Tests for the window that the mean power covers."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from furiosa_perf.benchmark.vllm import VllmPerformanceBenchmark
from furiosa_perf.runner.monitor import HardwareMonitor, HardwareMonitorData

START = datetime(2026, 11, 3, 12, 0, 0, tzinfo=UTC)

STDOUT = """\
Warming up with 1 requests...
Warmup run completed.
Starting main benchmark run...
============ Serving Benchmark Result ============
Successful requests:                     24
Failed requests:                         0
Benchmark duration (s):                  10.00
"""


def test_window_covers_only_the_timed_requests() -> None:
    start, end = VllmPerformanceBenchmark._power_window(STDOUT, START, "proc-start", "proc-end")
    assert datetime.fromisoformat(start) == START
    assert datetime.fromisoformat(end) == START + timedelta(seconds=10)


def test_window_falls_back_to_the_process_without_markers() -> None:
    assert VllmPerformanceBenchmark._power_window("no markers", None, "a", "b") == ("a", "b")
    assert VllmPerformanceBenchmark._power_window(STDOUT, None, "a", "b") == ("a", "b")
    assert VllmPerformanceBenchmark._power_window("Starting main benchmark run...", START, "a", "b") == ("a", "b")


def test_idle_seconds_before_the_timed_requests_are_left_out(tmp_path: Path) -> None:
    # 5 idle seconds at 0 W while vLLM bench loads, then 10 seconds at 400 W.
    csv = tmp_path / "monitor.csv"
    with csv.open("w") as f:
        f.write(HardwareMonitorData.header())
        for i in range(-5, 11):
            row = HardwareMonitorData()
            row.timestamp = (START + timedelta(seconds=i)).isoformat(timespec="microseconds")
            row.power_consumption = 0 if i < 0 else 400
            f.write(str(row))

    whole = HardwareMonitor.get_benchmark_power_summary(
        csv_file_path=csv,
        start_dt=(START - timedelta(seconds=5)).isoformat(),
        end_dt=(START + timedelta(seconds=10)).isoformat(),
        target_csv_file_path=None,
    )
    start, end = VllmPerformanceBenchmark._power_window(STDOUT, START, "unused", "unused")
    timed = HardwareMonitor.get_benchmark_power_summary(
        csv_file_path=csv, start_dt=start, end_dt=end, target_csv_file_path=None
    )

    assert whole["mean_power"] < 300
    assert timed["mean_power"] == 400
