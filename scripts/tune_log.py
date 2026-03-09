"""Shared logging helper for tune scripts.

Tees all print output to both stdout and a timestamped log file.
Usage:
    from tune_log import setup_log
    log_path = setup_log("orb")  # creates results/orb/tune_YYYY-MM-DD.log
"""

import sys
from datetime import datetime
from pathlib import Path


class TeeWriter:
    """Writes to both stdout and a log file, with timestamps."""

    def __init__(self, log_file, original_stdout):
        self._log = log_file
        self._stdout = original_stdout
        self._at_line_start = True

    def write(self, text):
        self._stdout.write(text)
        self._stdout.flush()
        # Add timestamps at the start of each line in the log
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                self._log.write("\n")
                self._at_line_start = True
            if line:
                if self._at_line_start:
                    ts = datetime.now().strftime("%H:%M:%S")
                    self._log.write(f"[{ts}] {line}")
                else:
                    self._log.write(line)
                self._at_line_start = False
        self._log.flush()

    def flush(self):
        self._stdout.flush()
        self._log.flush()


def setup_log(strategy_name: str) -> Path:
    """Set up tee logging for a tune script.

    Creates results/<strategy>/tune_YYYY-MM-DD.log and redirects
    sys.stdout to write to both console and the log file.

    Returns the log file path.
    """
    out_dir = Path("results") / strategy_name
    out_dir.mkdir(parents=True, exist_ok=True)

    run_date = datetime.now().strftime("%Y-%m-%d")
    log_path = out_dir / f"tune_{run_date}.log"

    # Append if same-day re-run (log accumulates, JSON gets _2 suffix)
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n{'=' * 60}\n")
    log_file.write(f"Run started: {datetime.now().isoformat()}\n")
    log_file.write(f"{'=' * 60}\n")
    log_file.flush()

    sys.stdout = TeeWriter(log_file, sys.__stdout__)
    return log_path
