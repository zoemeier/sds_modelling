"""Logging helpers and the shared console output format.

Every script prints:
    header()   once at the start: title + main settings
    section()  at the start of each step
    done()     once at the end
Detail lines are indented by two spaces (four for sub-details).
`out` is print for plain scripts or logger.info for scripts that also log to file.
"""

import sys
from pathlib import Path

from sds.paths import project_relative

RULE_WIDTH = 70


def header(title, info=None, out=print):
    out("=" * RULE_WIDTH)
    out(title)
    out("=" * RULE_WIDTH)
    if info:
        width = max(len(k) for k in info)
        for k, v in info.items():
            out(f"  {k:<{width}} : {v}")


def section(title, out=print):
    out("")
    out(title)
    out("-" * RULE_WIDTH)


def done(result=None, out=print):
    """Final line; a path is shown relative to the project root."""
    if isinstance(result, Path):
        result = project_relative(result)
    out("")
    out(f"Done: {result}" if result else "Done")


class TeeLogger:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, logpath: Path):
        self.terminal = sys.stdout
        self.logfile  = open(logpath, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()
