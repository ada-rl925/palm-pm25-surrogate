import csv
from pathlib import Path


class CSVLogger:
    """Append-mode CSV logger. Creates header on first log() call."""

    def __init__(self, path):
        self.path = Path(path)
        self._file = None
        self._writer = None

    def log(self, row: dict):
        if self._file is None:
            self._file = open(self.path, 'w', newline='')
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
