"""Project directory paths."""

from pathlib import Path


class ProjectPaths:
    """Project directory paths."""

    def __init__(self, root: Path | None = None):
        self.root = root or Path(__file__).parent.parent.parent.parent
        self.src = self.root / "src"
        self.tests = self.root / "tests"
        self.logs = self.root / "logs"
        self.results = self.root / "results"
        self.data = self.root / "data"
        self.experiments = self.root / "experiments"

        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for path in [self.logs, self.results, self.data, self.experiments]:
            path.mkdir(parents=True, exist_ok=True)
