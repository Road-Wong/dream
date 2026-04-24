"""project_paths.py - Path constants for the minimal DREAM simulation environment."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = RESULTS_DIR / "logs"


def ensure_results_layout() -> None:
    for path in (RESULTS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
