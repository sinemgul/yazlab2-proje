"""Entry point for ``python -m src.main``."""

from src.config import CONFIG
from src.pipeline.runner import run_experiments


if __name__ == "__main__":
    run_experiments(CONFIG)
