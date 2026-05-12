"""pytest fixtures for JLegal-ChatBot tests."""
import sys
from pathlib import Path

# Add project root to path so 'src.*' imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


@pytest.fixture(scope="session")
def project_root():
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def db_path(project_root):
    return project_root / "jlegal.db"


@pytest.fixture(scope="session")
def vector_store_dir(project_root):
    return project_root / "vector_store_arabertv02"
