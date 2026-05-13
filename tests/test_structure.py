"""Tests that verify the project structure and required files exist."""
import pytest


def test_project_root_exists(project_root):
    assert project_root.exists()
    assert project_root.is_dir()


def test_src_package_exists(project_root):
    src = project_root / "src"
    assert src.exists()
    assert (src / "__init__.py").exists()


def test_required_source_files(project_root):
    src = project_root / "src"
    required = ["embedder.py", "vector_store.py", "retriever.py",
                "generator.py", "pipeline.py", "database.py", "ingest.py"]
    for name in required:
        assert (src / name).exists(), f"Missing required source file: {name}"


def test_app_entrypoint_exists(project_root):
    assert (project_root / "app.py").exists()
    assert (project_root / "run_ingestion.py").exists()


def test_requirements_exists(project_root):
    assert (project_root / "requirements.txt").exists()


def test_all_eighteen_law_pdfs_present(project_root):
    expected = [
        "labor.pdf", "commercial.pdf", "personal_status.pdf",
        "personal_status_2019.pdf", "cybercrime.pdf", "civil_service.pdf",
        "civil_status.pdf", "hr_system.pdf", "traffic.pdf", "penal_code.pdf",
        "social_security.pdf", "income_tax.pdf", "landlord_tenant.pdf",
        "consumer_protection.pdf", "investment.pdf", "companies.pdf",
        "constitution.pdf", "customs.pdf",
    ]
    for pdf in expected:
        assert (project_root / pdf).exists(), f"Missing PDF: {pdf}"
