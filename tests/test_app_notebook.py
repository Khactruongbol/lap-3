import importlib
import json
from pathlib import Path

from src.reporting import write_notebook_stub


def test_app_imports_without_training_side_effects():
    app = importlib.import_module("app")
    assert hasattr(app, "artifact_paths")


def test_final_notebook_contains_required_sections():
    write_notebook_stub()
    notebook_path = Path("notebooks/99_customer_segmentation_workflow.ipynb")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for section in [
        "Define Problem",
        "Raw Data Sources",
        "Data Cleaning and Validation",
        "Train Clustering Models",
        "Python UI",
        "Final Conclusion",
    ]:
        assert section in text
