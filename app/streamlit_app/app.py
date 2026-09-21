"""V3 deployment entry point.

Run from this directory with ``streamlit run app.py``. The implementation lives
in the project root so the command-line trainer and UI share one code path.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import ml_pipeline as project_pipeline  # noqa: E402

# The deployment entry point can stay alive across Streamlit reruns. Reload the
# shared pipeline before the UI so newly added contracts are not hidden by the
# Python module cache.
project_pipeline = importlib.reload(project_pipeline)

# Import the project-root app by absolute file path. A plain ``import app``
# resolves back to this wrapper because Streamlit puts ``streamlit_app`` first
# on ``sys.path``, which causes a recursive self-import and hides ``main``.
PROJECT_APP_MODULE = "_alzheimer_project_app"
project_app_spec = importlib.util.spec_from_file_location(
    PROJECT_APP_MODULE,
    PROJECT_ROOT / "app.py",
)
if project_app_spec is None or project_app_spec.loader is None:
    raise ImportError(f"Unable to load project app from {PROJECT_ROOT / 'app.py'}")
project_app = sys.modules.get(PROJECT_APP_MODULE)
if project_app is None:
    project_app = importlib.util.module_from_spec(project_app_spec)
    sys.modules[PROJECT_APP_MODULE] = project_app
    project_app_spec.loader.exec_module(project_app)
else:
    project_app.__spec__ = project_app_spec
    project_app_spec.loader.exec_module(project_app)
main = project_app.main


if __name__ == "__main__":
    main()
