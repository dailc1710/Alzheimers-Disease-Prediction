$ErrorActionPreference = "Stop"

python -m pip install -r requirements-dev.txt
python train_pipeline.py
python scripts/verify_artifact.py
python -m pytest --cov=ml_pipeline --cov-report=term-missing --cov-fail-under=80
ruff check .
mypy ml_pipeline.py scripts --ignore-missing-imports --follow-imports=skip

Write-Host "Completed: retraining, artifact verification, tests, lint, and type checking."
Write-Host "Now run: python scripts/update_report.py"
