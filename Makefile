PYTHON ?= python3.12
PIP ?= $(PYTHON) -m pip
UVICORN ?= $(PYTHON) -m uvicorn

.PHONY: install test lint cli-smoke api-smoke docker-build docker-run reproduce-results

install:
	$(PIP) install -e .

test:
	$(PYTHON) -m unittest tests.test_final_imo_lock tests.test_final_imo_release tests.test_proofgrade_runtime tests.test_proofgrade_api

lint:
	$(PYTHON) -m py_compile proofgrade/*.py analysis/*.py tests/test_final_imo_lock.py tests/test_final_imo_release.py tests/test_proofgrade_runtime.py tests/test_proofgrade_api.py

cli-smoke:
	$(PYTHON) -m proofgrade.cli version

api-smoke:
	$(UVICORN) proofgrade.api:app --host 127.0.0.1 --port 8001

docker-build:
	docker build -t proofgrade:0.1.0 .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env proofgrade:0.1.0

reproduce-results:
	$(PYTHON) analysis/build_imo_result_tables.py --config configs/baseline_freeze/final_imo_release.yaml
	$(PYTHON) analysis/build_imo_casebook.py --config configs/baseline_freeze/final_imo_release.yaml
