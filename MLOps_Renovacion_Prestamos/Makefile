install:
	python -m pip install -r requirements.txt
train:
	python -m src.train --mlflow
test:
	pytest -q
serve:
	uvicorn src.api:app --reload
monitor:
	python -m src.monitor --data "data/raw/Dataset Renovacion_prestamo.csv"
