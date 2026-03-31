.PHONY: install serve validate test test-unit test-all test-llm

install:
	pip install -r requirements.txt

serve:
	uvicorn server.env:app --reload --host 0.0.0.0 --port 7860

validate:
	openenv validate .

test-unit:
	pytest tests/ -v

test-llm:
	python inference.py

test-all: test-unit test-llm

docker-build:
	docker build -t openenv-cr .

docker-run:
	docker run -p 8000:8000 openenv-cr
