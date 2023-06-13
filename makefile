install:
	pip install -r requirements.txt

lint:
	pylint src --output-format=parseable --output=linting_results.txt