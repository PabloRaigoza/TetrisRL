build:
	python3.10 -m venv venv
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install --upgrade pip setuptools wheel

collect_bc:
	venv/bin/python -m data.collect_bc
