build:
	python3 -m venv venv
	venv/bin/pip3 install -r requirements.txt

run_bc_collect:
	venv/bin/python3 bc_collect.py