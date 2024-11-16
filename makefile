build:
	python3.10 -m venv venv
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install --upgrade pip setuptools wheel

collect_bc:
	venv/bin/python -m data.collect_bc

replay_bc:
	venv/bin/python -m data.replay_bc

stats:
	venv/bin/python -m utils.stats
