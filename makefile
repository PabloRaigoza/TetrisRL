# Environment Setup
build:
	python3.10 -m venv venv
	venv/bin/pip install --upgrade pip setuptools wheel
	venv/bin/pip install -r requirements.txt

clean:
	rm -rf __pycache__/ */__pycache__/ */*/__pycache__/


# Data Collection
collect:
	venv/bin/python -m data.collect

replay:
	venv/bin/python -m data.replay

stats:
	venv/bin/python -m data.stats


# Training
bc:
	venv/bin/python -m methods.bc --epochs 100 --val_freq 100

test:
	venv/bin/python -m methods.test --agent agents/M2_BC10000.dat --attempts 10
