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

expert:
	venv/bin/python -m data.expert --attempts 100 --threads 16

replay:
	venv/bin/python -m data.replay

stats:
	venv/bin/python -m data.stats


# Training
bc:
	venv/bin/python -m methods.bc --model AgentM4 --grouped true --epochs 25000 --val_freq 100

dagger:
	venv/bin/python -m methods.dagger --epochs 5000 --val_freq 100

reinforce:
	venv/bin/python -m methods.reinforce --model AgentM2 --epochs 100 --val_freq 5

test:
	venv/bin/python -m methods.test --grouped true --agent agents/M3_BC50000.dat --model AgentM3 --attempts 10
