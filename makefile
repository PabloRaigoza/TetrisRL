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
	venv/bin/python -m methods.dagger --model AgentM4 --agent agents/M4_1DA25000.dat --epochs 10000 --val_freq 100

reinforce:
	venv/bin/python -m methods.reinforce --model AgentM3 --agent agents/M3_BC50000.dat --epochs 100 --val_freq 5 --grouped true

test:
	venv/bin/python -m methods.test --grouped true --agent agents/M4_4DA10000.dat --model AgentM4 --attempts 10
