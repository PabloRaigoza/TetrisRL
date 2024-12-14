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
testall:
	venv/bin/python -m methods.testall


# Training
bc:
	venv/bin/python -m methods.bc --model AgentM4 --grouped true --epochs 25000 --val_freq 100

dagger:
	venv/bin/python -m methods.dagger --model AgentM4 --agent agents/M4_1DA25000.dat --epochs 10000 --val_freq 100

reinforce:
	venv/bin/python -m methods.reinforce --model AgentM4 --epochs 30 --val_freq 5 --grouped true
# venv/bin/python -m methods.reinforce --model AgentM4 --agent agents/M4_4DA10000.dat --epochs 100 --val_freq 5 --grouped true

reinforce-avg:
	venv/bin/python -m methods.reinforce-avg --model AgentM4 --agent agents/M4_BC25000.dat --epochs 100 --val_freq 5 --grouped true --mix_weight 0.1

test:
	venv/bin/python -m methods.test --agent agents/M4_with0.2.dat --model AgentM4 --attempts 1 --grouped true
# venv/bin/python -m methods.test --agent agents/M2_BC10000.dat --model AgentM2 --attempts 25
# venv/bin/python -m methods.test --grouped true --agent agents/M4_BC25000.dat --model AgentM4 --attempts 25
