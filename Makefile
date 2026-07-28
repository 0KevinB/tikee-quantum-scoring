.PHONY: setup verify generate fidelity experiment multiseed app test clean

setup:
	python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt

verify:            ## F0: los 4 solucionadores coinciden en un QUBO de juguete
	.venv/bin/pytest tests/test_solvers_agree.py -v

generate:
	.venv/bin/python -m tikee.cli generate --seed 42 --n 8000

fidelity:
	.venv/bin/python -m tikee.cli fidelity --synthesizer all

experiment:
	.venv/bin/python -m tikee.cli all

multiseed:
	.venv/bin/python -m tikee.cli multiseed --seeds 42,101,202,303,404,505,606,707,808,909

app:
	.venv/bin/streamlit run app/Inicio.py

test:
	.venv/bin/pytest -q

clean:
	rm -rf data/interim/*.parquet data/processed/*.parquet reports/figures/*.png reports/cache/*
