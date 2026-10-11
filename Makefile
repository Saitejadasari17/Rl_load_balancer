.PHONY: install run-backend run-frontend train evaluate clean

install:
	pip install -r requirements.txt
	cd frontend && npm install

run-backend:
	python backend/main.py

run-frontend:
	cd frontend && npm run dev

train:
	python main.py --mode train --episodes 300

evaluate:
	python main.py --mode evaluate

all:
	python main.py --mode all

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf results/ logs/ models/

test:
	pytest tests/ -v
