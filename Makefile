# DiabeDose Makefile
# Automation for common tasks

.PHONY: help install setup test run clean deploy docker-build docker-run

# Variables
PYTHON = python3
PIP = pip3
STREAMLIT = streamlit
PORT = 8501
HOST = localhost

help:
	@echo "DiabeDose - Insulin Dose Predictor"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup       - Setup project (generate data, train model)"
	@echo "  make test        - Run unit tests"
	@echo "  make run         - Run web application"
	@echo "  make clean       - Clean generated files"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run in Docker container"
	@echo "  make deploy      - Deploy to cloud (simulated)"
	@echo "  make help        - Show this help message"

install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt

setup: install
	@echo "Setting up project..."
	$(PYTHON) -c "from utils.data_generator import DiabetesDataGenerator; \
	              g = DiabetesDataGenerator(); \
	              df = g.generate_patient_data(1000); \
	              g.save_to_csv(df, 'data/dummy_data.csv')"
	@echo "✓ Generated sample data"
	$(PYTHON) models/model_training.py
	@echo "✓ Trained model"

test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v

run:
	@echo "Starting DiabeDose..."
	$(STREAMLIT) run app.py --server.port $(PORT) --server.address $(HOST)

clean:
	@echo "Cleaning generated files..."
	rm -rf data/*.csv
	rm -rf models/*.pkl
	rm -rf models/*.json
	rm -rf models/*.txt
	rm -rf __pycache__
	rm -rf */__pycache__
	rm -rf .pytest_cache
	rm -rf .streamlit
	@echo "✓ Cleaned"

docker-build:
	@echo "Building Docker image..."
	docker build -t diabedose:latest .

docker-run:
	@echo "Running in Docker..."
	docker run -p $(PORT):8501 diabedose:latest

deploy:
	@echo "Deployment simulation..."
	@echo "This would deploy to cloud in a real scenario"
	@echo "For actual deployment, configure your cloud provider"

# Development shortcuts
dev: install setup run

all: install setup test run

# Data generation shortcut
data:
	$(PYTHON) -c "from utils.data_generator import DiabetesDataGenerator; \
	              g = DiabetesDataGenerator(); \
	              df = g.generate_patient_data(500); \
	              g.save_to_csv(df, 'data/dummy_data.csv')"

# Model retraining shortcut
retrain:
	$(PYTHON) models/model_training.py

# Quick run (assumes setup already done)
quick:
	$(STREAMLIT) run app.py

# Backup current state
backup:
	@echo "Creating backup..."
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		--exclude='.git' \
		.
	@echo "✓ Backup created"

# Update dependencies
update:
	@echo "Updating dependencies..."
	$(PIP) freeze > requirements_backup.txt
	$(PIP) install --upgrade -r requirements.txt
	@echo "✓ Dependencies updated"

# Format code
format:
	@echo "Formatting code..."
	black .
	isort .
	@echo "✓ Code formatted"

# Lint code
lint:
	@echo "Linting code..."
	flake8 .
	pylint app.py utils/ models/ --exit-zero
	@echo "✓ Linting complete"