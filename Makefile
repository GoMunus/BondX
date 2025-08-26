# BondX Makefile
# Quality assurance and testing targets

.PHONY: help quality test test-unit test-integration test-coverage lint format clean install

help:  ## Show this help message
	@echo "BondX Development Commands:"
	@echo ""
	@echo "Quality & Testing:"
	@echo "  quality          Run complete quality assurance pipeline"
	@echo "  test            Run all tests (unit + integration)"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-coverage   Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint            Run linting checks"
	@echo "  format          Format code with black/isort"
	@echo ""
	@echo "Development:"
	@echo "  install         Install dependencies"
	@echo "  clean           Clean build artifacts"
	@echo ""

# Quality assurance pipeline
quality:  ## Run complete quality assurance pipeline
	@echo "Running BondX Quality Assurance Pipeline..."
	@echo "=========================================="
	@echo ""
	@echo "Step 1: Running validators..."
	python -m bondx.quality.run --data-root data/synthetic --output-dir quality/reports --verbose
	@echo ""
	@echo "Step 2: Running quality gates..."
	@echo "Quality gates completed. Check quality/reports/ for results."
	@echo ""
	@echo "Step 3: Generating metrics..."
	@echo "Metrics collection completed."
	@echo ""
	@echo "Quality pipeline completed successfully!"
	@echo "Reports available in: quality/reports/"
	@echo "Last run report: quality/reports/last_run_report.json"

# Testing targets
test: test-unit test-integration  ## Run all tests

test-unit:  ## Run unit tests
	@echo "Running unit tests..."
	python -m pytest tests/unit/ -v --tb=short

test-integration:  ## Run integration tests
	@echo "Running integration tests..."
	python -m pytest tests/integration/ -v --tb=short

test-coverage:  ## Run tests with coverage report
	@echo "Running tests with coverage..."
	python -m pytest tests/ --cov=bondx/quality --cov-report=html --cov-report=term-missing --cov-fail-under=85
	@echo ""
	@echo "Coverage report generated in htmlcov/"

# Code quality targets
lint:  ## Run linting checks
	@echo "Running linting checks..."
	flake8 bondx/ tests/ --max-line-length=100 --ignore=E203,W503
	black --check bondx/ tests/
	isort --check-only bondx/ tests/

format:  ## Format code with black/isort
	@echo "Formatting code..."
	black bondx/ tests/
	isort bondx/ tests/

# Development targets
install:  ## Install dependencies
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .

clean:  ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	@echo "Cleanup completed."

# CI/CD targets
ci-test:  ## Run CI test suite
	@echo "Running CI test suite..."
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .
	@echo "Running linting..."
	$(MAKE) lint
	@echo "Running unit tests..."
	$(MAKE) test-unit
	@echo "Running integration tests..."
	$(MAKE) test-integration
	@echo "Running coverage check..."
	$(MAKE) test-coverage
	@echo "CI test suite completed successfully!"

ci-quality:  ## Run CI quality checks
	@echo "Running CI quality checks..."
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	pip install -e .
	@echo "Running quality pipeline..."
	$(MAKE) quality
	@echo "CI quality checks completed successfully!"

# Documentation targets
docs:  ## Generate documentation
	@echo "Generating documentation..."
	@echo "Documentation generation not yet implemented."

# Docker targets
docker-build:  ## Build Docker image
	@echo "Building Docker image..."
	docker build -t bondx:latest .

docker-run:  ## Run Docker container
	@echo "Running Docker container..."
	docker run -p 8000:8000 bondx:latest

# Development server targets
dev-server:  ## Start development server
	@echo "Starting development server..."
	python -m bondx.main

# Database targets
db-migrate:  ## Run database migrations
	@echo "Running database migrations..."
	@echo "Database migrations not yet implemented."

db-seed:  ## Seed database with test data
	@echo "Seeding database with test data..."
	@echo "Database seeding not yet implemented."

# Monitoring targets
monitor:  ## Start monitoring dashboard
	@echo "Starting monitoring dashboard..."
	@echo "Monitoring dashboard not yet implemented."

# Backup targets
backup:  ## Create backup
	@echo "Creating backup..."
	@echo "Backup functionality not yet implemented."

# Restore targets
restore:  ## Restore from backup
	@echo "Restoring from backup..."
	@echo "Restore functionality not yet implemented."

# Quality Assurance System Targets
.PHONY: golden-datasets golden-validate golden-baseline policy-drift regulator-pack shadow-run quality-dashboard

# Generate golden datasets for quality testing
golden-datasets:
	@echo "Generating golden datasets..."
	cd data/golden && python generate_golden_datasets.py

# Validate golden datasets against baselines
golden-validate:
	@echo "Validating golden datasets..."
	python scripts/golden/validate_golden.py --verbose

# Update golden dataset baselines (requires approval)
golden-baseline:
	@echo "Updating golden dataset baselines..."
	@echo "Usage: python scripts/golden/update_baseline.py --dataset <dataset> --approve --reason '<reason>' --reviewer '<name>'"

# Run policy drift detection
policy-drift:
	@echo "Running policy drift detection..."
	python -m bondx.quality.policy_monitor --summary --verbose

# Generate regulator evidence pack
regulator-pack:
	@echo "Generating regulator evidence pack..."
	python -m bondx.reporting.regulator.evidence_pack --input quality/last_run_report.json --out reports/regulator/ --mode strict

# Run shadow comparison (requires real and synthetic data)
shadow-run:
	@echo "Running shadow comparison..."
	@echo "Usage: python -m bondx.quality.shadow_runner --real <real_data> --synthetic <synthetic_data> --out quality/shadow"

# Generate quality dashboard metrics
quality-dashboard:
	@echo "Generating quality dashboard metrics..."
	python -m bondx.quality.metrics_exporter --input quality/last_run_report.json --out quality/metrics --grafana-dashboard

# Run complete quality assurance pipeline
quality-pipeline: golden-validate policy-drift quality-dashboard
	@echo "Quality assurance pipeline completed"

# Install quality system dependencies
quality-deps:
	@echo "Installing quality system dependencies..."
	pip install pyyaml pandas numpy

# Test quality system components
quality-test:
	@echo "Running quality system tests..."
	python -m pytest tests/golden/ -v
	python -m pytest tests/quality/ -v
