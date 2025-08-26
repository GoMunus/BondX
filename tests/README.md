# BondX Testing Framework

This directory contains the comprehensive testing framework for the BondX quality assurance system, including unit tests, integration tests, and test fixtures.

## Overview

The testing framework is designed to ensure:
- **Deterministic results** with fixed seeds for reproducible testing
- **Comprehensive coverage** of all quality components
- **Policy-driven testing** for different regulatory modes
- **Performance validation** for large datasets
- **CI/CD integration** with coverage gates

## Test Structure

```
tests/
├── unit/                          # Unit tests for individual components
│   ├── test_validators.py        # Data validation tests (24 tests)
│   ├── test_quality_gates.py     # Quality gate tests (24 tests)
│   └── test_metrics.py           # Metrics collection tests (24 tests)
├── integration/                   # Integration tests for full pipeline
│   └── test_pipeline.py          # End-to-end pipeline tests (10 tests)
└── fixtures/                      # Test data and baselines
    ├── small_perfect_dataset/     # Clean dataset with no violations
    ├── dataset_with_known_errors/ # Dataset with intentional issues
    ├── mixed_dataset/             # Combination of good/problematic records
    ├── baselines/                 # Baseline metrics for drift testing
    ├── generate_fixtures.py      # Script to generate test fixtures
    └── fixtures_metadata.json    # Metadata about test datasets
```

## Running Tests

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

2. Install test dependencies:
```bash
pip install pytest pytest-cov flake8 black isort
```

### Quick Start

Run all tests:
```bash
make test
```

Run quality pipeline:
```bash
make quality
```

### Individual Test Suites

#### Unit Tests
```bash
# Run all unit tests
make test-unit

# Run specific unit test file
python -m pytest tests/unit/test_validators.py -v

# Run specific test method
python -m pytest tests/unit/test_validators.py::TestDataValidator::test_validate_bond_data_success -v
```

#### Integration Tests
```bash
# Run all integration tests
make test-integration

# Run specific integration test file
python -m pytest tests/integration/test_pipeline.py -v

# Run specific test method
python -m pytest tests/integration/test_pipeline.py::TestFullQualityPipeline::test_complete_pipeline_perfect_data -v
```

#### Coverage Testing
```bash
# Run tests with coverage report
make test-coverage

# Coverage will fail if below 85%
# HTML report generated in htmlcov/
```

### Test Execution Options

#### Verbose Output
```bash
python -m pytest tests/ -v
```

#### Stop on First Failure
```bash
python -m pytest tests/ -x
```

#### Run Tests in Parallel
```bash
python -m pytest tests/ -n auto
```

#### Generate JUnit XML Report
```bash
python -m pytest tests/ --junitxml=test-results.xml
```

## Test Fixtures

### Generated Test Datasets

The test fixtures are generated deterministically with a fixed seed (42) to ensure reproducible results:

- **small_perfect_dataset/**: Clean dataset with no quality violations
- **dataset_with_known_errors/**: Dataset with intentional quality issues for testing
- **mixed_dataset/**: Combination of clean and problematic records
- **baselines/**: Baseline metrics for drift testing

### Regenerating Fixtures

To regenerate test fixtures:
```bash
cd tests/fixtures
python generate_fixtures.py
```

**Note**: Fixtures are generated with seed 42 for deterministic testing. Changing the seed will affect test results.

### Fixture Metadata

Each fixture includes metadata describing:
- Expected validation results
- Known quality issues
- Record counts and characteristics
- Generation timestamp and seed

## Test Categories

### Unit Tests (72 total)

#### Validators (`test_validators.py`)
- **Critical Field Validation**: Missing required fields, field presence
- **Data Range Validation**: Bounds checking for numeric fields
- **Categorical Validation**: Rating buckets, sector values, currency codes
- **Chronological Validation**: Maturity dates, issue dates, timestamps
- **Duplicate Detection**: Primary keys, compound keys
- **Staleness Validation**: Data freshness, quote age
- **Edge Cases**: Boundary values, empty datasets, single records
- **Performance**: Large dataset handling

#### Quality Gates (`test_quality_gates.py`)
- **Policy Loading**: Default vs custom policies, error handling
- **Gate Evaluation**: Coverage, ESG completeness, liquidity index
- **Mode Switching**: Strict vs exploratory ESG modes
- **Sector Adjustment**: Global vs sector-specific liquidity thresholds
- **Severity Mapping**: FAIL/WARN/INFO behavior
- **Environment Overrides**: Development, testing, production modes
- **Regulator Mode**: Enhanced reporting, audit trails
- **Policy Validation**: Threshold ranges, configuration integrity

#### Metrics (`test_metrics.py`)
- **Coverage Calculation**: Overall and per-column coverage
- **Freshness Metrics**: Staleness detection, configurable thresholds
- **Distribution Statistics**: Mean, std, min, max, median
- **Drift Detection**: Baseline comparison, change measurement
- **PSD-Safe Covariance**: Matrix validation, numerical stability
- **Sector Analysis**: Per-sector coverage and metrics
- **Performance Testing**: Large dataset scalability
- **Export/Import**: JSON serialization, baseline management

### Integration Tests (10 total)

#### Pipeline Tests (`test_pipeline.py`)
- **Complete Pipeline**: End-to-end quality assurance workflow
- **Multiple Datasets**: Perfect, error, and mixed data processing
- **Deterministic Results**: Fixed seed reproducibility
- **Report Generation**: JSON output, console summaries
- **Policy Behavior**: Different policy modes and outcomes
- **Error Handling**: Graceful failure and recovery
- **Performance**: Large dataset processing time
- **Output Validation**: Schema compliance, data integrity

## Expected Runtime

### Unit Tests
- **Validators**: ~2-3 seconds
- **Quality Gates**: ~2-3 seconds  
- **Metrics**: ~3-4 seconds
- **Total Unit**: ~7-10 seconds

### Integration Tests
- **Pipeline Tests**: ~5-8 seconds
- **Total Integration**: ~5-8 seconds

### Full Test Suite
- **Complete Run**: ~12-18 seconds
- **With Coverage**: ~15-20 seconds

## Seed Setup

### Deterministic Testing

All tests use a fixed seed (42) for reproducible results:

```python
@pytest.fixture(autouse=True)
def setup_seed(self):
    """Set deterministic seed for all tests."""
    np.random.seed(42)
```

### Why Fixed Seeds?

1. **Reproducibility**: Same inputs always produce same outputs
2. **Debugging**: Failures can be reproduced exactly
3. **CI Consistency**: Tests behave identically across environments
4. **Regression Detection**: Changes in behavior are immediately apparent

### Changing Seeds

To test with different seeds:
1. Modify the `setup_seed` fixture in test classes
2. Update fixture generation script (`generate_fixtures.py`)
3. Regenerate test fixtures
4. Update baseline metrics if needed

## Adding New Tests

### Unit Test Guidelines

1. **Test Structure**:
```python
def test_feature_name(self, fixture_name):
    """Test description of what is being tested."""
    # Arrange: Set up test data and conditions
    test_data = self.fixture_name()
    
    # Act: Execute the functionality being tested
    result = self.component.method_name(test_data)
    
    # Assert: Verify expected outcomes
    assert result.expected_property == expected_value
    assert len(result.items) == expected_count
```

2. **Naming Convention**:
   - Test methods: `test_<feature>_<scenario>`
   - Fixtures: descriptive names like `sample_data`, `error_dataset`
   - Classes: `Test<ComponentName>`

3. **Coverage Requirements**:
   - Each validation rule should have at least one test
   - Edge cases and error conditions must be covered
   - Performance tests for large datasets
   - Policy variations and mode switching

### Integration Test Guidelines

1. **Test Structure**:
```python
def test_pipeline_feature(self, dataset_fixture, baseline_fixture):
    """Test complete pipeline with specific scenario."""
    # Run full pipeline
    report = self._run_pipeline(dataset_fixture, "test_name", baseline_fixture)
    
    # Verify complete output structure
    assert "run_id" in report
    assert "status" in report
    assert "metrics" in report
    
    # Verify specific behavior
    assert report["status"] == "EXPECTED_STATUS"
    assert report["metrics"]["coverage"] >= expected_coverage
```

2. **Pipeline Testing**:
   - Test complete workflows, not individual components
   - Verify output schemas and data integrity
   - Test error handling and recovery
   - Validate performance characteristics

### Fixture Guidelines

1. **Data Characteristics**:
   - Realistic but controlled test data
   - Known quality issues for testing
   - Sufficient size for performance testing
   - Representative of production scenarios

2. **Baseline Management**:
   - Deterministic generation
   - Version control for changes
   - Documentation of expected values
   - Migration scripts for updates

## CI/CD Integration

### Coverage Gates

Tests will fail if coverage drops below 85%:
```bash
make test-coverage
# Coverage report generated in htmlcov/
# Build fails if coverage < 85%
```

### CI Pipeline Steps

1. **Install Dependencies**: `pip install -r requirements.txt`
2. **Linting**: `make lint` (flake8, black, isort)
3. **Unit Tests**: `make test-unit`
4. **Integration Tests**: `make test-integration`
5. **Coverage Check**: `make test-coverage`
6. **Quality Pipeline**: `make quality`

### CI Commands

```bash
# Full CI test suite
make ci-test

# CI quality checks
make ci-quality
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure `pip install -e .` has been run
   - Check Python path includes project root
   - Verify all dependencies are installed

2. **Fixture Errors**:
   - Regenerate fixtures: `python generate_fixtures.py`
   - Check fixture paths and permissions
   - Verify baseline files exist

3. **Coverage Failures**:
   - Run `make test-coverage` to see detailed report
   - Check `htmlcov/index.html` for uncovered lines
   - Add tests for uncovered functionality

4. **Performance Issues**:
   - Tests should complete within expected timeframes
   - Large dataset tests may need adjustment
   - Check for infinite loops or inefficient algorithms

### Debug Mode

Run tests with debug output:
```bash
python -m pytest tests/ -v -s --tb=long
```

### Test Isolation

Ensure tests don't interfere with each other:
```bash
python -m pytest tests/ --strict-markers --disable-warnings
```

## Best Practices

1. **Test Independence**: Each test should be able to run in isolation
2. **Deterministic Results**: Use fixed seeds and controlled test data
3. **Comprehensive Coverage**: Test both success and failure scenarios
4. **Performance Validation**: Ensure tests complete in reasonable time
5. **Clear Assertions**: Test one concept per test method
6. **Documentation**: Clear test names and docstrings
7. **Maintenance**: Keep tests up to date with code changes

## Contributing

When adding new tests:

1. Follow existing naming conventions
2. Ensure deterministic behavior
3. Add appropriate fixtures if needed
4. Update this documentation
5. Verify coverage requirements
6. Test in isolation and as part of suite

## Support

For testing framework questions:
- Check this documentation first
- Review existing test examples
- Consult the pytest documentation
- Review BondX quality system documentation
