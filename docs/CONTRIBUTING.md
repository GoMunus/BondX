# Contributing to BondX

Thank you for your interest in contributing to BondX! This document provides guidelines for contributing to the project, including code standards, testing requirements, and the development workflow.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Quality Assurance](#quality-assurance)
- [Pull Request Process](#pull-request-process)
- [Adding New Features](#adding-new-features)
- [Documentation](#documentation)
- [Code Review](#code-review)
- [Release Process](#release-process)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda for package management
- Docker (optional, for containerized development)

### Fork and Clone

1. Fork the BondX repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/yourusername/BondX.git
cd BondX
```

3. Add the upstream remote:
```bash
git remote add upstream https://github.com/original/BondX.git
```

### Development Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

3. Install development dependencies:
```bash
pip install pytest pytest-cov flake8 black isort mypy
```

## Development Environment

### Project Structure

```
BondX/
├── bondx/                    # Main package
│   ├── quality/             # Quality assurance system
│   ├── api/                 # API endpoints
│   ├── core/                # Core functionality
│   └── ...
├── tests/                   # Test suite
│   ├── unit/               # Unit tests
│   ├── integration/        # Integration tests
│   └── fixtures/           # Test data
├── docs/                   # Documentation
├── deploy/                 # Deployment configurations
└── requirements.txt        # Dependencies
```

### Environment Configuration

Create a `.env` file for local development:

```bash
# .env
BONDX_ENV=development
BONDX_LOG_LEVEL=DEBUG
BONDX_DATA_ROOT=data/synthetic
BONDX_OUTPUT_DIR=quality/reports
```

### IDE Configuration

#### VS Code

Install recommended extensions:
- Python
- Pylance
- Python Test Explorer
- GitLens

#### PyCharm

Configure the project:
- Set Python interpreter to your virtual environment
- Configure pytest as the test runner
- Enable code inspection and formatting

## Code Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line Length**: 100 characters maximum
- **Import Order**: Standard library, third-party, local
- **Docstrings**: Google-style docstrings for all public functions
- **Type Hints**: Required for all function parameters and return values

### Code Formatting

We use automated formatting tools:

```bash
# Format code
make format

# Check formatting
make lint
```

#### Black Configuration

```toml
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py38']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
```

#### isort Configuration

```toml
# pyproject.toml
[tool.isort]
profile = "black"
line_length = 100
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `DataValidator`)
- **Functions/Methods**: snake_case (e.g., `validate_bond_data`)
- **Variables**: snake_case (e.g., `bond_data`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_RETRY_COUNT`)
- **Private Methods**: Leading underscore (e.g., `_internal_method`)

### Documentation Standards

#### Function Docstrings

```python
def validate_bond_data(data: pd.DataFrame) -> List[ValidationResult]:
    """Validate bond instrument data for quality assurance.
    
    This function performs comprehensive validation of bond data including
    schema validation, business rule validation, and data integrity checks.
    
    Args:
        data: DataFrame containing bond data to validate
        
    Returns:
        List of ValidationResult objects describing validation outcomes
        
    Raises:
        ValueError: If data is empty or malformed
        TypeError: If data is not a pandas DataFrame
        
    Example:
        >>> validator = DataValidator()
        >>> results = validator.validate_bond_data(bond_df)
        >>> len(results)
        5
    """
    pass
```

#### Class Docstrings

```python
class DataValidator:
    """Core data validator implementing enterprise-grade quality checks.
    
    The DataValidator class provides comprehensive validation capabilities
    for bond market data, including schema validation, business rule
    enforcement, and data integrity verification.
    
    Attributes:
        validation_results: List of validation results from last run
        policy: Quality policy configuration
        
    Example:
        >>> validator = DataValidator()
        >>> validator.load_policy("production.yaml")
        >>> results = validator.validate_dataset(bond_data)
    """
    
    def __init__(self):
        """Initialize the DataValidator with default configuration."""
        self.validation_results = []
        self.policy = self._load_default_policy()
```

### Error Handling

- Use specific exception types
- Include meaningful error messages
- Log errors with appropriate context
- Provide recovery suggestions when possible

```python
def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Process data with comprehensive error handling."""
    try:
        if data.empty:
            raise ValueError("DataFrame cannot be empty")
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        # Process data
        result = data.copy()
        result['processed'] = True
        
        return result
        
    except (ValueError, TypeError) as e:
        logger.error(f"Data processing failed: {e}")
        logger.info("Please ensure data is a non-empty DataFrame")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")
        raise RuntimeError(f"Data processing failed: {e}") from e
```

## Testing Requirements

### Test Coverage

**Minimum Coverage: 85%**

All new code must maintain or improve test coverage:

```bash
# Run coverage check
make test-coverage

# Coverage report will be generated in htmlcov/
```

### Test Structure

#### Unit Tests

- **Location**: `tests/unit/`
- **Naming**: `test_<module>_<function>_<scenario>`
- **Coverage**: Test all public methods and edge cases
- **Isolation**: Tests should not depend on each other

```python
def test_validate_bond_data_success(self, validator, sample_data):
    """Test successful bond data validation."""
    # Arrange
    expected_count = len(sample_data)
    
    # Act
    results = validator.validate_bond_data(sample_data)
    
    # Assert
    assert len(results) > 0
    failures = [r for r in results if r.severity == "FAIL"]
    assert len(failures) == 0
```

#### Integration Tests

- **Location**: `tests/integration/`
- **Naming**: `test_<feature>_<scenario>`
- **Coverage**: End-to-end workflows and component interaction
- **Data**: Use realistic test fixtures

```python
def test_complete_pipeline_perfect_data(self, perfect_dataset, baseline_metrics):
    """Test complete quality pipeline with clean data."""
    # Run full pipeline
    report = self._run_pipeline(perfect_dataset, "test", baseline_metrics)
    
    # Verify complete output
    assert "run_id" in report
    assert "status" in report
    assert report["status"] == "PASS"
```

### Test Fixtures

#### Creating Test Data

1. **Use the fixture generator**:
```bash
cd tests/fixtures
python generate_fixtures.py
```

2. **Add new fixtures**:
```python
@pytest.fixture
def custom_dataset(self):
    """Create custom test dataset."""
    return pd.DataFrame({
        'field1': [1, 2, 3],
        'field2': ['a', 'b', 'c']
    })
```

3. **Update fixture metadata**:
```json
{
    "custom_dataset": {
        "description": "Custom test data for specific scenarios",
        "records": 3,
        "expected_validation_results": "Should pass all validations"
    }
}
```

#### Baseline Management

1. **Create baseline metrics**:
```python
def generate_baseline_metrics():
    """Generate baseline metrics for drift testing."""
    return {
        "distribution_stats": {
            "field1": {"mean": 2.0, "std": 1.0}
        },
        "coverage_baseline": {
            "overall_coverage_pct": 100.0
        }
    }
```

2. **Version control baselines**:
- Commit baseline changes with clear messages
- Document baseline updates in changelog
- Maintain backward compatibility when possible

### Test Execution

#### Running Tests

```bash
# All tests
make test

# Unit tests only
make test-unit

# Integration tests only
make test-integration

# With coverage
make test-coverage

# Specific test file
python -m pytest tests/unit/test_validators.py -v

# Specific test method
python -m pytest tests/unit/test_validators.py::TestDataValidator::test_validate_bond_data_success -v
```

#### Test Debugging

```bash
# Run with debug output
python -m pytest tests/ -v -s --tb=long

# Run single test with debugger
python -m pytest tests/unit/test_validators.py::TestDataValidator::test_validate_bond_data_success -v -s --pdb

# Generate HTML coverage report
python -m pytest tests/ --cov=bondx/quality --cov-report=html
```

## Quality Assurance

### Code Quality Checks

Run all quality checks before submitting:

```bash
# Linting
make lint

# Formatting
make format

# Type checking
mypy bondx/ tests/

# Security scanning
bandit -r bondx/
```

### Quality Pipeline

Test the quality system itself:

```bash
# Run quality pipeline
make quality

# Verify outputs
ls -la quality/reports/
cat quality/reports/last_run_report.json
```

### Performance Testing

Ensure new code doesn't degrade performance:

```bash
# Run performance tests
python -m pytest tests/ -m "performance" -v

# Benchmark specific functions
python -m pytest tests/unit/test_metrics.py::TestMetricsCollector::test_large_dataset_performance -v
```

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**:
```bash
make test
make test-coverage
```

2. **Run quality checks**:
```bash
make lint
make format
make quality
```

3. **Update documentation**:
- Update relevant README files
- Add docstrings for new functions
- Update API documentation if applicable

### PR Checklist

- [ ] Tests pass locally
- [ ] Coverage maintained or improved (≥85%)
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Changelog entry added
- [ ] No breaking changes (or documented)
- [ ] Performance impact assessed
- [ ] Security implications reviewed

### PR Description Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed
- [ ] Performance testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Related Issues
Closes #123
Related to #456

## Screenshots (if applicable)
Add screenshots for UI changes.
```

## Adding New Features

### Feature Development Workflow

1. **Create feature branch**:
```bash
git checkout -b feature/new-validation-rule
```

2. **Implement feature**:
- Add new code with tests
- Update documentation
- Add configuration options

3. **Test thoroughly**:
```bash
make test
make test-coverage
make quality
```

4. **Update changelog**:
```markdown
# Changelog

## [Unreleased]
### Added
- New validation rule for bond maturity dates
- Configurable threshold for maturity validation
- Enhanced error reporting for validation failures
```

### Configuration Management

When adding new configuration options:

1. **Update policy schema**:
```yaml
# bondx/quality/config/quality_policy.yaml
thresholds:
  new_threshold: 50.0

severity_mapping:
  new_validation: "WARN"
```

2. **Add configuration validation**:
```python
def _validate_policy(self, policy: Dict[str, Any]) -> None:
    """Validate policy configuration."""
    if "new_threshold" in policy["thresholds"]:
        threshold = policy["thresholds"]["new_threshold"]
        if not 0 <= threshold <= 100:
            raise ValueError("new_threshold must be between 0 and 100")
```

3. **Provide defaults**:
```python
def _load_default_policy(self) -> Dict[str, Any]:
    """Load default policy configuration."""
    return {
        "thresholds": {
            "new_threshold": 50.0,  # Default value
            # ... other thresholds
        }
    }
```

### Backward Compatibility

- **Deprecation warnings**: Use warnings for deprecated features
- **Gradual migration**: Provide migration paths for breaking changes
- **Version checking**: Check configuration version and migrate if needed

```python
import warnings

def deprecated_method(self):
    """Deprecated method - use new_method instead."""
    warnings.warn(
        "deprecated_method is deprecated, use new_method instead",
        DeprecationWarning,
        stacklevel=2
    )
    return self.new_method()
```

## Documentation

### Documentation Standards

- **Clear and concise**: Explain what, why, and how
- **Examples**: Provide working code examples
- **API reference**: Complete parameter and return value documentation
- **Troubleshooting**: Common issues and solutions

### Documentation Types

#### Code Documentation

- **Docstrings**: All public functions and classes
- **Inline comments**: Complex logic explanations
- **Type hints**: Parameter and return type annotations

#### User Documentation

- **README files**: Project overview and quick start
- **User guides**: Step-by-step usage instructions
- **API documentation**: Complete API reference
- **Examples**: Sample code and use cases

#### Developer Documentation

- **Architecture**: System design and components
- **Development setup**: Environment configuration
- **Testing**: Test execution and debugging
- **Deployment**: Build and deployment processes

### Documentation Updates

When updating documentation:

1. **Update relevant files**:
- README files
- API documentation
- Configuration guides
- Changelog

2. **Verify accuracy**:
- Test code examples
- Verify configuration examples
- Check link validity

3. **Review for clarity**:
- Clear explanations
- Consistent terminology
- Logical organization

## Code Review

### Review Process

1. **Automated checks**:
- CI/CD pipeline validation
- Code coverage verification
- Style and quality checks

2. **Peer review**:
- At least one approval required
- Address all review comments
- Resolve conflicts and issues

3. **Final approval**:
- Maintainer approval for significant changes
- Documentation review for user-facing changes
- Performance review for optimization changes

### Review Guidelines

#### What to Look For

- **Correctness**: Logic and algorithm accuracy
- **Performance**: Efficiency and scalability
- **Security**: Potential vulnerabilities
- **Maintainability**: Code clarity and structure
- **Testing**: Adequate test coverage
- **Documentation**: Clear and complete documentation

#### Review Comments

- **Constructive**: Focus on improvement, not criticism
- **Specific**: Point to specific issues with examples
- **Actionable**: Provide clear guidance for fixes
- **Respectful**: Maintain professional tone

### Review Checklist

- [ ] Code follows project standards
- [ ] Tests are comprehensive and pass
- [ ] Documentation is updated
- [ ] No breaking changes (or documented)
- [ ] Performance impact is acceptable
- [ ] Security implications are addressed
- [ ] Error handling is appropriate
- [ ] Logging and monitoring are adequate

## Release Process

### Release Preparation

1. **Update version**:
```python
# bondx/__init__.py
__version__ = "1.2.0"
```

2. **Update changelog**:
```markdown
# Changelog

## [1.2.0] - 2024-01-15
### Added
- New validation rules for ESG compliance
- Enhanced quality gate configuration
- Performance improvements for large datasets

### Changed
- Updated default quality thresholds
- Improved error reporting

### Fixed
- Bug in coverage calculation for empty datasets
- Memory leak in large dataset processing
```

3. **Run full test suite**:
```bash
make ci-test
make ci-quality
```

### Release Process

1. **Create release branch**:
```bash
git checkout -b release/1.2.0
```

2. **Final testing**:
```bash
make test
make test-coverage
make quality
```

3. **Create release**:
- Tag the release: `git tag v1.2.0`
- Push tag: `git push origin v1.2.0`
- Create GitHub release with changelog

4. **Deploy**:
- Update production environment
- Monitor deployment health
- Verify functionality

### Post-Release

1. **Update documentation**:
- Update version references
- Add migration guides if needed
- Update examples and tutorials

2. **Monitor and support**:
- Watch for issues and bugs
- Provide user support
- Plan next release

## Getting Help

### Support Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and community support
- **Documentation**: Comprehensive guides and references
- **Code Examples**: Sample implementations and use cases

### Contributing Guidelines

- **Be respectful**: Maintain professional and inclusive communication
- **Follow standards**: Adhere to project coding and documentation standards
- **Test thoroughly**: Ensure code quality and test coverage
- **Document changes**: Update relevant documentation
- **Ask questions**: Don't hesitate to ask for clarification

### Recognition

Contributors are recognized through:

- **Contributor list**: GitHub contributors page
- **Changelog entries**: Credit for significant contributions
- **Documentation**: Recognition in relevant documentation
- **Community**: Acknowledgment in community discussions

## Conclusion

Thank you for contributing to BondX! Your contributions help make the system more robust, feature-rich, and valuable for the bond market community. By following these guidelines, you ensure that your contributions integrate smoothly with the existing codebase and maintain the high quality standards of the project.

If you have questions about any aspect of contributing, please don't hesitate to ask. The BondX community is here to help and support your contributions.
