# BondX Quality Assurance System

A comprehensive quality assurance system for BondX that prevents silent drift, detects policy changes, generates regulatory evidence, and provides real-time monitoring dashboards.

## 🎯 System Overview

The BondX Quality Assurance System consists of five core components:

1. **Golden Dataset Vault** - Frozen datasets with known violations and expected outcomes
2. **Policy Drift Detection** - Automated monitoring of quality policy effectiveness
3. **Regulator Evidence Packs** - Human-readable PDF reports for audit compliance
4. **Shadow Runner** - Hybrid testing between synthetic and real data
5. **Quality Dashboard** - Grafana-ready metrics and visualizations

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Quality Assurance System                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Golden Dataset  │  │ Policy Drift    │  │ Regulator      │  │
│  │ Vault           │  │ Detection       │  │ Evidence Packs │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│           │                    │                    │           │
│           ▼                    ▼                    ▼           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Shadow Runner   │  │ Quality         │  │ CI/CD          │  │
│  │ (Hybrid Test)   │  │ Dashboard       │  │ Integration    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
make quality-deps
```

### 2. Generate Golden Datasets

```bash
make golden-datasets
```

This creates three datasets:
- `v1_clean` - Perfect dataset (100% PASS)
- `v1_dirty` - Known violations (mixed PASS/WARN/FAIL)
- `v1_mixed` - Edge cases and thresholds

### 3. Run Quality Pipeline

```bash
make quality-pipeline
```

This runs:
- Golden dataset validation
- Policy drift detection
- Quality dashboard generation

## 📁 Directory Structure

```
bondx/
├── quality/
│   ├── policy_monitor.py          # Policy drift detection
│   ├── metrics_exporter.py        # Grafana metrics export
│   ├── config/
│   │   ├── quality_policy.yaml    # Quality policy configuration
│   │   └── policy_drift.yaml      # Drift detection thresholds
│   └── policy_version.txt         # Policy version tracking
├── reporting/
│   └── regulator/
│       └── evidence_pack.py       # Regulator evidence generator
└── scripts/
    └── golden/
        ├── validate_golden.py     # Golden dataset validation
        └── update_baseline.py     # Baseline update (with approval)
data/
├── golden/                        # Golden dataset vault
│   ├── v1_clean/                 # Perfect dataset
│   ├── v1_dirty/                 # Known violations
│   ├── v1_mixed/                 # Edge cases
│   └── baselines/                # Expected outputs
└── real_shadow/                  # Real data slice for shadow testing
tests/
├── golden/                        # Golden harness tests
└── quality/                       # Quality system tests
docs/
├── GOLDEN_POLICY.md              # Baseline update policy
└── POLICY_DRIFT.md               # Drift detection documentation
```

## 🔧 Core Components

### 1. Golden Dataset Vault

**Purpose**: Prevents silent drift in quality behavior by maintaining frozen baselines.

**Usage**:
```bash
# Validate all golden datasets
python scripts/golden/validate_golden.py --verbose

# Validate specific dataset
python scripts/golden/validate_golden.py --dataset v1_dirty

# Update baseline (requires approval)
python scripts/golden/update_baseline.py \
  --dataset v1_dirty \
  --approve \
  --reason "Policy update for ESG thresholds" \
  --reviewer "John Doe"
```

**Key Features**:
- Deterministic validation with fixed seeds
- Byte-for-byte baseline comparison
- Comprehensive changelog and audit trail
- CI/CD integration to block PRs with drift

### 2. Policy Drift Detection

**Purpose**: Detects when quality policies effectively loosen without explicit changes.

**Usage**:
```bash
# Run drift detection
python -m bondx.quality.policy_monitor --summary --verbose

# Export metrics for dashboards
python -m bondx.quality.policy_monitor --csv quality/policy_drift_metrics.csv
```

**Key Features**:
- Moving average analysis of quality metrics
- Configurable drift thresholds
- Policy version tracking
- Automated alerts and recommendations

### 3. Regulator Evidence Packs

**Purpose**: Generates human-readable evidence packs for regulatory compliance.

**Usage**:
```bash
# Generate strict mode evidence pack
python -m bondx.reporting.regulator.evidence_pack \
  --input quality/last_run_report.json \
  --out reports/regulator/ \
  --mode strict

# Generate exploratory mode summary
python -m bondx.reporting.regulator.evidence_pack \
  --input quality/last_run_report.json \
  --out reports/regulator/ \
  --mode exploratory
```

**Key Features**:
- Executive summary with overall status
- Detailed gate outcomes and violations
- ESG analysis with greenwashing risk assessment
- Liquidity risk analysis with TTE considerations
- Comprehensive appendices with data provenance

### 4. Shadow Runner

**Purpose**: Compares quality outcomes between synthetic and real data for validation.

**Usage**:
```bash
# Run shadow comparison
python -m bondx.quality.shadow_runner \
  --real data/real_shadow/sample.csv \
  --synthetic data/synthetic_subset/sample.csv \
  --out quality/shadow
```

**Key Features**:
- Automatic data anonymization
- Configurable tolerance thresholds
- Comprehensive delta reporting
- Risk assessment and recommendations

### 5. Quality Dashboard

**Purpose**: Exports metrics for Grafana dashboards and monitoring.

**Usage**:
```bash
# Export all formats
python -m bondx.quality.metrics_exporter \
  --input quality/last_run_report.json \
  --out quality/metrics \
  --grafana-dashboard

# Export specific format
python -m bondx.quality.metrics_exporter \
  --input quality/last_run_report.json \
  --out quality/metrics \
  --formats csv
```

**Key Features**:
- Prometheus-formatted metrics
- CSV export for Grafana import
- Pre-configured dashboard JSON
- Real-time metric updates

## 📊 Configuration

### Quality Policy Configuration

```yaml
# bondx/quality/config/quality_policy.yaml
thresholds:
  coverage_min: 90.0
  esg_missing_max: 20.0
  liquidity_index_median_min: 30.0

environments:
  production:
    coverage_min: 95.0
    esg_missing_max: 10.0
```

### Policy Drift Configuration

```yaml
# bondx/quality/config/policy_drift.yaml
thresholds:
  fail_rate_drop_threshold: 0.15      # 15% drop triggers alert
  warn_rate_drop_threshold: 0.20      # 20% drop triggers alert
  moving_average_window: 30           # 30-day analysis window
```

## 🧪 Testing

### Run All Tests

```bash
make quality-test
```

### Run Specific Test Suites

```bash
# Golden harness tests
python -m pytest tests/golden/ -v

# Quality system tests
python -m pytest tests/quality/ -v
```

### Test Coverage

```bash
python -m pytest --cov=bondx.quality --cov-report=html
```

## 🔄 CI/CD Integration

### GitHub Actions Example

```yaml
name: Quality Assurance
on: [push, pull_request]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: make quality-deps
      
      - name: Generate golden datasets
        run: make golden-datasets
      
      - name: Validate golden datasets
        run: make golden-validate
      
      - name: Check policy drift
        run: make policy-drift
      
      - name: Generate quality dashboard
        run: make quality-dashboard
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: quality-reports
          path: |
            quality/
            reports/regulator/
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: golden-validation
        name: Golden Dataset Validation
        entry: python scripts/golden/validate_golden.py
        language: system
        pass_filenames: false
        always_run: true
```

## 📈 Monitoring and Alerting

### Grafana Dashboard

1. Import the generated dashboard JSON
2. Configure data sources (CSV or Prometheus)
3. Set up refresh intervals (recommended: 5 minutes)

### Key Metrics

- `bondx_quality_pass_rate` - Overall pass rate
- `bondx_quality_fail_rate` - Overall fail rate
- `bondx_quality_gate_pass_total` - Gates passing
- `bondx_quality_gate_fail_total` - Gates failing

### Alerting Rules

```yaml
# Example Prometheus alerting rules
groups:
  - name: quality_alerts
    rules:
      - alert: HighFailureRate
        expr: bondx_quality_fail_rate > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High quality failure rate detected"
```

## 🚨 Troubleshooting

### Common Issues

#### Golden Validation Fails

1. **Check baseline drift**:
   ```bash
   python scripts/golden/validate_golden.py --dataset v1_dirty --verbose
   ```

2. **Review recent changes**:
   ```bash
   cat data/golden/CHANGELOG.json
   ```

3. **Update baseline if intentional**:
   ```bash
   python scripts/golden/update_baseline.py \
     --dataset v1_dirty \
     --approve \
     --reason "Fix for validation logic change" \
     --reviewer "Your Name"
   ```

#### Policy Drift Detected

1. **Check policy version**:
   ```bash
   cat bondx/quality/policy_version.txt
   ```

2. **Review drift report**:
   ```bash
   cat quality/policy_drift.json
   ```

3. **Update policy version if intentional**:
   ```bash
   echo "1.1.0" > bondx/quality/policy_version.txt
   ```

#### Metrics Export Fails

1. **Check input file**:
   ```bash
   ls -la quality/last_run_report.json
   ```

2. **Verify file format**:
   ```bash
   python -c "import json; json.load(open('quality/last_run_report.json'))"
   ```

3. **Check output directory permissions**:
   ```bash
   mkdir -p quality/metrics
   ```

### Debug Mode

Enable verbose logging for all components:

```bash
# Golden validation
python scripts/golden/validate_golden.py --verbose

# Policy drift detection
python -m bondx.quality.policy_monitor --verbose

# Evidence pack generation
python -m bondx.reporting.regulator.evidence_pack --verbose

# Shadow runner
python -m bondx.quality.shadow_runner --verbose

# Metrics export
python -m bondx.quality.metrics_exporter --verbose
```

## 📚 Documentation

### Component Documentation

- [Golden Dataset Vault](data/golden/README.md) - Dataset structure and usage
- [Golden Policy](docs/GOLDEN_POLICY.md) - Baseline update policy and procedures
- [Policy Drift](docs/POLICY_DRIFT.md) - Drift detection thresholds and interpretation
- [Shadow Data](docs/SHADOW_DATA.md) - Real data anonymization guidelines

### API Documentation

- [Quality API](bondx/quality/README.md) - Core quality system API
- [Reporting API](bondx/reporting/README.md) - Evidence pack generation API
- [Metrics API](bondx/quality/metrics_exporter.py) - Metrics export API

### Examples

- [Golden Dataset Examples](data/golden/examples/) - Sample datasets and baselines
- [Dashboard Examples](quality/metrics/examples/) - Sample Grafana dashboards
- [Report Examples](reports/regulator/examples/) - Sample evidence packs

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch**:
   ```bash
   git checkout -b feature/quality-enhancement
   ```
3. **Make changes and test**:
   ```bash
   make quality-test
   make golden-validate
   ```
4. **Update documentation**
5. **Submit pull request**

### Code Standards

- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Write unit tests for new functionality
- Update relevant documentation

### Testing Requirements

- All new features must have tests
- Golden validation must pass
- Policy drift detection must work
- Metrics export must function

## 📞 Support

### Getting Help

- **Documentation**: Check the component-specific READMEs
- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Wiki**: Check the project wiki for additional resources

### Contact Information

- **Quality Team**: [email protected]
- **DevOps Team**: [email protected]
- **Compliance Team**: [email protected]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Quality Engineering Community** - Best practices and patterns
- **Open Source Contributors** - Libraries and tools used
- **BondX Team** - Domain expertise and requirements
- **Regulatory Community** - Compliance requirements and guidance

---

**Note**: This system is designed for enterprise use and includes comprehensive audit trails, compliance features, and security considerations. Always review and customize configurations for your specific environment and requirements.
