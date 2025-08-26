# BondX Golden Dataset Vault

The Golden Dataset Vault contains frozen, curated datasets with known violations and expected outcomes. This vault serves as a "source of truth" for quality validation and prevents silent drift in the quality pipeline.

## Purpose

- **Prevent Silent Drift**: Catch unintended changes in quality behavior before they reach production
- **Regression Testing**: Ensure quality improvements don't break existing validations
- **Audit Trail**: Maintain historical baselines for compliance and debugging
- **CI/CD Integration**: Block PRs that would change quality outcomes without approval

## Dataset Structure

```
data/golden/
├── v1_clean/           # Perfect dataset - all validators pass
├── v1_dirty/           # Known violations - negative spreads, invalid ratings, etc.
├── v1_mixed/           # Mixed dataset - pass + warn + fail records
└── baselines/          # Expected outputs for each dataset
    ├── v1_clean/
    │   ├── last_run_report.json
    │   ├── metrics.json
    │   └── summary.txt
    ├── v1_dirty/
    └── v1_mixed/
```

## Dataset Descriptions

### v1_clean (Perfect Dataset)
- **Size**: 100 records
- **Characteristics**: All validators pass, no violations
- **Purpose**: Baseline for "green" quality runs
- **Expected Outcome**: 100% PASS, 0% WARN, 0% FAIL

### v1_dirty (Known Violations)
- **Size**: 100 records
- **Known Violations**:
  - 5% negative spreads (FAIL)
  - 10% invalid ratings (FAIL)
  - 15% bad maturity dates (FAIL)
  - 20% stale quotes (WARN)
  - 25% missing ESG data (WARN)
- **Purpose**: Test failure detection and reporting
- **Expected Outcome**: ~45% PASS, ~35% WARN, ~20% FAIL

### v1_mixed (Mixed Results)
- **Size**: 100 records
- **Characteristics**: Balanced mix of pass/warn/fail scenarios
- **Purpose**: Test warning thresholds and edge cases
- **Expected Outcome**: ~60% PASS, ~25% WARN, ~15% FAIL

## Baseline Files

Each dataset has three baseline files:

1. **last_run_report.json**: Complete validation results with timestamps
2. **metrics.json**: Aggregated quality metrics and KPIs
3. **summary.txt**: Human-readable summary of outcomes

## Usage

### Validation Harness
```bash
# Run validation on all golden datasets
python scripts/golden/validate_golden.py

# Validate specific dataset
python scripts/golden/validate_golden.py --dataset v1_dirty
```

### Baseline Updates
```bash
# Update baselines (requires approval)
python scripts/golden/update_baseline.py --dataset v1_dirty --approve

# Review changes before approval
python scripts/golden/update_baseline.py --dataset v1_dirty --dry-run
```

## CI/CD Integration

The golden validation harness runs on every PR:

1. **Pre-merge Check**: Validates all golden datasets
2. **Baseline Comparison**: Compares outputs to frozen baselines
3. **Drift Detection**: Blocks merge if outputs don't match baselines
4. **Artifact Upload**: Uploads validation reports for review

## Update Policy

Baselines can only be updated via explicit maintainer approval:

1. **Review Required**: Changes must be reviewed by quality team
2. **Policy Version**: Must include policy version bump
3. **Changelog**: Must document rationale and impact
4. **Testing**: Must pass all other golden datasets

## Violations List

### Critical Failures (v1_dirty)
- Negative spreads: 5 records (5%)
- Invalid ratings: 10 records (10%)
- Bad maturity dates: 15 records (15%)

### Warnings (v1_dirty)
- Stale quotes: 20 records (20%)
- Missing ESG data: 25 records (25%)

### Edge Cases (v1_mixed)
- Borderline maturity dates
- Threshold violations
- Mixed severity scenarios

## Maintenance

- **Monthly Review**: Audit baseline accuracy
- **Policy Updates**: Update baselines when policies change
- **Version Control**: Tag baseline versions with policy versions
- **Documentation**: Maintain changelog of all updates

## Troubleshooting

### Baseline Mismatch
1. Check if quality policy changed
2. Verify dataset integrity
3. Review validation logic changes
4. Update baseline if intentional

### Validation Failures
1. Check dataset format
2. Verify validator configuration
3. Review quality policy settings
4. Check for environment differences

## Security

- **No Secrets**: Datasets contain only synthetic data
- **Access Control**: Baseline updates require approval
- **Audit Trail**: All changes are logged and tracked
- **Version Locking**: Baselines are immutable without approval
