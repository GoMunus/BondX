# BondX MLOps Module

The BondX MLOps module provides end-to-end machine learning operations capabilities including experiment tracking, model registry, drift detection, automated retraining, and canary deployments.

## Overview

This module implements a comprehensive MLOps layer that manages the complete lifecycle of ML models in the BondX system:

- **Experiment Tracking**: Log parameters, metrics, artifacts, and environment information
- **Model Registry**: Version control and stage management (Development → Staging → Production)
- **Drift Detection**: Monitor feature distributions and target residuals for data drift
- **Automated Retraining**: Trigger retraining pipelines when drift is detected
- **Canary Deployments**: Gradual rollout of new models with performance monitoring
- **CLI Interface**: Command-line tools for all MLOps operations

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Experiment    │    │   Model         │    │   Drift         │
│   Tracking      │    │   Registry      │    │   Monitor       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Retraining    │    │   Canary        │    │   CLI           │
│   Pipeline      │    │   Deployment    │    │   Interface     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Model Lifecycle

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Development │───▶│  Staging   │───▶│ Production │───▶│  Archived   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Train     │    │   Validate  │    │   Monitor   │    │   Cleanup   │
│   Model     │    │   & Test    │    │   & Alert   │    │   Old       │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Canary Deployment Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Deploy    │───▶│   Route     │───▶│   Monitor   │───▶│   Evaluate  │
│   Candidate │    │   Traffic   │    │   Metrics   │    │   Results   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                              │
                                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Rollback  │◀───│   Decision  │◀───│   Compare   │◀───│   Threshold │
│   if Failed │    │   Point     │    │   vs Prod   │    │   Check     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Installation

The MLOps module is included with BondX. Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- `scikit-learn` - Machine learning algorithms
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scipy` - Statistical functions
- `pyyaml` - YAML configuration
- `psutil` - System monitoring

## Configuration

The MLOps module uses a YAML configuration file (`policy.yaml`) that defines:

- Drift detection thresholds
- Canary deployment policies
- Model promotion gates
- Experiment tracking settings
- Model-specific configurations

### Example Configuration

```yaml
# Drift detection
drift:
  feature_drift_threshold: 0.1
  target_drift_threshold: 0.15
  residual_drift_threshold: 0.2

# Canary deployment
canary:
  initial_canary_percentage: 0.05  # 5%
  max_canary_percentage: 0.20      # 20%
  promotion_threshold: 0.95        # 95%

# Model promotion
promotion:
  min_accuracy: 0.85
  max_drift: 0.1
  require_explainability: true
```

## Usage

### Command Line Interface

The MLOps module provides a comprehensive CLI for all operations:

```bash
# Train a new model
bondx-mlops train --model spread --data data.csv

# Evaluate a model
bondx-mlops evaluate --model spread --version 1.0.0 --test-data test.csv

# Register a trained model
bondx-mlops register --model spread --version 1.0.0 --path ./models/spread_v1

# Promote a model to production
bondx-mlops promote --model spread --version 1.0.0 --stage production --user admin

# Check for data drift
bondx-mlops drift-check --model spread --baseline baseline.csv --current current.csv

# Create canary deployment
bondx-mlops deploy --model spread --version 1.0.1 --user admin --traffic 0.1

# List models and deployments
bondx-mlops list --models --stage production
bondx-mlops list --deployments
```

### Python API

You can also use the MLOps modules programmatically:

```python
from bondx.mlops import MLOpsConfig, ExperimentTracker, ModelRegistry, DriftMonitor

# Initialize components
config = MLOpsConfig.from_yaml('policy.yaml')
tracker = ExperimentTracker(config)
registry = ModelRegistry(config)
drift_monitor = DriftMonitor(config)

# Start experiment
run_id = tracker.start_run(
    experiment_name="spread_model_training",
    model_type="spread"
)

# Log parameters
tracker.log_parameters(run_id, {
    'learning_rate': 0.01,
    'max_depth': 10,
    'random_state': 42
})

# Log metrics
tracker.log_metrics(run_id, {
    'mae': 0.15,
    'rmse': 0.25,
    'r2_score': 0.85
})

# End experiment
tracker.end_run(run_id, "completed")

# Register model
model_id = registry.register_model(
    model_name="spread",
    version="1.0.0",
    run_id=run_id,
    experiment_name="spread_model_training",
    model_type="spread",
    feature_columns=["coupon_rate", "maturity_years"],
    target_column="spread_bps",
    performance_metrics={'mae': 0.15, 'r2_score': 0.85},
    hyperparameters={'learning_rate': 0.01},
    model_path="./model.pkl",
    metadata_path="./metadata.json"
)

# Promote model to staging
registry.promote_model(
    model_name="spread",
    version="1.0.0",
    target_stage=ModelStage.STAGING,
    deployed_by="ml_engineer"
)
```

## Drift Detection

The drift detection system monitors:

- **Feature Drift**: Changes in feature distributions using statistical tests (KS test, PSI, Chi-square)
- **Target Drift**: Changes in target variable distributions
- **Residual Drift**: Changes in model prediction errors

### Drift Detection Example

```python
# Detect drift
drift_report = drift_monitor.detect_drift(
    model_name="spread",
    model_version="1.0.0",
    baseline_data=baseline_df,
    current_data=current_df,
    feature_columns=["coupon_rate", "maturity_years"],
    target_column="spread_bps"
)

print(f"Drift detected: {drift_report.requires_retraining}")
print(f"Drifted features: {drift_report.drifted_features}")
print(f"Overall drift score: {drift_report.overall_drift_score}")
```

## Automated Retraining

When drift is detected, the system can automatically trigger retraining:

```python
from bondx.mlops import RetrainPipeline

retrain_pipeline = RetrainPipeline(config, tracker, registry)

# Trigger retraining based on drift
result = retrain_pipeline.trigger_retraining(drift_report, training_data)

if result.success:
    print(f"New model version: {result.new_model_version}")
    print(f"Training time: {result.training_time_seconds:.2f}s")
```

## Canary Deployments

Canary deployments allow gradual rollout of new models:

```python
from bondx.mlops import CanaryDeploymentManager

deployment_manager = CanaryDeploymentManager(config, registry)

# Create canary deployment
deployment_id = deployment_manager.create_canary_deployment(
    model_name="spread",
    candidate_version="1.0.1",
    deployed_by="ml_engineer",
    initial_traffic_percentage=0.05
)

# Route predictions
route_to_canary, request_id = deployment_manager.route_prediction(
    model_name="spread",
    features={"coupon_rate": 5.0, "maturity_years": 10}
)

# Promote if performance is good
deployment_manager.promote_canary(deployment_id, "ml_engineer")
```

## Monitoring and Alerting

The system provides comprehensive monitoring:

- **Performance Metrics**: Real-time model performance tracking
- **Drift Alerts**: Automatic notifications when drift is detected
- **Health Checks**: Regular model health assessments
- **Audit Logs**: Complete audit trail of all operations

## Security and Compliance

The MLOps module includes:

- **Access Control**: Authentication and authorization for all operations
- **Audit Logging**: Complete audit trail for compliance
- **Data Privacy**: Anonymization and encryption of sensitive data
- **Regulatory Compliance**: Support for Basel III, IFRS 9, MiFID II

## Testing

Run the test suite:

```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Coverage report
python -m pytest --cov=bondx.mlops tests/ -v
```

## Deployment

### Local Development

```bash
# Set environment
export BONDX_ENV=development

# Run MLOps services
python -m bondx.mlops.cli list --models
```

### Production Deployment

```bash
# Set environment
export BONDX_ENV=production

# Load production config
python -m bondx.mlops.cli --config production_policy.yaml list --models
```

## Troubleshooting

### Common Issues

1. **Configuration Errors**: Ensure `policy.yaml` is valid YAML
2. **Permission Errors**: Check file permissions for model storage directories
3. **Drift Detection Failures**: Verify data formats and column names
4. **Model Registration Issues**: Check model file paths and metadata

### Logs

Logs are stored in the configured log directory:
- `mlops_cli.log` - CLI operation logs
- `drift_detection.log` - Drift detection logs
- `model_registry.log` - Model registry logs

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('bondx.mlops').setLevel(logging.DEBUG)
```

## Contributing

1. Follow the existing code style
2. Add tests for new functionality
3. Update documentation
4. Ensure all tests pass
5. Submit a pull request

## License

This module is part of the BondX system and follows the same licensing terms.

## Support

For support and questions:
- Check the documentation
- Review the test examples
- Open an issue on the repository
- Contact the development team
