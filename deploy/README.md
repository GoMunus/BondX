# BondX Production Deployment Guide

This directory contains all the configuration files and scripts needed to deploy BondX to production with high availability, security, and monitoring.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Load Balancer │    │   Load Balancer │
│   (Nginx/HAProxy)│    │   (Nginx/HAProxy)│    │   (Nginx/HAProxy)│
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │      Kubernetes Cluster   │
                    │                           │
                    │  ┌─────────────────────┐  │
                    │  │   BondX Backend     │  │
                    │  │   (3+ replicas)     │  │
                    │  └─────────────────────┘  │
                    │                           │
                    │  ┌─────────────────────┐  │
                    │  │   Redis HA Cluster  │  │
                    │  │  Master + 2 Replicas│  │
                    │  │   + 3 Sentinels     │  │
                    │  └─────────────────────┘  │
                    │                           │
                    │  ┌─────────────────────┐  │
                    │  │   PostgreSQL DB     │  │
                    │  │   + Read Replicas   │  │
                    │  └─────────────────────┘  │
                    │                           │
                    │  ┌─────────────────────┐  │
                    │  │   Monitoring Stack  │  │
                    │  │ Prometheus + Grafana│  │
                    │  └─────────────────────┘  │
                    └───────────────────────────┘
```

## Directory Structure

```
deploy/
├── kubernetes/           # Kubernetes manifests
│   ├── namespace.yaml    # Namespace definitions
│   ├── configmap.yaml    # Application configuration
│   ├── secret.yaml       # Sensitive configuration
│   ├── deployment.yaml   # Application deployment
│   ├── service.yaml      # Service definitions
│   ├── hpa.yaml         # Horizontal Pod Autoscaler
│   ├── pdb.yaml         # Pod Disruption Budget
│   └── ingress.yaml     # Ingress configuration
├── nginx/               # Nginx configuration
│   └── nginx.conf       # Production Nginx config
├── haproxy/             # HAProxy configuration
│   └── haproxy.cfg      # Production HAProxy config
├── redis/               # Redis HA configuration
│   ├── redis-master.conf    # Redis master config
│   ├── redis-replica.conf   # Redis replica config
│   ├── redis-sentinel.conf  # Redis Sentinel config
│   └── docker-compose-ha.yml # Redis HA Docker setup
├── monitoring/          # Monitoring configuration
│   ├── prometheus-config.yaml    # Prometheus config
│   ├── prometheus-rules.yaml     # Alerting rules
│   └── grafana-dashboard.yaml    # Grafana dashboards
└── security/            # Security configuration
    └── security-checklist.md     # Security hardening checklist
```

## Prerequisites

### System Requirements
- Kubernetes cluster (v1.24+)
- kubectl configured and accessible
- Helm (optional, for additional charts)
- Docker (for local testing)

### Infrastructure Requirements
- **Nodes**: Minimum 3 nodes for production
- **CPU**: 4+ cores per node
- **Memory**: 8GB+ RAM per node
- **Storage**: 100GB+ per node
- **Network**: High-speed, low-latency network

### Software Requirements
- **Kubernetes**: v1.24+
- **Container Runtime**: containerd or Docker
- **CNI**: Calico, Flannel, or similar
- **Storage**: CSI-compatible storage driver

## Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd BondX
```

### 2. Configure Environment
```bash
# Copy and edit environment configuration
cp deploy/kubernetes/secret.yaml.example deploy/kubernetes/secret.yaml
# Edit the secret.yaml file with your actual values
```

### 3. Deploy to Production
```bash
# Using Makefile (recommended)
make deploy

# Or using the deployment script
./ops/scripts/deploy-production.sh

# Or manually with kubectl
kubectl apply -f deploy/kubernetes/
```

### 4. Verify Deployment
```bash
# Check deployment status
make status

# Check application logs
make logs

# Run health checks
make health-check
```

## Configuration

### Environment Variables
The application is configured through Kubernetes ConfigMaps and Secrets:

- **ConfigMap**: Contains non-sensitive configuration
- **Secret**: Contains sensitive data (API keys, passwords, etc.)

### Key Configuration Areas

#### Application Configuration
- Server settings (host, port, workers)
- Database connection settings
- Redis connection settings
- Logging configuration
- Monitoring settings

#### Security Configuration
- JWT secret keys
- API keys for external services
- SSL/TLS certificates
- Authentication settings

#### Infrastructure Configuration
- Resource limits and requests
- Scaling policies
- Health check settings
- Network policies

## Deployment Options

### Load Balancer Options

#### Nginx (Recommended)
- **Pros**: Mature, feature-rich, excellent WebSocket support
- **Cons**: More complex configuration
- **Use Case**: Production environments with complex routing needs

#### HAProxy
- **Pros**: High performance, simple configuration
- **Cons**: Less feature-rich than Nginx
- **Use Case**: High-performance requirements, simple routing

### Redis HA Options

#### Redis Sentinel (Recommended)
- **Pros**: Automatic failover, mature technology
- **Cons**: More complex setup
- **Use Case**: Production environments requiring high availability

#### Redis Cluster
- **Pros**: Horizontal scaling, sharding
- **Cons**: More complex, some limitations
- **Use Case**: Very large datasets, horizontal scaling needs

## Monitoring and Observability

### Metrics Collection
- **Prometheus**: Collects metrics from all components
- **Custom Metrics**: Business KPIs, application-specific metrics
- **Infrastructure Metrics**: Node, pod, and cluster metrics

### Alerting
- **Business Alerts**: Order volume, fill rates, risk metrics
- **Infrastructure Alerts**: Performance, availability, errors
- **Market Data Alerts**: Data freshness, ingestion failures

### Dashboards
- **Business KPIs**: Trading metrics, risk indicators
- **Infrastructure**: System performance, resource usage
- **Market Data**: External API status, data quality

## Security Features

### Network Security
- TLS 1.2+ only
- Modern cipher suites
- HSTS headers
- Rate limiting
- DDoS protection

### Application Security
- JWT authentication
- Role-based access control
- Input validation
- SQL injection prevention
- XSS protection

### Infrastructure Security
- Pod security policies
- Network policies
- Resource limits
- Security contexts
- Vulnerability scanning

## High Availability Features

### Application HA
- Multiple replicas (3+)
- Pod disruption budgets
- Anti-affinity rules
- Health checks and probes

### Data HA
- Redis master-replica with Sentinel
- Database read replicas
- Connection pooling
- Automatic failover

### Infrastructure HA
- Multiple availability zones
- Load balancer redundancy
- Monitoring redundancy
- Backup and recovery

## Scaling

### Horizontal Scaling
- **HPA**: Automatic scaling based on metrics
- **Manual Scaling**: Direct replica management
- **Resource-based**: CPU and memory utilization

### Vertical Scaling
- **Resource Limits**: CPU and memory limits
- **Resource Requests**: Minimum resource allocation
- **Node Scaling**: Adding more nodes to cluster

## Backup and Recovery

### Backup Strategy
- **Application**: Kubernetes manifests
- **Data**: Database dumps, Redis snapshots
- **Configuration**: ConfigMaps, Secrets
- **Monitoring**: Prometheus data

### Recovery Procedures
- **Application**: kubectl apply
- **Data**: Database restore, Redis restore
- **Configuration**: ConfigMap/Secret restore
- **Full Recovery**: Complete cluster restore

## Troubleshooting

### Common Issues

#### Deployment Issues
```bash
# Check pod status
kubectl get pods -n bondx

# Check pod logs
kubectl logs -n bondx <pod-name>

# Check events
kubectl get events -n bondx --sort-by='.lastTimestamp'
```

#### Network Issues
```bash
# Check service endpoints
kubectl get endpoints -n bondx

# Check ingress status
kubectl get ingress -n bondx

# Test connectivity
kubectl exec -n bondx <pod-name> -- curl <service-url>
```

#### Redis Issues
```bash
# Check Redis status
make redis-status

# Check Sentinel status
kubectl exec -n bondx <sentinel-pod> -- redis-cli -p 26379 sentinel master bondx-master
```

### Debug Commands
```bash
# Get detailed pod information
kubectl describe pod -n bondx <pod-name>

# Check resource usage
kubectl top pods -n bondx

# Check configuration
kubectl get configmap -n bondx bondx-config -o yaml
```

## Maintenance

### Regular Tasks
- **Security Updates**: Monthly security patches
- **Performance Tuning**: Quarterly performance reviews
- **Capacity Planning**: Monthly resource usage review
- **Backup Testing**: Weekly backup restoration tests

### Update Procedures
- **Application Updates**: Rolling updates with health checks
- **Infrastructure Updates**: Blue-green deployment
- **Configuration Updates**: ConfigMap/Secret updates
- **Security Updates**: Security patch management

## Compliance

### SEBI Compliance
- Market abuse prevention
- Insider trading detection
- Regulatory reporting
- Audit logging

### RBI Compliance
- KYC/AML procedures
- Transaction monitoring
- Suspicious activity reporting
- Data retention policies

### Data Protection
- Data encryption at rest
- Data encryption in transit
- Access controls
- Audit trails

## Support and Documentation

### Runbooks
- [Redis Failover](../ops/runbooks/redis-failover.md)
- [Incident Response](../ops/runbooks/incident-response.md)
- [Disaster Recovery](../ops/runbooks/disaster-recovery.md)

### Scripts
- [Production Deployment](../ops/scripts/deploy-production.sh)
- [Health Checks](../ops/scripts/health-checks.sh)
- [Backup and Restore](../ops/scripts/backup-restore.sh)

### Monitoring
- [Prometheus Configuration](./monitoring/prometheus-config.yaml)
- [Alerting Rules](./monitoring/prometheus-rules.yaml)
- [Grafana Dashboards](./monitoring/grafana-dashboard.yaml)

## Contributing

### Adding New Components
1. Create Kubernetes manifests
2. Add to deployment scripts
3. Update monitoring configuration
4. Document in this README

### Configuration Changes
1. Update ConfigMap/Secret
2. Test in staging environment
3. Deploy to production
4. Update documentation

### Security Updates
1. Review security checklist
2. Update security policies
3. Test security measures
4. Deploy security updates

## License

This deployment configuration is part of the BondX project and is licensed under the same terms as the main project.
