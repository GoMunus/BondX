# BondX Production Runbooks Index
## Quick-Action Cards for Critical Scenarios

**Document Version:** 1.0  
**Last Updated:** $(Get-Date -Format "yyyy-MM-dd")  
**Owner:** Operations Team  
**Audience:** DevOps, SRE, Operations Engineers  

---

## 📚 Runbooks Index

### Core Operational Runbooks
- **[GO_LIVE_RUNBOOK.md](./GO_LIVE_RUNBOOK.md)** - Complete go-live process and validation gates
- **[DATA_ADAPTER_CUTOVER_PLAN.md](./DATA_ADAPTER_CUTOVER_PLAN.md)** - Live data provider integration
- **[BUSINESS_VALIDATION_UAT.md](./BUSINESS_VALIDATION_UAT.md)** - User acceptance testing framework
- **[RELEASE_1_0_LAUNCH_CHECKLIST.md](./RELEASE_1_0_LAUNCH_CHECKLIST.md)** - Single source of truth for launch

### Playbooks and Drills
- **[CHAOS_INCIDENT_DRILLS.md](../playbooks/CHAOS_INCIDENT_DRILLS.md)** - Chaos engineering and incident response
- **[Security Hardening](../security/SECURITY_HARDENING_SPRINT.md)** - Security controls and compliance

### Monitoring and SLOs
- **[SLOs and Alerts](../../deploy/monitoring/slos.yaml)** - Service level objectives and alerting rules
- **[Trading Dashboard](../../deploy/monitoring/grafana-dashboard-trading.json)** - Trading performance metrics
- **[Prometheus Rules](../../deploy/monitoring/prometheus-rules.yaml)** - Alerting and recording rules

---

## 🚨 Quick-Action Cards

### Card 1: Degraded WebSockets

#### 🚨 Symptoms
- Connection count >80% of capacity
- Message delivery latency >150ms p95
- Reconnection success rate <99%
- Client complaints about "laggy" updates

#### 📊 Key Metrics to Check
```bash
# Check WebSocket health
make health-check --service=websocket
make validate-production-metrics --component=websocket

# Monitor connection count
kubectl top pods -n bondx -l app=websocket-manager
```

#### ⚡ Immediate Actions (0-5 minutes)
1. **Scale Out WebSocket Managers**
   ```bash
   kubectl scale deployment websocket-manager -n bondx --replicas=8
   make scale-up --service=websocket-manager
   ```

2. **Enable Compression**
   ```bash
   kubectl patch configmap websocket-config -n bondx \
     --patch '{"data":{"compression_enabled":"true"}}'
   ```

3. **Verify Stickiness**
   ```bash
   # Check if load balancer is maintaining session affinity
   kubectl get svc websocket-service -n bondx -o yaml | grep sessionAffinity
   ```

#### 🔍 Investigation (5-15 minutes)
- Check Redis connection pool health
- Verify network latency between clients and WebSocket nodes
- Review recent deployments for configuration changes
- Check for memory leaks or high CPU usage

#### ✅ Exit Criteria
- Connection count <70% of capacity
- Message delivery latency <100ms p95
- Reconnection success rate >99.5%
- No client complaints for 15 minutes

---

### Card 2: Redis Failover

#### 🚨 Symptoms
- Redis connection errors in logs
- High latency on Redis operations
- Sentinel failover events
- Service degradation across multiple components

#### 📊 Key Metrics to Check
```bash
# Check Redis status
make redis-status
make health-check --service=redis

# Monitor Redis metrics
kubectl exec -n bondx redis-master-0 -- redis-cli info replication
```

#### ⚡ Immediate Actions (0-5 minutes)
1. **Force Failover if Needed**
   ```bash
   kubectl exec -n bondx redis-sentinel-0 -- redis-cli -p 26379 sentinel failover mymaster
   ```

2. **Validate New Master**
   ```bash
   kubectl exec -n bondx redis-master-0 -- redis-cli info replication
   ```

3. **Restart Consumers**
   ```bash
   kubectl rollout restart deployment order-manager -n bondx
   kubectl rollout restart deployment risk-engine -n bondx
   ```

#### 🔍 Investigation (5-15 minutes)
- Check Redis memory usage and eviction policies
- Verify network connectivity between Redis nodes
- Review Redis logs for errors or warnings
- Check if failover was triggered by resource constraints

#### ✅ Exit Criteria
- Redis master-slave replication working
- All services successfully reconnected
- No Redis connection errors in logs
- Normal latency restored (<10ms for simple operations)

---

### Card 3: Replica Lag

#### 🚨 Symptoms
- Database read latency >2s
- Inconsistent data between primary and replicas
- Risk calculations using stale data
- Compliance reports with outdated information

#### 📊 Key Metrics to Check
```bash
# Check replica lag
kubectl exec -n bondx postgres-primary-0 -- psql -c "SELECT * FROM pg_stat_replication;"

# Monitor database performance
make health-check --service=database
```

#### ⚡ Immediate Actions (0-5 minutes)
1. **Throttle Read Traffic**
   ```bash
   # Update service configuration to use primary for critical reads
   kubectl patch configmap database-config -n bondx \
     --patch '{"data":{"read_from_replica":"false"}}'
   ```

2. **Re-route Critical Services**
   ```bash
   # Force risk engine to use primary
   kubectl patch deployment risk-engine -n bondx \
     --patch '{"spec":{"template":{"spec":{"containers":[{"name":"risk-engine","env":[{"name":"DB_READ_REPLICA","value":"false"}]}]}}}}'
   ```

3. **Check Replica Health**
   ```bash
   kubectl exec -n bondx postgres-replica-0 -- pg_isready
   ```

#### 🔍 Investigation (5-15 minutes)
- Check replica server resources (CPU, memory, disk I/O)
- Verify network latency between primary and replica
- Review recent database schema changes
- Check for long-running queries blocking replication

#### ✅ Exit Criteria
- Replica lag <1s
- All services using appropriate database endpoints
- Normal read latency restored (<100ms)
- Risk and compliance data freshness verified

---

### Card 4: Data Provider Outage

#### 🚨 Symptoms
- Market data staleness >5 minutes
- Yield curve data not updating
- Credit rating changes not reflected
- Fallback mechanisms activated

#### 📊 Key Metrics to Check
```bash
# Check data freshness
make validate-canary-data
make check-provider-health

# Monitor data provider status
kubectl logs -n bondx -l app=data-adapter --tail=100
```

#### ⚡ Immediate Actions (0-5 minutes)
1. **Activate Fallback Ladder**
   ```bash
   # Switch to backup provider
   kubectl patch configmap data-provider-config -n bondx \
     --patch '{"data":{"primary_provider":"backup","fallback_mode":"true"}}'
   ```

2. **Enable Mock Data Mode**
   ```bash
   make enable-mock-data-providers
   ```

3. **Notify Stakeholders**
   ```bash
   # Send alert to trading desk and risk team
   kubectl exec -n bondx notification-service-0 -- /app/notify.py --severity=high --message="Data provider outage detected, fallback activated"
   ```

#### 🔍 Investigation (5-15 minutes)
- Check provider API status and response times
- Verify authentication and rate limiting
- Review recent configuration changes
- Check network connectivity to external providers

#### ✅ Exit Criteria
- Data freshness <1 minute
- All data sources providing updates
- Fallback mechanisms deactivated
- Stakeholders notified of resolution

---

### Card 5: Risk Alerts Storm

#### 🚨 Symptoms
- High volume of risk alerts
- Risk engine performance degradation
- Multiple limit breaches reported
- Risk dashboard showing unusual patterns

#### 📊 Key Metrics to Check
```bash
# Check risk engine status
make health-check --service=risk-engine
make validate-production-metrics --component=risk

# Monitor alert volume
kubectl logs -n bondx -l app=risk-engine --tail=200 | grep "ALERT"
```

#### ⚡ Immediate Actions (0-5 minutes)
1. **Temporarily Adjust Thresholds**
   ```bash
   # Increase alert thresholds to reduce noise
   kubectl patch configmap risk-config -n bondx \
     --patch '{"data":{"alert_threshold_multiplier":"2.0"}}'
   ```

2. **Enable Rate Limiting**
   ```bash
   # Limit alert generation rate
   kubectl patch configmap risk-config -n bondx \
     --patch '{"data":{"max_alerts_per_minute":"100"}}'
   ```

3. **Scale Risk Engine**
   ```bash
   kubectl scale deployment risk-engine -n bondx --replicas=4
   ```

#### 🔍 Investigation (5-15 minutes)
- Check for unusual market conditions
- Verify risk model calculations
- Review recent portfolio changes
- Check for data quality issues

#### ✅ Exit Criteria
- Alert volume normalized
- Risk engine performance restored
- Thresholds reset to normal levels
- Root cause identified and documented

---

## 🛠️ Common Commands Reference

### Health Checks
```bash
# Overall system health
make health-check

# Service-specific health
make health-check --service=websocket-manager
make health-check --service=risk-engine
make health-check --service=trading-engine

# Database health
make health-check --service=database
```

### Scaling Operations
```bash
# Scale up services
make scale-up --service=websocket-manager --replicas=8
make scale-up --service=risk-engine --replicas=4

# Scale down services
make scale-down --service=websocket-manager --replicas=4
make scale-down --service=risk-engine --replicas=2
```

### Monitoring and Metrics
```bash
# View service status
make status

# Check logs
make logs --service=websocket-manager --tail=100
make logs --service=risk-engine --tail=100

# Monitor KPIs
make monitor-kpis
```

### Emergency Operations
```bash
# Emergency rollback
make emergency-rollback

# Traffic management
make drain-traffic
make restore-traffic

# Feature toggles
make rollback --feature=websocket-compression
make rollback --feature=risk-engine
```

---

## 📞 Escalation Matrix

| Scenario | First Responder | Escalation (15 min) | Escalation (30 min) |
|----------|-----------------|---------------------|---------------------|
| **WebSocket Degradation** | DevOps Engineer | Platform Engineer | DevOps Lead |
| **Redis Failover** | Platform Engineer | DevOps Engineer | DevOps Lead |
| **Database Replica Lag** | DevOps Engineer | Database Engineer | DevOps Lead |
| **Data Provider Outage** | Platform Engineer | DevOps Engineer | Head of Operations |
| **Risk Alerts Storm** | Risk Engineer | DevOps Engineer | Head of Risk |

---

## 🔗 Related Documentation

- **[Operations README](../README.md)** - Complete operations overview
- **[Makefile](../../Makefile)** - Automation targets and commands
- **[PowerShell Scripts](../scripts/operational-commands.ps1)** - Windows-compatible operations
- **[Kubernetes Configs](../../deploy/kubernetes/)** - Deployment configurations
- **[Monitoring Setup](../../deploy/monitoring/)** - SLOs, dashboards, and alerts

---

**Next Review:** Monthly or after major incidents  
**Last Incident Update:** [Date]  
**Owner:** Operations Team Lead

