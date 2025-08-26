# BondX Go-Live Runbook
## Production Deployment with Validation Gates and Rollback

**Version:** 1.0  
**Last Updated:** $(date)  
**Owner:** DevOps Team  
**Approvers:** CTO, Head of Risk, Head of Compliance  

---

## Table of Contents
1. [Overview and Prerequisites](#overview-and-prerequisites)
2. [Environment Setup](#environment-setup)
3. [Promotion Steps](#promotion-steps)
4. [Validation Gates and SLOs](#validation-gates-and-slos)
5. [Rollback Plan](#rollback-plan)
6. [Communications](#communications)
7. [Appendices](#appendices)

---

## Overview and Prerequisites

### Success Criteria
- Zero data loss during promotion
- All SLOs met within 24 hours of go-live
- Rollback capability maintained throughout process
- Regulatory compliance maintained

### Risk Assessment
- **High Risk:** Data integrity, regulatory compliance
- **Medium Risk:** Performance degradation, user experience
- **Low Risk:** Feature availability, monitoring gaps

### Prerequisites Checklist
- [ ] Staging environment fully validated (48h soak)
- [ ] All feature flags configured and tested
- [ ] Secrets rotated and distributed
- [ ] Load balancer health verified
- [ ] TLS certificates valid for 90+ days
- [ ] Redis HA cluster healthy (3 masters, 3 replicas)
- [ ] Database replica lag < 100ms
- [ ] Monitoring stack operational
- [ ] Rollback images tagged and accessible
- [ ] Stakeholder approval obtained

---

## Environment Setup

### Version Matrix
| Component | Staging | Production | Rollback |
|-----------|---------|------------|----------|
| BondX Core | v1.2.0 | v1.2.0 | v1.1.5 |
| AI Engine | v2.1.0 | v2.1.0 | v2.0.8 |
| Trading Engine | v1.3.2 | v1.3.2 | v1.3.1 |
| Risk Engine | v1.4.0 | v1.4.0 | v1.3.9 |
| Database | 14.5 | 14.5 | 14.5 |
| Redis | 7.2 | 7.2 | 7.2 |

### Feature Flags Configuration
```yaml
# Production Feature Flags
FEATURE_FLAGS:
  LIVE_DATA_PROVIDERS: false          # Start with mock data
  AI_RISK_SCORING: true              # Enable AI risk
  REAL_TIME_COMPLIANCE: false        # Start with batch
  ADVANCED_ORDER_TYPES: true         # Enable all order types
  WEBSOCKET_COMPRESSION: true        # Enable compression
  RISK_LIMITS_ENFORCEMENT: true      # Strict risk limits
  AUDIT_LOGGING: true                # Full audit trail
  PERFORMANCE_MONITORING: true       # Enable monitoring
```

### Secrets Rotation
- [ ] JWT signing keys rotated
- [ ] Database passwords updated
- [ ] API keys refreshed
- [ ] External provider credentials rotated
- [ ] SSL certificates renewed

---

## Promotion Steps

### Phase 1: Staging Soak (24-48 hours)
**Duration:** 48 hours minimum  
**Success Criteria:** Zero critical errors, all SLOs met

#### 1.1 Synthetic Auction Testing
```bash
# Run synthetic auctions every 2 hours
make test-synthetic-auctions
make validate-auction-results
```

**Validation Checks:**
- [ ] Auction allocation accuracy: 100%
- [ ] Settlement completion: 100%
- [ ] Risk calculations: within 0.01% tolerance
- [ ] Compliance reporting: on-time

#### 1.2 Continuous Trading Simulation
```bash
# Simulate continuous trading for 24h
make test-continuous-trading
make validate-trading-metrics
```

**Validation Checks:**
- [ ] Order matching latency: p95 < 10ms
- [ ] Trade execution: 100% success rate
- [ ] Risk limit enforcement: 100% compliance
- [ ] WebSocket message delivery: 99.99%

#### 1.3 Risk Batch Processing
```bash
# Run overnight risk calculations
make test-risk-batch
make validate-risk-outputs
```

**Validation Checks:**
- [ ] VaR calculation: within 0.1% tolerance
- [ ] Stress test completion: 100%
- [ ] Risk report generation: on-time
- [ ] Data lineage: verified

#### 1.4 Compliance Export Validation
```bash
# Test compliance reporting
make test-compliance-export
make validate-compliance-data
```

**Validation Checks:**
- [ ] Report accuracy: 100%
- [ ] Data completeness: 100%
- [ ] Format compliance: 100%
- [ ] Delivery timing: on-schedule

### Phase 2: Dark Launch in Production
**Duration:** 4-6 hours  
**Success Criteria:** Production stack operational, staging traffic mirrored

#### 2.1 Production Stack Deployment
```bash
# Deploy production stack with observe-only mode
make deploy-production-observable
make validate-production-health
```

**Configuration:**
- Data adapters in observe-only mode
- Risk limits at 50% of production levels
- Trading limits at 25% of production levels
- Full monitoring enabled

#### 2.2 Traffic Mirroring Setup
```bash
# Configure traffic mirroring from staging
make setup-traffic-mirroring
make validate-mirroring
```

**Mirroring Rules:**
- 10% of staging traffic to production
- All critical paths covered
- Error monitoring enabled

#### 2.3 Validation and Metrics
```bash
# Validate production metrics
make validate-production-metrics
make compare-staging-production
```

**Validation Checks:**
- [ ] Response times: within 20% of staging
- [ ] Error rates: < 0.1%
- [ ] Resource utilization: < 70%
- [ ] Database performance: replica lag < 50ms

### Phase 3: Limited-Access Go-Live
**Duration:** 24-48 hours  
**Success Criteria:** Whitelisted users operational, KPIs within bounds

#### 3.1 User Whitelisting
```yaml
# Limited access configuration
LIMITED_ACCESS:
  ENABLED: true
  MAX_USERS: 50
  MAX_ORDERS_PER_USER: 100
  MAX_ORDER_VALUE: 1000000
  TRADING_HOURS: "09:00-15:30"
  RISK_LIMITS: 75%
```

#### 3.2 KPI Monitoring
```bash
# Continuous KPI monitoring
make monitor-kpis
make validate-error-budgets
```

**KPIs to Monitor:**
- Order acceptance rate: > 99.9%
- Trade execution latency: p95 < 50ms
- Risk calculation time: p95 < 1s
- WebSocket message delivery: > 99.95%
- Database query performance: p95 < 100ms

#### 3.3 Error Budget Validation
```bash
# Check error budgets
make check-error-budgets
make validate-slos
```

**Error Budget Thresholds:**
- Trading: 0.1% (8.64 minutes per day)
- Risk: 0.5% (43.2 minutes per day)
- Compliance: 0.01% (8.64 seconds per day)
- Infrastructure: 0.1% (8.64 minutes per day)

### Phase 4: General Availability
**Duration:** Gradual rollout over 1 week  
**Success Criteria:** All users operational, full limits enabled

#### 4.1 Gradual Limit Increase
```yaml
# Gradual limit increase schedule
LIMIT_INCREASE_SCHEDULE:
  DAY_1: 25% of production limits
  DAY_3: 50% of production limits
  DAY_5: 75% of production limits
  DAY_7: 100% of production limits
```

#### 4.2 Feature Enablement
```bash
# Enable production features
make enable-live-data-providers
make enable-real-time-compliance
make enable-advanced-features
```

#### 4.3 Final Validation
```bash
# Final production validation
make validate-production-readiness
make run-production-smoke-tests
```

---

## Validation Gates and SLOs

### Gate 1: Staging Soak Complete
**Criteria:**
- [ ] 48 hours of stable operation
- [ ] Zero critical errors
- [ ] All synthetic tests passed
- [ ] Performance within 10% of baseline

**Blocking Conditions:**
- Any critical error
- Performance degradation > 10%
- Risk calculation failures
- Compliance violations

### Gate 2: Production Stack Healthy
**Criteria:**
- [ ] All pods running and healthy
- [ ] Database replica lag < 100ms
- [ ] Redis cluster healthy
- [ ] Monitoring operational

**Blocking Conditions:**
- Any pod not ready
- Database lag > 100ms
- Redis cluster unhealthy
- Monitoring gaps

### Gate 3: Traffic Mirroring Validated
**Criteria:**
- [ ] 10% traffic successfully mirrored
- [ ] Response times within 20% of staging
- [ ] Error rates < 0.1%
- [ ] Data consistency verified

**Blocking Conditions:**
- Traffic mirroring failures
- Response time degradation > 20%
- Error rate > 0.1%
- Data inconsistencies

### Gate 4: Limited Access Operational
**Criteria:**
- [ ] Whitelisted users operational
- [ ] KPIs within bounds
- [ ] Error budgets healthy
- [ ] Risk limits enforced

**Blocking Conditions:**
- User access failures
- KPI violations
- Error budget exhaustion
- Risk limit breaches

### SLO Definitions

#### Trading SLOs
- **Order Accept to Match:** p95 ≤ 10ms (staging), p95 ≤ 50ms (production)
- **Trade Execution Success:** ≥ 99.9%
- **Order Book Depth:** Real-time updates within 100ms

#### WebSocket SLOs
- **Message Delivery:** p95 ≤ 150ms
- **Reconnection Success:** ≥ 99.9%
- **Dropped Messages:** ≤ 0.05%

#### Risk SLOs
- **Snapshot Freshness:** p95 ≤ 1s
- **Overnight VaR Completion:** By 06:00 with ≥ 99.5% success
- **Stress Test Completion:** Within 30 minutes

#### Compliance SLOs
- **Scheduled Report Generation:** ≥ 99.9% on-time
- **Real-time Alert Latency:** p95 ≤ 1s
- **Data Completeness:** 100%

---

## Rollback Plan

### Rollback Triggers
- **Immediate Rollback:**
  - Data corruption detected
  - Regulatory compliance violation
  - Security breach
  - Complete system failure

- **Gradual Rollback:**
  - Performance degradation > 20%
  - Error rate > 1%
  - Risk calculation failures
  - User complaints > 10%

### Rollback Procedure

#### 1. Emergency Rollback (5 minutes)
```bash
# Emergency rollback to previous version
make emergency-rollback
make validate-rollback
make notify-stakeholders
```

**Steps:**
1. Stop all incoming traffic
2. Revert to previous deployment
3. Validate system health
4. Restore previous configuration
5. Resume traffic

#### 2. Controlled Rollback (15 minutes)
```bash
# Controlled rollback with traffic draining
make drain-traffic
make controlled-rollback
make validate-rollback
make restore-traffic
```

**Steps:**
1. Drain traffic to 0%
2. Revert deployment
3. Validate health
4. Gradually restore traffic
5. Monitor stability

#### 3. Feature Flag Rollback
```yaml
# Rollback feature flags
FEATURE_FLAGS:
  LIVE_DATA_PROVIDERS: false          # Revert to mock
  AI_RISK_SCORING: false             # Disable AI risk
  REAL_TIME_COMPLIANCE: false        # Revert to batch
  ADVANCED_ORDER_TYPES: false        # Disable advanced orders
  RISK_LIMITS_ENFORCEMENT: true      # Keep strict limits
```

#### 4. Configuration Rollback
```bash
# Rollback configuration changes
make rollback-config
make validate-config
```

**Rollback Items:**
- Load balancer configuration
- Database connection settings
- Redis cluster configuration
- Monitoring thresholds

### Post-Rollback Validation
```bash
# Post-rollback validation
make validate-system-health
make check-data-integrity
make verify-compliance
make run-smoke-tests
```

**Validation Checks:**
- [ ] System operational
- [ ] Data integrity maintained
- [ ] Compliance verified
- [ ] Performance restored
- [ ] User access functional

---

## Communications

### Stakeholder Matrix
| Role | Name | Contact | Escalation Path |
|------|------|---------|------------------|
| **Project Sponsor** | CTO | cto@bondx.com | CEO |
| **Technical Lead** | Head of Engineering | eng@bondx.com | CTO |
| **Risk Owner** | Head of Risk | risk@bondx.com | CRO |
| **Compliance Owner** | Head of Compliance | compliance@bondx.com | CCO |
| **Operations Lead** | DevOps Manager | ops@bondx.com | Head of Engineering |
| **Business Lead** | Head of Trading | trading@bondx.com | CTO |

### Status Update Cadence
- **Hourly:** During critical phases (deployment, validation)
- **4-Hourly:** During limited access phase
- **Daily:** During general availability rollout
- **Weekly:** Post-go-live status review

### Communication Templates

#### Go-Live Status Update
```
Subject: BondX Go-Live Status Update - [Phase] - [Date/Time]

Status: [Green/Amber/Red]
Phase: [Current Phase]
Duration: [Time Elapsed]
Next Milestone: [Next Phase]

Key Metrics:
- [Metric 1]: [Value] ([Target])
- [Metric 2]: [Value] ([Target])

Issues/Concerns:
- [Issue 1] - [Status]
- [Issue 2] - [Status]

Next Update: [Time]
```

#### Incident Communication
```
Subject: URGENT: BondX Go-Live Incident - [Severity] - [Date/Time]

Incident Summary:
[Brief description of the incident]

Impact:
- [Impact 1]
- [Impact 2]

Actions Taken:
- [Action 1]
- [Action 2]

Next Steps:
- [Next step 1]
- [Next step 2]

Contact: [Primary Contact] - [Phone]
Escalation: [Escalation Contact] - [Phone]
```

---

## Appendices

### Appendix A: Smoke Test Checklist
```bash
# Pre-deployment smoke tests
make smoke-test-staging
make smoke-test-production
make validate-smoke-results
```

**Smoke Test Items:**
- [ ] Health check endpoints
- [ ] Database connectivity
- [ ] Redis connectivity
- [ ] Basic API functionality
- [ ] WebSocket connection
- [ ] Risk calculation
- [ ] Compliance reporting

### Appendix B: WebSocket Reconnection Test
```bash
# WebSocket resilience testing
make test-ws-reconnection
make validate-ws-resilience
```

**Test Scenarios:**
- Network interruption (5s, 30s, 2min)
- Service restart
- Load balancer failover
- Client reconnection

### Appendix C: Redis Failover Drill
```bash
# Redis failover testing
make test-redis-failover
make validate-redis-recovery
```

**Test Steps:**
1. Stop Redis master
2. Verify failover to replica
3. Validate data consistency
4. Restore master
5. Verify cluster health

### Appendix D: Database Replica Lag Drill
```bash
# Database performance testing
make test-db-replica-lag
make validate-db-performance
```

**Test Scenarios:**
- Heavy write load
- Replica network issues
- Backup operations
- Maintenance windows

---

## Document Control

| Version | Date | Author | Changes | Approver |
|---------|------|--------|---------|----------|
| 1.0 | $(date) | DevOps Team | Initial version | CTO |
| 1.1 | TBD | TBD | TBD | TBD |

**Next Review Date:** [Date]  
**Reviewers:** [Names]  
**Distribution:** [List]
