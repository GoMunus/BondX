# BondX Operational Framework
## Comprehensive Go-Live and Production Operations

This directory contains the complete operational framework for BondX production deployment, including go-live runbooks, security hardening, UAT scripts, and operational procedures.

---

## 📋 Table of Contents

1. [Go-Live Runbook](#go-live-runbook)
2. [Data Adapter Cutover Plan](#data-adapter-cutover-plan)
3. [SLOs and Monitoring](#slos-and-monitoring)
4. [Chaos Engineering & Drills](#chaos-engineering--drills)
5. [Security Hardening](#security-hardening)
6. [Business Validation & UAT](#business-validation--uat)
7. [Release 1.0 Launch Checklist](#release-10-launch-checklist)
8. [Production Runbooks Index](#production-runbooks-index)
9. [Business Validation Sprint Pack](#business-validation-sprint-pack)
10. [Data Integrity and Reconciliation](#data-integrity-and-reconciliation)
11. [Capacity Guardrails and Auto-Tuning](#capacity-guardrails-and-auto-tuning)
12. [Customer Status Page and Uptime Policy](#customer-status-page-and-uptime-policy)
13. [Compliance Evidence Binder](#compliance-evidence-binder)
14. [Production Operator Handbook](#production-operator-handbook)
15. [Vendor and Partner Integration Kit](#vendor-and-partner-integration-kit)
16. [Post-GA Continuous Improvement Backlog](#post-ga-continuous-improvement-backlog)
17. [Quick Start Guide](#quick-start-guide)
18. [Make Targets Reference](#make-targets-reference)

---

## 🚀 Go-Live Runbook

**File:** `runbooks/GO_LIVE_RUNBOOK.md`

The comprehensive go-live runbook covers:
- **4-Phase Promotion Strategy:** Staging soak → Dark launch → Limited access → General availability
- **Validation Gates:** Explicit criteria that must be met before proceeding
- **SLOs & Error Budgets:** Service level objectives with burn-rate alerts
- **Rollback Procedures:** Emergency and controlled rollback scenarios
- **Communication Templates:** Stakeholder updates and incident communications

**Key Commands:**
```bash
# Deploy to staging
make deploy-staging

# Deploy production in observable mode
make deploy-production-observable

# Promote staging to production
make promote-prod

# Emergency rollback
make emergency-rollback
```

---

## 🔄 Data Adapter Cutover Plan

**File:** `runbooks/DATA_ADAPTER_CUTOVER_PLAN.md`

Comprehensive plan for transitioning from mock to live data providers:
- **Provider Matrix:** NSE, BSE, RBI, CRISIL, ICRA, CARE specifications
- **3-Phase Cutover:** Canary → Batch rollout → Full enablement
- **Fallback Mechanisms:** Provider A → B → Mock with runtime switching
- **Compliance Requirements:** SEBI/RBI compliance with audit trails

**Key Commands:**
```bash
# Validate canary deployment
make validate-canary-data

# Check provider health
make check-provider-health

# Enable live data providers
make enable-live-data-providers
```

---

## 📊 SLOs and Monitoring

**Files:** 
- `deploy/monitoring/slos.yaml`
- `deploy/monitoring/grafana-dashboard-trading.json`

Service Level Objectives with burn-rate alerts:
- **Trading SLOs:** Order accept→match ≤50ms, execution success ≥99.9%
- **WebSocket SLOs:** Message delivery ≤150ms, reconnection ≥99.9%
- **Risk SLOs:** Snapshot freshness ≤1s, VaR completion by 06:00
- **Compliance SLOs:** Report generation ≥99.9% on-time

**Key Commands:**
```bash
# Monitor KPIs
make monitor-kpis

# Check error budgets
make check-error-budgets

# Validate SLOs
make validate-slos
```

---

## 🧪 Chaos Engineering & Drills

**File:** `playbooks/CHAOS_INCIDENT_DRILLS.md`

Resiliency testing and incident response practice:
- **Infrastructure Drills:** WebSocket failure, Redis failover, DB replica lag
- **Data Provider Drills:** Single/multiple provider outages
- **Load Testing:** Order bursts, resource pressure scenarios
- **Incident Response:** Templates and escalation procedures

**Key Commands:**
```bash
# Test WebSocket resilience
make test-websocket-resilience

# Test Redis failover
make test-redis-failover

# Run load tests
make test-load
```

---

## 📋 Release 1.0 Launch Checklist

**File:** `runbooks/RELEASE_1_0_LAUNCH_CHECKLIST.md`

The single source of truth for the go-live process:
- **Decision Log:** Final go/no-go approvers and change windows
- **Pre-Flight Gates:** 5 validation gates with explicit criteria
- **Launch Timeline:** Exact timestamps for dark launch, limited access, GA
- **Rollback Matrix:** Failure symptom → action mapping with time limits
- **Post-Launch Watch:** Dashboards, on-call rotation, incident thresholds

**Key Commands:**
```bash
# Deploy production in observable mode
make deploy-production-observable

# Setup traffic mirroring
make setup-traffic-mirroring

# Enable limited access
make enable-limited-access

# Monitor KPIs during launch
make monitor-kpis
```

---

## 🚨 Production Runbooks Index

**File:** `runbooks/PRODUCTION_RUNBOOKS_INDEX.md`

Quick-action cards for critical scenarios with immediate response procedures:
- **Degraded WebSockets:** Scale-out, compression, stickiness verification
- **Redis Failover:** Force failover, validate master, restart consumers
- **Replica Lag:** Throttle reads, re-route critical services
- **Data Provider Outage:** Activate fallback, enable mock mode
- **Risk Alerts Storm:** Adjust thresholds, rate limiting, scaling

**Key Commands:**
```bash
# Emergency rollback
make emergency-rollback

# Scale services
make scale-up --service=websocket-manager --replicas=8

# Check service health
make health-check --service=risk-engine

# Monitor metrics
make validate-production-metrics --component=websocket
```

---

## 🔒 Security Hardening

**File:** `deploy/security/SECURITY_HARDENING_SPRINT.md`

Production security controls implementation:
- **Idempotency & Replay Protection:** Prevent duplicate operations
- **JWT Key Management:** 24-hour rotation with secure revocation
- **RBAC Implementation:** Role-based access control with granular permissions
- **WAF & Edge Security:** Authentication bypass, injection protection
- **Secrets Management:** HashiCorp Vault integration with auto-rotation

**Key Commands:**
```bash
# Run security tests
make run-pentest
make run-dast-scan
make run-sast-scan

# Validate compliance
make validate-audit-trail
make generate-compliance-report
```

---

## ✅ Business Validation & UAT

**File:** `runbooks/BUSINESS_VALIDATION_UAT.md`

User acceptance testing and business validation:
- **End-to-End Scenarios:** Auction → Allocation → Settlement → Portfolio → Risk → Compliance
- **Trading Scenarios:** Partial fills, order modifications, cancellations
- **Risk Management:** Limit breaches, alert generation, mitigation
- **Data Integration:** Provider outages, fallback mechanisms
- **Scoring Rubric:** Functional and performance scoring criteria

**Key Commands:**
```bash
# Run full UAT suite
make run-uat-suite

# Run individual test modules
make run-trading-flow-test
make run-risk-management-test

# Generate UAT reports
make generate-uat-report
```

---

## 📋 Business Validation Sprint Pack

**File:** `runbooks/BUSINESS_VALIDATION_SPRINT.md`

Comprehensive 2-week stakeholder validation plan:
- **User Cohorts:** Traders, Market Makers, Risk Management, Operations
- **Daily Scenarios:** 40 scenarios across 10 business days
- **Acceptance Evidence:** Screenshots, logs, data exports, user feedback
- **Exit Criteria:** 95%+ pass rate, zero P1 defects, stakeholder sign-off

**Key Commands:**
```bash
# Run business validation suite
make run-business-validation

# Run specific scenario categories
make run-validation --category=trading
make run-validation --category=risk
make run-validation --category=compliance

# Generate validation reports
make generate-validation-report
```

---

## 🔍 Data Integrity and Reconciliation

**File:** `runbooks/DATA_INTEGRITY_RECONCILIATION.md`

Comprehensive data validation and reconciliation framework:
- **Trade vs Position vs Cash:** Real-time and end-of-day reconciliation
- **Auction Settlement:** Allocations, fills, settlement ledger validation
- **Market Data Quality:** Provider feeds, internal quotes, data freshness
- **Risk Calculations:** Snapshot validation, activity reconciliation

**Key Commands:**
```bash
# Run daily reconciliation
make reconcile-daily

# Reconcile specific domains
make reconcile-trades
make reconcile-auctions
make validate-data
make reconcile-risk

# Monitor real-time
make monitor-realtime

# Generate reports
make generate-report
```

---

## 💰 Capacity Guardrails and Auto-Tuning

**File:** `runbooks/CAPACITY_GUARDRAILS_AUTO_TUNING.md`

Intelligent capacity management with auto-tuning and cost optimization:
- **KPI to Scaling Mapping:** WebSocket connections, message throughput, order processing, auction volume, VaR runtime
- **Safe Maximums and Soft Caps:** Resource limits for all services with emergency thresholds
- **Auto-Tuning Suggestions:** Cache TTL optimization, snapshot coalescing, market maker aggressiveness
- **Cost Mode Toggle:** Nights/weekends optimization with rollback procedures

**Key Commands:**
```bash
# Check capacity status
make capacity-status

# Enable/disable cost mode
make enable-cost-mode
make disable-cost-mode

# Rollback cost mode if needed
make rollback-cost-mode

# Optimize resources
make optimize-resources

# Generate capacity reports
make capacity-report
```

---

## 🌐 Customer Status Page and Uptime Policy

**File:** `runbooks/CUSTOMER_STATUS_PAGE_UPTIME_POLICY.md`

Public-facing system status and incident communication framework:
- **Status Page Components:** Trading platform, WebSocket services, risk management, compliance
- **Incident Taxonomy:** P1-P4 classification with SLA commitments and communication protocols
- **Maintenance Windows:** Scheduled maintenance with advance notice and communication
- **Communication Channels:** Webhooks, RSS feeds, email subscriptions, postmortem workflow

**Key Commands:**
```bash
# Deploy status page updates
make status-page-deploy

# Manage incidents
make incident-create
make incident-update

# Schedule maintenance
make maintenance-schedule

# Publish postmortems
make postmortem-publish

# Test webhooks
make webhook-test
```

---

## 🚀 Quick Start Guide

### 1. Pre-Deployment Setup
```bash
# Check prerequisites
make check-prerequisites

# Validate environment
make validate-test-environment

# Run security scans
make run-security-scan
```

### 2. Staging Deployment
```bash
# Deploy to staging
make deploy-staging

# Run UAT tests
make run-uat-suite

# Validate staging health
make validate-staging-health
```

### 3. Production Promotion
```bash
# Deploy production in observable mode
make deploy-production-observable

# Set up traffic mirroring
make setup-traffic-mirroring

# Validate production metrics
make validate-production-metrics
```

### 4. Go-Live Execution
```bash
# Enable limited access
make enable-limited-access

# Monitor KPIs
make monitor-kpis

# Gradually increase limits
make enable-live-data-providers
make enable-real-time-compliance
```

### 5. Post-Go-Live Validation
```bash
# Run production smoke tests
make run-production-smoke-tests

# Validate system health
make validate-production-readiness

# Generate compliance report
make report-compliance
```

---

## 🛠️ Make Targets Reference

### Deployment Targets
```bash
make deploy-staging              # Deploy to staging
make deploy-production-observable # Deploy production in observable mode
make promote-prod                 # Promote staging to production
make rollback                     # Rollback deployment
make emergency-rollback          # Emergency rollback (stops traffic)
```

### Feature Management
```bash
make enable-live-data-providers  # Enable live data providers
make enable-real-time-compliance # Enable real-time compliance
make enable-advanced-features    # Enable advanced order types
```

### Validation & Monitoring
```bash
make validate-production-health  # Validate production health
make validate-canary-data        # Validate canary deployment
make monitor-kpis                # Monitor key performance indicators
make check-error-budgets         # Check error budget status
```

### Testing & Drills
```bash
make test-websocket-resilience  # Test WebSocket resilience
make test-redis-failover        # Test Redis failover
make test-load                  # Run load tests
make run-uat-suite              # Run full UAT suite
```

### Security & Compliance
```bash
make run-pentest                 # Run penetration tests
make run-dast-scan              # Run DAST scans
make run-sast-scan              # Run SAST scans
make validate-audit-trail        # Validate audit trail
make generate-compliance-report  # Generate compliance report
```

### Traffic Management
```bash
make setup-traffic-mirroring    # Set up traffic mirroring
make drain-traffic               # Drain traffic to 0%
make restore-traffic             # Restore traffic to normal
```

---

## 📁 Directory Structure

```
ops/
├── README.md                           # This file
├── runbooks/
│   ├── GO_LIVE_RUNBOOK.md             # Go-live procedures
│   ├── DATA_ADAPTER_CUTOVER_PLAN.md   # Data provider cutover
│   └── BUSINESS_VALIDATION_UAT.md     # UAT scripts and scenarios
├── playbooks/
│   └── CHAOS_INCIDENT_DRILLS.md       # Chaos engineering and drills
└── deploy/
    ├── monitoring/
    │   ├── slos.yaml                   # SLO definitions
    │   └── grafana-dashboard-trading.json # Trading dashboard
    └── security/
        └── SECURITY_HARDENING_SPRINT.md # Security controls
```

---

## 🔧 Configuration

### Environment Variables
```bash
export NAMESPACE=bondx
export ENVIRONMENT=production
export KUBECONFIG=$HOME/.kube/config
export REPLICAS=3
```

### Feature Flags
```yaml
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

---

## 📞 Support and Escalation

### On-Call Rotation
- **Primary:** DevOps Engineer (24/7)
- **Secondary:** DevOps Manager (Business Hours)
- **Escalation:** Head of Engineering → CTO

### Emergency Contacts
- **DevOps:** ops@bondx.com
- **Engineering:** eng@bondx.com
- **Security:** security@bondx.com
- **Compliance:** compliance@bondx.com

### Communication Channels
- **Immediate:** Phone, SMS
- **Urgent:** Slack, Email
- **Standard:** Email, Dashboard
- **Informational:** Dashboard, Reports

---

## 📚 Additional Resources

- [BondX API Documentation](https://api.bondx.com/docs)
- [Trading Engine Documentation](https://docs.bondx.com/trading)
- [Risk Management Guide](https://docs.bondx.com/risk)
- [Compliance Framework](https://docs.bondx.com/compliance)
- [Security Policy](https://docs.bondx.com/security)

---

## 🤝 Vendor and Partner Integration Kit

**File:** `runbooks/VENDOR_PARTNER_INTEGRATION_KIT.md`

Comprehensive integration guide for external partners and vendors:
- **API Contracts:** Detailed endpoint specifications with rate limits and authentication
- **Security Requirements:** TLS standards, authentication, IP whitelisting, and compliance
- **Certification Process:** Pre-integration checklists, testing procedures, and validation
- **Support Framework:** Multi-tier support with SLAs and escalation procedures
- **Integration Timeline:** 8-week phased approach from planning to go-live

**Key Features:**
- Standardized API formats and error handling
- Comprehensive security and compliance requirements
- Automated testing and certification procedures
- Dedicated support channels with defined SLAs
- Success metrics and KPI tracking

---

## 📈 Post-GA Continuous Improvement Backlog

**File:** `runbooks/POST_GA_CONTINUOUS_IMPROVEMENT_BACKLOG.md`

Prioritized roadmap for post-General Availability improvements:
- **Reliability (P1):** WebSocket pooling, backpressure control, circuit breakers
- **Performance (P1):** Dynamic partitioning, hot-path optimization, multi-level caching
- **Security (P1):** Red-team drills, credential scanning, zero-trust architecture
- **Product (P1):** User analytics, feature flags, experimentation framework
- **ML Ops (P1):** Drift detection, shadow deployments, auto-retraining

**Implementation Timeline:**
- **Q1 (Weeks 1-12):** Critical reliability and performance improvements
- **Q2 (Weeks 13-24):** Security enhancements and advanced features
- **Q3 (Weeks 25-36):** Advanced ML Ops and experimentation capabilities

**Success Metrics:**
- System uptime >99.95%, error rate <0.05%
- Order matching latency <10ms p95, throughput >10,000 orders/second
- User activation rate >80%, retention rate >70%
- Zero critical security breaches, compliance score >95%

---

## 🏷️ Version Control

| Component | Version | Last Updated | Owner |
|-----------|---------|--------------|-------|
| Go-Live Runbook | 1.0 | $(date) | DevOps Team |
| Data Cutover Plan | 1.0 | $(date) | Data Engineering Team |
| SLOs & Monitoring | 1.0 | $(date) | DevOps Team |
| Chaos Engineering | 1.0 | $(date) | DevOps Team |
| Security Hardening | 1.0 | $(date) | Security Team |
| Business Validation | 1.0 | $(date) | Business Analysis Team |
| Release 1.0 Launch Checklist | 1.0 | $(date) | DevOps Team |
| Production Runbooks Index | 1.0 | $(date) | DevOps Team |
| Business Validation Sprint Pack | 1.0 | $(date) | Business Analysis Team |
| Data Integrity & Reconciliation | 1.0 | $(date) | Data Engineering Team |
| Capacity Guardrails | 1.0 | $(date) | DevOps Team |
| Customer Status Page | 1.0 | $(date) | Product Team |
| Compliance Evidence Binder | 1.0 | $(date) | Compliance Team |
| Production Operator Handbook | 1.0 | $(date) | DevOps Team |
| Vendor Integration Kit | 1.0 | $(date) | Business Development Team |
| Continuous Improvement Backlog | 1.0 | $(date) | Engineering Team |

---

## 📝 Contributing

To contribute to this operational framework:

1. **Create a feature branch** from `main`
2. **Make your changes** following the established patterns
3. **Update version numbers** and changelog
4. **Submit a pull request** with detailed description
5. **Get approval** from relevant stakeholders
6. **Merge after review** and testing

---

## ⚠️ Important Notes

- **Never run chaos experiments in production without approval**
- **Always have rollback procedures ready**
- **Monitor systems closely during all operations**
- **Document all changes and observations**
- **Follow the established escalation procedures**
- **Maintain compliance throughout all operations**

---

**Last Updated:** $(date)  
**Next Review:** [Date]  
**Maintainers:** DevOps Team, Security Team, Business Analysis Team
