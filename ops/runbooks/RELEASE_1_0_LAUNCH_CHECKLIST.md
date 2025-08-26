# BondX Release 1.0 Launch Checklist
## Single Source of Truth for Go-Live

**Document Version:** 1.0  
**Last Updated:** $(Get-Date -Format "yyyy-MM-dd")  
**Owner:** Release Manager  
**Approval Required:** CTO, Head of Risk, Head of Compliance  

---

## 📋 Executive Summary

This document consolidates all go-live activities into a single, executable checklist. It references existing artifacts under `ops/runbooks/` and `deploy/` while providing decision gates, timelines, and rollback procedures.

**Critical Path:** Staging Soak → Dark Launch → Limited Access → General Availability  
**Total Timeline:** 7-10 days  
**Rollback SLA:** 15 minutes from decision to traffic cutoff  

---

## 🎯 Decision Log

### Final Go/No-Go Approvers
- **CTO:** Final technical approval
- **Head of Risk:** Risk assessment sign-off
- **Head of Compliance:** Regulatory compliance approval
- **Head of Operations:** Operational readiness approval

### Change Window
- **Primary:** Sunday 02:00-06:00 IST (lowest trading activity)
- **Fallback:** Saturday 22:00-02:00 IST
- **Emergency:** Any time with 2-hour notice

### Rollback Owner
- **Primary:** DevOps Lead
- **Backup:** Platform Engineer
- **Escalation:** CTO (if >30 minutes)

---

## 🚀 Pre-Flight Gates

### Gate 1: Staging Soak Results (48h minimum)
- [ ] **Synthetic Auctions:** 100+ successful auctions with 0% failure rate
- [ ] **Continuous Trading:** 24h continuous operation with <0.1% error rate
- [ ] **Risk Batches:** Overnight VaR completion <99.5% success rate
- [ ] **Compliance Exports:** All scheduled reports generated on-time
- [ ] **WebSocket Stickiness:** 99.9% reconnection success rate

**Reference:** `ops/runbooks/GO_LIVE_RUNBOOK.md#staging-soak`

### Gate 2: SLO Burn Rates Within Budget
- [ ] **Trading:** p95 accept→match ≤10ms (staging target)
- [ ] **WebSockets:** p95 deliver-to-receive ≤150ms
- [ ] **Risk:** Snapshot freshness p95 ≤1s
- [ ] **Compliance:** Report generation ≤99.9% on-time
- [ ] **Error Budgets:** All services within 5% monthly burn rate

**Reference:** `deploy/monitoring/slos.yaml`

### Gate 3: Data Freshness SLAs Green
- [ ] **NSE/BSE:** Market data <100ms stale
- [ ] **RBI:** Yield curves <5min stale
- [ ] **Ratings:** Credit updates <1h stale
- [ ] **Fallback Mechanisms:** Tested and verified
- [ ] **Data Lineage:** Provenance tags working

**Reference:** `ops/runbooks/DATA_ADAPTER_CUTOVER_PLAN.md`

### Gate 4: Security & RBAC Penetration Checks
- [ ] **Authentication:** JWT rotation tested
- [ ] **Authorization:** RBAC matrix validated
- [ ] **WAF Rules:** Edge security verified
- [ ] **Secrets Management:** Vault policies tested
- [ ] **Vulnerability Scan:** Zero critical findings

**Reference:** `deploy/security/SECURITY_HARDENING_SPRINT.md`

### Gate 5: Chaos Drills Completed
- [ ] **WebSocket Node Kill:** Recovery <30s
- [ ] **Redis Failover:** <10s switchover
- [ ] **DB Replica Lag:** Throttling working
- [ ] **Data Provider Outage:** Fallback tested
- [ ] **Load Burst:** Auto-scaling verified

**Reference:** `ops/playbooks/CHAOS_INCIDENT_DRILLS.md`

---

## ⏰ Launch Timeline

### Phase 1: Dark Launch (Day 1, 02:00-04:00 IST)
**Who:** DevOps Lead + Platform Engineer  
**What:** Bring up production stack with adapters in observe-only mode

**Timeline:**
- **02:00-02:15:** Infrastructure deployment
- **02:15-02:30:** Service startup and health checks
- **02:30-03:00:** Traffic mirroring from staging
- **03:00-04:00:** Metrics validation and baseline establishment

**Commands:**
```bash
make deploy-production-observable
make setup-traffic-mirroring
make validate-production-health
```

### Phase 2: Limited Access (Day 2-3, 09:00-17:00 IST)
**Who:** DevOps Lead + QA Lead  
**What:** Whitelisted users with conservative limits

**Timeline:**
- **Day 2:** Internal team testing (10 users max)
- **Day 3:** Early partner access (25 users max)
- **Daily:** KPI review and limit adjustments

**Commands:**
```bash
make enable-limited-access
make monitor-kpis
make validate-canary-data
```

### Phase 3: General Availability (Day 4, 09:00 IST)
**Who:** DevOps Lead + Operations Team  
**What:** Full platform access with monitoring

**Timeline:**
- **09:00:** Enable all features and limits
- **09:00-10:00:** Gradual traffic ramp-up
- **10:00-17:00:** Continuous monitoring
- **17:00:** Daily business KPI report

**Commands:**
```bash
make enable-advanced-features
make enable-real-time-compliance
make monitor-kpis
```

---

## 🔄 Rollback Matrix

### Failure Symptom → Rollback Action

| Symptom | Toggle to Flip | Max Time | Verification |
|---------|----------------|----------|--------------|
| **Data Adapter Stale** | `DATA_PROVIDER_MOCK_MODE=true` | 5 min | Data freshness metrics |
| **WebSocket Saturation** | `WS_COMPRESSION_ENABLED=true` | 3 min | Connection count |
| **Matching Latency >50ms** | `ORDER_QUEUE_LIMIT=1000` | 2 min | p95 latency metrics |
| **Risk Snapshot Delays** | `RISK_BATCH_SIZE=100` | 5 min | Snapshot completion time |
| **Compliance Report Failures** | `COMPLIANCE_ASYNC_MODE=true` | 10 min | Report generation success |

### Rollback Commands
```bash
# Emergency rollback
make emergency-rollback

# Feature-specific rollback
make rollback --feature=data-adapters
make rollback --feature=risk-engine
make rollback --feature=compliance

# Traffic drain
make drain-traffic
make restore-traffic
```

---

## 👀 Post-Launch Watch

### Dashboards to Keep Open (First 72h)
1. **Trading Performance:** `deploy/monitoring/grafana-dashboard-trading.json`
2. **WebSocket Health:** Connection count, message rates, reconnection success
3. **Risk Metrics:** VaR completion, snapshot freshness, limit breaches
4. **Infrastructure:** CPU, memory, Redis operations, DB QPS
5. **Business KPIs:** Order volume, trade success rate, user activity

### On-Call Rotation (First 72h)
- **Day 1:** DevOps Lead (24h coverage)
- **Day 2:** Platform Engineer + DevOps Lead (12h shifts)
- **Day 3:** Platform Engineer + Operations Engineer (12h shifts)

### Incident Triage Thresholds
- **P1 (Critical):** >5% error rate, >100ms latency, data staleness >1min
- **P2 (High):** 1-5% error rate, 50-100ms latency, data staleness 30s-1min
- **P3 (Medium):** <1% error rate, <50ms latency, data staleness <30s

### Daily Business KPI Report Recipients
- **Executives:** CTO, Head of Operations
- **Operations:** DevOps Lead, Platform Engineer
- **Business:** Head of Trading, Head of Risk
- **Compliance:** Head of Compliance

---

## 📚 References and Artifacts

### Core Runbooks
- **Go-Live Process:** `ops/runbooks/GO_LIVE_RUNBOOK.md`
- **Data Cutover:** `ops/runbooks/DATA_ADAPTER_CUTOVER_PLAN.md`
- **Chaos Drills:** `ops/playbooks/CHAOS_INCIDENT_DRILLS.md`
- **Security Hardening:** `deploy/security/SECURITY_HARDENING_SPRINT.md`
- **Business Validation:** `ops/runbooks/BUSINESS_VALIDATION_UAT.md`

### Monitoring and SLOs
- **SLO Definitions:** `deploy/monitoring/slos.yaml`
- **Trading Dashboard:** `deploy/monitoring/grafana-dashboard-trading.json`
- **Alert Rules:** `deploy/monitoring/prometheus-rules.yaml`

### Automation
- **Make Targets:** `Makefile` (deploy, rollback, monitoring)
- **PowerShell Scripts:** `ops/scripts/operational-commands.ps1`
- **Quick Reference:** `ops/README.md`

---

## ✅ Checklist Execution

### Pre-Launch (T-7 days)
- [ ] All pre-flight gates completed
- [ ] Stakeholder approvals obtained
- [ ] Communication plan distributed
- [ ] Rollback team briefed
- [ ] Monitoring dashboards configured

### Launch Day
- [ ] Dark launch completed successfully
- [ ] Metrics baseline established
- [ ] Limited access enabled
- [ ] User feedback collected
- [ ] Daily KPI report generated

### Post-Launch (Day 1-7)
- [ ] General availability achieved
- [ ] All SLOs within targets
- [ ] Business KPIs meeting expectations
- [ ] Incident response tested
- [ ] Post-mortem completed

---

## 🚨 Emergency Contacts

| Role | Name | Phone | Escalation |
|------|------|-------|------------|
| **DevOps Lead** | [Name] | [Phone] | 15 min |
| **Platform Engineer** | [Name] | [Phone] | 30 min |
| **CTO** | [Name] | [Phone] | 1 hour |
| **Head of Operations** | [Name] | [Phone] | 2 hours |

---

## 📝 Change Log

| Date | Change | Author | Approval |
|------|--------|--------|----------|
| [Date] | Initial version | [Name] | [Name] |
| [Date] | Updated rollback matrix | [Name] | [Name] |

---

**Next Review:** Weekly during launch preparation  
**Final Approval:** 24h before launch window

