# BondX Production Operator Handbook
## Day-1/Week-1 Operational Guide

**Document Version:** 1.0  
**Last Updated:** $(Get-Date -Format "yyyy-MM-dd")  
**Owner:** Operations Team  
**Audience:** Production Operators, SREs, DevOps Engineers  
**Scope:** Day-1 operations, monitoring, incident response, daily procedures  

---

## 🎯 Executive Summary

This handbook provides production operators with essential knowledge and procedures for managing BondX production systems. It covers critical monitoring dashboards, common alerts, incident response procedures, and daily operational tasks to ensure smooth platform operations.

**Key Objectives:**
- Enable operators to quickly understand system health
- Provide clear procedures for common scenarios
- Establish monitoring and alerting best practices
- Ensure consistent operational procedures

---

## 🚀 Day-1 Onboarding Checklist

### Pre-Shift Preparation
```yaml
pre_shift_checklist:
  system_access:
    - [ ] VPN access configured and tested
    - [ ] Production environment access granted
    - [ ] Monitoring dashboards accessible
    - [ ] Incident management tools configured
    - [ ] Communication channels established
    
  documentation_review:
    - [ ] Runbooks and playbooks reviewed
    - [ ] Emergency contact list verified
    - [ ] Escalation procedures understood
    - [ ] Change management process reviewed
    - [ ] Compliance requirements understood
    
  team_introduction:
    - [ ] Introduced to on-call team
    - [ ] Handover procedures reviewed
    - [ ] Communication protocols established
    - [ ] Escalation paths confirmed
    - [ ] Backup contacts identified
```

### First Hour Activities
```yaml
first_hour_activities:
  system_overview:
    - "Review current system status"
    - "Check for active incidents"
    - "Review recent alerts and notifications"
    - "Understand current system load"
    - "Identify any ongoing maintenance"
    
  monitoring_setup:
    - "Configure monitoring dashboards"
    - "Set up alert notifications"
    - "Configure incident management tools"
    - "Test communication channels"
    - "Verify backup procedures"
```

---

## 📊 The "Ten Things to Check" Dashboard List

### 1. Overall System Health Dashboard
**Location:** `https://grafana.bondx.com/d/system-health`  
**Check Frequency:** Every 15 minutes  
**Critical Thresholds:**
- **System Status:** All services should be GREEN
- **Uptime:** >99.9% for all critical services
- **Response Time:** <100ms p95 for all APIs
- **Error Rate:** <0.1% for all services

**Expected Values:**
```yaml
system_health_expected:
  overall_status: "HEALTHY"
  services_healthy: "100%"
  uptime_24h: "99.95%"
  response_time_p95: "75ms"
  error_rate: "0.05%"
```

### 2. Trading Performance Dashboard
**Location:** `https://grafana.bondx.com/d/trading-performance`  
**Check Frequency:** Every 5 minutes during trading hours  
**Critical Thresholds:**
- **Order Processing:** <50ms p95 order accept to execution
- **Trade Success Rate:** >99.9%
- **Order Book Depth:** Sufficient liquidity for all instruments
- **Market Data Freshness:** <1 minute stale

**Expected Values:**
```yaml
trading_performance_expected:
  order_latency_p95: "35ms"
  trade_success_rate: "99.95%"
  order_book_depth: "Sufficient"
  market_data_freshness: "30 seconds"
  active_orders: "1000-5000"
```

### 3. WebSocket Connection Dashboard
**Location:** `https://grafana.bondx.com/d/websocket-health`  
**Check Frequency:** Every 2 minutes  
**Critical Thresholds:**
- **Active Connections:** <80% of capacity
- **Connection Success Rate:** >99.9%
- **Message Delivery Latency:** <150ms p95
- **Reconnection Success Rate:** >99.5%

**Expected Values:**
```yaml
websocket_health_expected:
  active_connections: "5000-8000"
  connection_success_rate: "99.95%"
  message_latency_p95: "100ms"
  reconnection_success_rate: "99.8%"
  dropped_messages: "<0.01%"
```

### 4. Risk Management Dashboard
**Location:** `https://grafana.bondx.com/d/risk-management`  
**Check Frequency:** Every 10 minutes  
**Critical Thresholds:**
- **VaR Calculation Time:** <1 second p95
- **Risk Snapshot Frequency:** Every 5 minutes
- **Limit Breaches:** 0 active breaches
- **Portfolio Risk:** Within acceptable ranges

**Expected Values:**
```yaml
risk_management_expected:
  var_calculation_time_p95: "800ms"
  risk_snapshot_frequency: "5 minutes"
  active_limit_breaches: "0"
  portfolio_var_95_1d: "Within limits"
  concentration_risk: "LOW"
```

### 5. Database Performance Dashboard
**Location:** `https://grafana.bondx.com/d/database-performance`  
**Check Frequency:** Every 5 minutes  
**Critical Thresholds:**
- **Query Response Time:** <100ms p95
- **Connection Pool Utilization:** <80%
- **Replica Lag:** <1 second
- **Transaction Throughput:** >1000 TPS

**Expected Values:**
```yaml
database_performance_expected:
  query_response_time_p95: "75ms"
  connection_pool_utilization: "60-70%"
  replica_lag: "<500ms"
  transaction_throughput: "1500 TPS"
  active_connections: "200-300"
```

### 6. Redis Cache Dashboard
**Location:** `https://grafana.bondx.com/d/redis-cache`  
**Check Frequency:** Every 3 minutes  
**Critical Thresholds:**
- **Cache Hit Rate:** >95%
- **Memory Utilization:** <80%
- **Response Time:** <10ms p95
- **Connection Count:** <80% of limit

**Expected Values:**
```yaml
redis_cache_expected:
  cache_hit_rate: "97-99%"
  memory_utilization: "60-70%"
  response_time_p95: "5ms"
  active_connections: "100-150"
  keyspace_hits: "High"
```

### 7. Infrastructure Resources Dashboard
**Location:** `https://grafana.bondx.com/d/infrastructure`  
**Check Frequency:** Every 5 minutes  
**Critical Thresholds:**
- **CPU Utilization:** <80% per node
- **Memory Utilization:** <85% per node
- **Disk Usage:** <80% per volume
- **Network I/O:** <70% of capacity

**Expected Values:**
```yaml
infrastructure_expected:
  cpu_utilization: "40-60%"
  memory_utilization: "50-70%"
  disk_usage: "60-70%"
  network_io: "30-50%"
  pod_count: "50-80"
```

### 8. Data Provider Health Dashboard
**Location:** `https://grafana.bondx.com/d/data-providers`  
**Check Frequency:** Every 2 minutes  
**Critical Thresholds:**
- **Data Freshness:** <1 minute for critical feeds
- **API Response Time:** <500ms p95
- **Success Rate:** >99.5%
- **Fallback Status:** Primary providers active

**Expected Values:**
```yaml
data_providers_expected:
  data_freshness: "30 seconds"
  api_response_time_p95: "200ms"
  success_rate: "99.8%"
  fallback_status: "Not active"
  active_providers: "All primary"
```

### 9. Compliance and Reporting Dashboard
**Location:** `https://grafana.bondx.com/d/compliance`  
**Check Frequency:** Every 15 minutes  
**Critical Thresholds:**
- **Report Generation:** 100% on-time
- **Audit Trail Completeness:** 100%
- **Regulatory Submissions:** All successful
- **Data Retention Compliance:** 100%

**Expected Values:**
```yaml
compliance_expected:
  report_generation: "100% on-time"
  audit_trail_completeness: "100%"
  regulatory_submissions: "All successful"
  data_retention_compliance: "100%"
  last_audit: "Within 24 hours"
```

### 10. Business Metrics Dashboard
**Location:** `https://grafana.bondx.com/d/business-metrics`  
**Check Frequency:** Every 30 minutes  
**Critical Thresholds:**
- **Trading Volume:** Within expected ranges
- **User Activity:** Normal patterns
- **Revenue Metrics:** Meeting targets
- **Customer Satisfaction:** >95%

**Expected Values:**
```yaml
business_metrics_expected:
  trading_volume: "Within expected range"
  active_users: "100-200 during trading hours"
  revenue_metrics: "Meeting targets"
  customer_satisfaction: "96-98%"
  market_share: "Growing"
```

---

## 🚨 Common Alerts and What They Mean

### High Severity Alerts (P1)

#### "Trading Engine Unavailable"
**Alert:** `Trading engine service is down`  
**What It Means:** The core trading functionality is completely unavailable  
**Immediate Actions:**
1. Check if it's a planned maintenance
2. Verify infrastructure health
3. Check for recent deployments
4. Escalate to platform team immediately

**Expected Resolution Time:** <15 minutes

#### "Database Connection Pool Exhausted"
**Alert:** `Database connection pool at 100% utilization`  
**What It Means:** All database connections are in use, new requests are being queued  
**Immediate Actions:**
1. Check for long-running queries
2. Verify database health
3. Check application logs for connection leaks
4. Consider scaling database connections

**Expected Resolution Time:** <10 minutes

#### "WebSocket Service Overloaded"
**Alert:** `WebSocket service at 95% capacity`  
**What It Means:** WebSocket service is approaching maximum capacity  
**Immediate Actions:**
1. Scale out WebSocket managers
2. Check for connection leaks
3. Verify client behavior
4. Monitor connection growth

**Expected Resolution Time:** <5 minutes

### Medium Severity Alerts (P2)

#### "High Order Latency"
**Alert:** `Order processing latency >100ms p95`  
**What It Means:** Order processing is slower than expected  
**Immediate Actions:**
1. Check system resources
2. Verify database performance
3. Check for high load
4. Monitor for degradation

**Expected Resolution Time:** <30 minutes

#### "Risk Calculation Delays"
**Alert:** `Risk calculation taking >2 seconds`  
**What It Means:** Risk calculations are slower than expected  
**Immediate Actions:**
1. Check risk engine resources
2. Verify data freshness
3. Check calculation complexity
4. Monitor for trends

**Expected Resolution Time:** <1 hour

#### "Data Provider Issues"
**Alert:** `Market data stale >2 minutes`  
**What It Means:** Market data is not being updated timely  
**Immediate Actions:**
1. Check provider API status
2. Verify network connectivity
3. Check fallback mechanisms
4. Monitor data quality

**Expected Resolution Time:** <15 minutes

### Low Severity Alerts (P3)

#### "High Memory Usage"
**Alert:** `Memory utilization >85%`  
**What It Means:** System memory usage is high but not critical  
**Immediate Actions:**
1. Monitor memory trends
2. Check for memory leaks
3. Verify scaling policies
4. Document for review

**Expected Resolution Time:** <4 hours

#### "Disk Space Warning"
**Alert:** `Disk usage >80%`  
**What It Means:** Disk space is getting low  
**Immediate Actions:**
1. Check log rotation
2. Verify backup cleanup
3. Monitor growth rate
4. Plan for expansion

**Expected Resolution Time:** <24 hours

---

## 🛠️ First-Response Playbooks

### Incident Response Template

#### Initial Response (0-5 minutes)
```yaml
initial_response:
  acknowledge_alert:
    - "Acknowledge the alert in monitoring system"
    - "Update incident status to 'Investigating'"
    - "Notify team lead if high severity"
    
  assess_impact:
    - "Determine affected services"
    - "Assess customer impact"
    - "Identify scope of issue"
    
  gather_information:
    - "Check monitoring dashboards"
    - "Review recent logs"
    - "Check for recent changes"
    - "Verify system status"
```

#### Investigation (5-15 minutes)
```yaml
investigation:
  root_cause_analysis:
    - "Check system metrics"
    - "Review error logs"
    - "Verify configuration"
    - "Check dependencies"
    
  impact_assessment:
    - "Number of affected users"
    - "Business impact"
    - "Data integrity"
    - "Compliance implications"
    
  mitigation_planning:
    - "Identify immediate actions"
    - "Plan recovery steps"
    - "Assess rollback options"
    - "Determine escalation need"
```

#### Resolution and Recovery (15+ minutes)
```yaml
resolution_recovery:
  immediate_actions:
    - "Implement immediate fixes"
    - "Restart services if needed"
    - "Scale resources if required"
    - "Activate fallbacks"
    
  recovery_verification:
    - "Verify fix effectiveness"
    - "Test functionality"
    - "Monitor metrics"
    - "Validate customer impact"
    
  post_incident:
    - "Document incident details"
    - "Update runbooks if needed"
    - "Schedule post-mortem"
    - "Implement preventive measures"
```

### Common Incident Scenarios

#### WebSocket Service Overload
```yaml
websocket_overload_response:
  symptoms:
    - "High connection count"
    - "Increased latency"
    - "Connection failures"
    - "Client complaints"
    
  immediate_actions:
    - "Scale out WebSocket managers"
    - "Enable compression"
    - "Check for connection leaks"
    - "Verify load balancer health"
    
  verification:
    - "Connection count <80% capacity"
    - "Latency <150ms p95"
    - "Success rate >99.9%"
    - "No client complaints"
```

#### Database Performance Issues
```yaml
database_performance_response:
  symptoms:
    - "High query latency"
    - "Connection pool exhaustion"
    - "Replica lag"
    - "Timeout errors"
    
  immediate_actions:
    - "Check for long-running queries"
    - "Verify connection pool settings"
    - "Check replica health"
    - "Review recent changes"
    
  verification:
    - "Query latency <100ms p95"
    - "Connection pool <80%"
    - "Replica lag <1s"
    - "No timeout errors"
```

#### Risk Engine Delays
```yaml
risk_engine_delay_response:
  symptoms:
    - "VaR calculation delays"
    - "Risk snapshot delays"
    - "Limit breach delays"
    - "Performance degradation"
    
  immediate_actions:
    - "Check risk engine resources"
    - "Verify data freshness"
    - "Check calculation complexity"
    - "Scale risk engine if needed"
    
  verification:
    - "VaR calculation <1s p95"
    - "Risk snapshots on time"
    - "Limit breaches detected promptly"
    - "Performance restored"
```

---

## 📍 Where to Find Logs, Traces, and Dashboards

### Log Locations and Access

#### Application Logs
```yaml
application_logs:
  trading_engine:
    location: "kubectl logs -n bondx -l app=trading-engine"
    retention: "30 days"
    log_level: "INFO"
    
  risk_engine:
    location: "kubectl logs -n bondx -l app=risk-engine"
    retention: "30 days"
    log_level: "INFO"
    
  websocket_manager:
    location: "kubectl logs -n bondx -l app=websocket-manager"
    retention: "30 days"
    log_level: "INFO"
    
  compliance_engine:
    location: "kubectl logs -n bondx -l app=compliance-engine"
    retention: "30 days"
    log_level: "INFO"
```

#### System Logs
```yaml
system_logs:
  kubernetes:
    location: "kubectl logs -n kube-system"
    retention: "7 days"
    
  infrastructure:
    location: "Cloud provider console"
    retention: "30 days"
    
  database:
    location: "kubectl logs -n bondx -l app=postgresql"
    retention: "30 days"
    
  cache:
    location: "kubectl logs -n bondx -l app=redis"
    retention: "30 days"
```

### Tracing and Correlation

#### Distributed Tracing
```yaml
distributed_tracing:
  jaeger_dashboard:
    url: "https://jaeger.bondx.com"
    access: "Production team access"
    
  trace_correlation:
    correlation_id: "X-Correlation-ID header"
    trace_id: "X-Trace-ID header"
    span_id: "X-Span-ID header"
    
  trace_examples:
    order_flow: "Order creation → validation → execution → settlement"
    risk_calculation: "Data input → calculation → validation → output"
    compliance_check: "Order → risk check → compliance check → approval"
```

#### Monitoring Dashboards
```yaml
monitoring_dashboards:
  grafana:
    url: "https://grafana.bondx.com"
    access: "Production team access"
    
  prometheus:
    url: "https://prometheus.bondx.com"
    access: "Production team access"
    
  alertmanager:
    url: "https://alertmanager.bondx.com"
    access: "Production team access"
    
  custom_dashboards:
    trading: "https://grafana.bondx.com/d/trading-overview"
    risk: "https://grafana.bondx.com/d/risk-overview"
    compliance: "https://grafana.bondx.com/d/compliance-overview"
    infrastructure: "https://grafana.bondx.com/d/infrastructure-overview"
```

---

## 📋 On-Call Etiquette and Procedures

### On-Call Responsibilities

#### Primary On-Call
```yaml
primary_oncall:
  responsibilities:
    - "First responder to all alerts"
    - "Initial investigation and assessment"
    - "Immediate mitigation actions"
    - "Escalation to appropriate teams"
    - "Incident documentation"
    
  response_times:
    p1_alerts: "<5 minutes"
    p2_alerts: "<15 minutes"
    p3_alerts: "<1 hour"
    p4_alerts: "<4 hours"
    
  escalation_triggers:
    - "Unable to resolve within SLA"
    - "Multiple services affected"
    - "Customer impact significant"
    - "Compliance implications"
    - "Security concerns"
```

#### Secondary On-Call
```yaml
secondary_oncall:
  responsibilities:
    - "Backup for primary on-call"
    - "Support for complex incidents"
    - "Review and validation"
    - "Knowledge sharing"
    
  activation_triggers:
    - "Primary on-call unavailable"
    - "Complex incident requiring support"
    - "Primary on-call escalation"
    - "Scheduled handover"
```

### Communication Protocols

#### Incident Communication
```yaml
incident_communication:
  internal_notification:
    - "Slack channel: #bondx-incidents"
    - "Email: incidents@bondx.com"
    - "Phone: Emergency contact list"
    
  customer_notification:
    - "Status page updates"
    - "Email notifications"
    - "In-app notifications"
    
  stakeholder_updates:
    - "Management team"
    - "Business teams"
    - "Compliance team"
    - "Legal team"
```

#### Handover Procedures
```yaml
handover_procedures:
  shift_handover:
    - "Review active incidents"
    - "Update incident status"
    - "Share context and progress"
    - "Hand over responsibilities"
    
  incident_handover:
    - "Incident summary"
    - "Current status"
    - "Actions taken"
    - "Next steps"
    - "Escalation paths"
```

---

## 📅 Daily Operational Procedures

### Start of Shift Checklist
```yaml
start_of_shift:
  system_review:
    - "Check overall system health"
    - "Review active incidents"
    - "Check for recent alerts"
    - "Verify monitoring status"
    
  handover_review:
    - "Review previous shift handover"
    - "Check incident progress"
    - "Verify action items"
    - "Understand current priorities"
    
  preparation:
    - "Configure monitoring tools"
    - "Set up communication channels"
    - "Review runbooks"
    - "Verify access to all systems"
```

### During Shift Activities
```yaml
during_shift:
  monitoring:
    - "Check dashboards every 15 minutes"
    - "Respond to alerts promptly"
    - "Monitor system trends"
    - "Validate metrics accuracy"
    
  maintenance:
    - "Execute scheduled maintenance"
    - "Perform health checks"
    - "Update documentation"
    - "Review and optimize processes"
    
  communication:
    - "Update incident status"
    - "Communicate with stakeholders"
    - "Coordinate with teams"
    - "Document activities"
```

### End of Shift Checklist
```yaml
end_of_shift:
  incident_handover:
    - "Update incident status"
    - "Document progress made"
    - "Identify next steps"
    - "Hand over responsibilities"
    
  documentation:
    - "Update runbooks if needed"
    - "Document lessons learned"
    - "Update procedures"
    - "Share knowledge with team"
    
  handover:
    - "Prepare handover summary"
    - "Schedule handover meeting"
    - "Transfer responsibilities"
    - "Ensure continuity"
```

---

## 🚨 Emergency Procedures

### Critical Incident Response
```yaml
critical_incident:
  immediate_actions:
    - "Assess immediate impact"
    - "Activate incident response team"
    - "Implement emergency procedures"
    - "Notify all stakeholders"
    
  escalation_path:
    - "Level 1: On-call operator"
    - "Level 2: Team lead"
    - "Level 3: Engineering manager"
    - "Level 4: CTO"
    
  communication_plan:
    - "Internal team notification"
    - "Customer communication"
    - "Regulatory notification"
    - "Media communication"
```

### Emergency Rollback Procedures
```yaml
emergency_rollback:
  triggers:
    - "Critical security vulnerability"
    - "Data integrity issues"
    - "Service unavailability"
    - "Compliance violations"
    
  procedures:
    - "Stop all deployments"
    - "Activate rollback automation"
    - "Verify rollback success"
    - "Monitor system health"
    - "Communicate status"
    
  verification:
    - "System functionality restored"
    - "Data integrity verified"
    - "Security controls active"
    - "Performance acceptable"
```

---

## 📚 Knowledge Resources and References

### Essential Documentation
```yaml
essential_documentation:
  runbooks:
    - "GO_LIVE_RUNBOOK.md"
    - "DATA_ADAPTER_CUTOVER_PLAN.md"
    - "CHAOS_INCIDENT_DRILLS.md"
    - "PRODUCTION_RUNBOOKS_INDEX.md"
    
  playbooks:
    - "Business validation procedures"
    - "Data reconciliation processes"
    - "Capacity management procedures"
    - "Security incident response"
    
  technical_docs:
    - "System architecture"
    - "API documentation"
    - "Database schemas"
    - "Configuration guides"
```

### Training and Certification
```yaml
training_certification:
  required_training:
    - "BondX platform overview"
    - "Incident response procedures"
    - "Security best practices"
    - "Compliance requirements"
    
  recommended_certifications:
    - "Kubernetes administration"
    - "Monitoring and observability"
    - "Security operations"
    - "Database administration"
    
  ongoing_learning:
    - "Weekly knowledge sharing sessions"
    - "Monthly training updates"
    - "Quarterly skill assessments"
    - "Annual certification renewal"
```

---

## 📞 Contact Information and Escalation

### Primary Contacts
- **Operations Team Lead:** [Name] - [Email] - [Phone]
- **Platform Engineering Lead:** [Name] - [Email] - [Phone]
- **Security Team Lead:** [Name] - [Email] - [Phone]

### Escalation Matrix
```yaml
escalation_matrix:
  technical_issues:
    - "Level 1: On-call operator"
    - "Level 2: Team lead (15 min)"
    - "Level 3: Engineering manager (30 min)"
    - "Level 4: CTO (1 hour)"
    
  security_incidents:
    - "Level 1: Security team"
    - "Level 2: CISO (15 min)"
    - "Level 3: Legal team (30 min)"
    - "Level 4: Executive team (1 hour)"
    
  compliance_issues:
    - "Level 1: Compliance team"
    - "Level 2: Head of Compliance (15 min)"
    - "Level 3: Legal team (30 min)"
    - "Level 4: Executive team (1 hour)"
```

### Support Channels
- **Immediate Issues:** Slack #bondx-ops
- **Escalations:** Phone calls to primary contacts
- **Documentation:** Shared drive and wiki
- **Training:** Weekly knowledge sharing sessions

---

**Next Review:** Monthly or after major incidents  
**Owner:** Operations Team Lead

