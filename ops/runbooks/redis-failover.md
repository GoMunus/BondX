# Redis Failover Runbook

## Overview
This runbook covers the procedures for handling Redis failover scenarios in the BondX production environment.

## Prerequisites
- Access to Redis Sentinel instances
- Access to Redis master/replica instances
- Monitoring access (Prometheus/Grafana)
- Communication channels for incident response

## Failover Detection

### Automatic Detection
Redis Sentinel automatically detects failures and initiates failover:
1. Monitor Sentinel logs for failover events
2. Check Prometheus alerts for Redis-related issues
3. Verify application connectivity

### Manual Detection
If automatic detection fails:
1. Check Redis master status: `redis-cli -h <master-host> -p 6379 ping`
2. Check Sentinel status: `redis-cli -h <sentinel-host> -p 26379 sentinel master bondx-master`
3. Check application error logs

## Failover Response

### Immediate Actions (0-5 minutes)
1. **Acknowledge the incident**
   - Update incident tracking system
   - Notify on-call team
   - Begin incident response procedures

2. **Assess impact**
   - Check which services are affected
   - Monitor error rates and latency
   - Identify affected users/transactions

3. **Verify failover status**
   - Check if Sentinel has promoted a new master
   - Verify new master is accepting writes
   - Confirm replicas are syncing with new master

### Short-term Actions (5-30 minutes)
1. **Monitor failover progress**
   - Watch Redis replication lag
   - Monitor application reconnection attempts
   - Check for data consistency issues

2. **Update application configuration**
   - Update Redis connection strings if needed
   - Restart affected services if necessary
   - Verify application connectivity

3. **Communicate status**
   - Update stakeholders on progress
   - Provide estimated recovery time
   - Document any workarounds

### Medium-term Actions (30 minutes - 2 hours)
1. **Investigate root cause**
   - Analyze Redis logs for failure reason
   - Check system resources (CPU, memory, disk)
   - Review network connectivity

2. **Stabilize the system**
   - Ensure all replicas are properly syncing
   - Monitor Redis performance metrics
   - Verify application stability

3. **Plan recovery**
   - Determine if manual intervention is needed
   - Plan for potential rollback
   - Prepare post-incident analysis

## Recovery Procedures

### Automatic Recovery
If Sentinel successfully promotes a replica:
1. Monitor the new master's performance
2. Verify all replicas are syncing
3. Check application connectivity
4. Monitor for any data inconsistencies

### Manual Recovery
If automatic failover fails:
1. **Stop the failed master**
   ```bash
   redis-cli -h <failed-master> -p 6379 shutdown
   ```

2. **Promote a replica manually**
   ```bash
   redis-cli -h <replica-host> -p 6379
   > SLAVEOF NO ONE
   ```

3. **Update Sentinel configuration**
   ```bash
   redis-cli -h <sentinel-host> -p 26379
   > SENTINEL SET bondx-master down-after-milliseconds 5000
   > SENTINEL SET bondx-master failover-timeout 10000
   ```

4. **Reconfigure other replicas**
   ```bash
   redis-cli -h <other-replica> -p 6379
   > SLAVEOF <new-master-host> 6379
   ```

### Data Consistency Verification
1. **Check replication lag**
   ```bash
   redis-cli -h <replica> -p 6379 info replication
   ```

2. **Verify data integrity**
   - Compare key counts between master and replicas
   - Check for any missing or corrupted data
   - Verify critical application data

3. **Monitor for data drift**
   - Set up alerts for replication lag
   - Monitor for data inconsistencies
   - Implement data validation checks

## Post-Failover Actions

### System Verification
1. **Performance monitoring**
   - Monitor Redis response times
   - Check memory usage and eviction rates
   - Verify connection pool utilization

2. **Application testing**
   - Test critical user flows
   - Verify WebSocket connections
   - Check trading functionality

3. **Data validation**
   - Verify order book integrity
   - Check trade data consistency
   - Validate risk calculations

### Documentation
1. **Incident report**
   - Document timeline of events
   - Record actions taken
   - Note any issues encountered

2. **Lessons learned**
   - Identify improvement opportunities
   - Update runbooks and procedures
   - Plan for future prevention

3. **Monitoring improvements**
   - Add additional alerts if needed
   - Improve failover detection
   - Enhance recovery procedures

## Prevention Measures

### Proactive Monitoring
1. **Health checks**
   - Regular Redis health checks
   - Monitor system resources
   - Check network connectivity

2. **Capacity planning**
   - Monitor memory usage trends
   - Plan for growth
   - Regular performance testing

3. **Backup and recovery**
   - Regular Redis snapshots
   - Test recovery procedures
   - Document backup procedures

### Regular Maintenance
1. **System updates**
   - Keep Redis versions current
   - Apply security patches
   - Update monitoring tools

2. **Testing**
   - Regular failover testing
   - Load testing scenarios
   - Disaster recovery drills

## Emergency Contacts

### Primary Contacts
- **On-call Engineer**: [Contact Information]
- **Database Administrator**: [Contact Information]
- **System Administrator**: [Contact Information]

### Escalation Contacts
- **Team Lead**: [Contact Information]
- **Engineering Manager**: [Contact Information]
- **CTO**: [Contact Information]

### External Contacts
- **Cloud Provider Support**: [Contact Information]
- **Redis Support**: [Contact Information]

## Related Documentation
- [Redis Configuration Guide](../deploy/redis/)
- [Monitoring Setup](../deploy/monitoring/)
- [Incident Response Plan](../incident-response.md)
- [System Architecture](../architecture.md)
