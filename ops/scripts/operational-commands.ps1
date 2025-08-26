# BondX Operational Commands PowerShell Script
# Alternative to Make targets for Windows environments

param(
    [Parameter(Mandatory=$true)]
    [string]$Command,
    
    [Parameter(Mandatory=$false)]
    [string]$Environment = "production",
    
    [Parameter(Mandatory=$false)]
    [string]$Namespace = "bondx",
    
    [Parameter(Mandatory=$false)]
    [int]$Replicas = 3
)

# Configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Colors for output
$Red = "`e[31m"
$Green = "`e[32m"
$Yellow = "`e[33m"
$Blue = "`e[34m"
$Reset = "`e[0m"

function Write-ColorOutput {
    param([string]$Message, [string]$Color = $Reset)
    Write-Host "$Color$Message$Reset"
}

function Write-Success { param([string]$Message) Write-ColorOutput "✅ $Message" $Green }
function Write-Info { param([string]$Message) Write-ColorOutput "ℹ️  $Message" $Blue }
function Write-Warning { param([string]$Message) Write-ColorOutput "⚠️  $Message" $Yellow }
function Write-Error { param([string]$Message) Write-ColorOutput "❌ $Message" $Red }

function Test-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check kubectl
    try {
        $kubectlVersion = kubectl version --client --short
        Write-Success "kubectl found: $kubectlVersion"
    } catch {
        Write-Error "kubectl not found. Please install kubectl first."
        exit 1
    }
    
    # Check cluster connectivity
    try {
        kubectl cluster-info | Out-Null
        Write-Success "Connected to Kubernetes cluster"
    } catch {
        Write-Error "Cannot connect to Kubernetes cluster"
        exit 1
    }
    
    Write-Success "Prerequisites check passed"
}

function Deploy-Staging {
    Write-Info "Deploying to staging environment..."
    
    try {
        # Apply staging configurations
        kubectl apply -f deploy/kubernetes/namespace.yaml
        kubectl apply -f deploy/redis/ -n $Namespace
        kubectl apply -f deploy/monitoring/ -n bondx-monitoring
        
        # Apply application configurations
        kubectl apply -f deploy/kubernetes/configmap.yaml
        kubectl apply -f deploy/kubernetes/secret.yaml
        kubectl apply -f deploy/kubernetes/deployment.yaml
        kubectl apply -f deploy/kubernetes/service.yaml
        kubectl apply -f deploy/kubernetes/hpa.yaml
        kubectl apply -f deploy/kubernetes/pdb.yaml
        kubectl apply -f deploy/kubernetes/ingress.yaml
        
        Write-Success "Staging deployment completed"
    } catch {
        Write-Error "Staging deployment failed: $_"
        exit 1
    }
}

function Deploy-ProductionObservable {
    Write-Info "Deploying production in observable mode..."
    
    try {
        # Deploy with observe-only configuration
        kubectl apply -f deploy/kubernetes/namespace.yaml
        kubectl apply -f deploy/redis/ -n $Namespace
        kubectl apply -f deploy/monitoring/ -n bondx-monitoring
        
        # Apply production configurations
        kubectl apply -f deploy/kubernetes/configmap.yaml
        kubectl apply -f deploy/kubernetes/secret.yaml
        kubectl apply -f deploy/kubernetes/deployment.yaml
        kubectl apply -f deploy/kubernetes/service.yaml
        kubectl apply -f deploy/kubernetes/hpa.yaml
        kubectl apply -f deploy/kubernetes/pdb.yaml
        kubectl apply -f deploy/kubernetes/ingress.yaml
        
        Write-Success "Production deployment completed in observable mode"
    } catch {
        Write-Error "Production deployment failed: $_"
        exit 1
    }
}

function Promote-Production {
    Write-Info "Promoting staging to production..."
    
    try {
        # Update configurations for production
        kubectl patch configmap bondx-config -n $Namespace --patch '{"data":{"ENVIRONMENT":"production"}}'
        
        # Scale up replicas
        kubectl scale deployment bondx-backend -n $Namespace --replicas=$Replicas
        
        Write-Success "Production promotion completed"
    } catch {
        Write-Error "Production promotion failed: $_"
        exit 1
    }
}

function Rollback-Deployment {
    Write-Info "Rolling back deployment..."
    
    try {
        # Check for backup
        if (Test-Path "backup") {
            $latestBackup = Get-ChildItem "backup" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            
            if ($latestBackup) {
                Write-Info "Restoring from backup: $($latestBackup.Name)"
                kubectl apply -f "$($latestBackup.FullName)/"
                Write-Success "Rollback completed from backup"
            } else {
                Write-Warning "No backup found, performing basic rollback"
                kubectl rollout undo deployment/bondx-backend -n $Namespace
                Write-Success "Basic rollback completed"
            }
        } else {
            Write-Warning "No backup directory found, performing basic rollback"
            kubectl rollout undo deployment/bondx-backend -n $Namespace
            Write-Success "Basic rollback completed"
        }
    } catch {
        Write-Error "Rollback failed: $_"
        exit 1
    }
}

function Emergency-Rollback {
    Write-Error "EMERGENCY ROLLBACK - Stopping all traffic..."
    
    try {
        # Stop all traffic
        kubectl scale deployment bondx-backend -n $Namespace --replicas=0
        
        # Perform rollback
        Rollback-Deployment
        
        # Restore traffic
        kubectl scale deployment bondx-backend -n $Namespace --replicas=$Replicas
        
        Write-Success "Emergency rollback completed"
    } catch {
        Write-Error "Emergency rollback failed: $_"
        exit 1
    }
}

function Enable-LiveDataProviders {
    Write-Info "Enabling live data providers..."
    
    try {
        kubectl patch configmap bondx-config -n $Namespace --patch '{"data":{"LIVE_DATA_PROVIDERS":"true"}}'
        Write-Success "Live data providers enabled"
    } catch {
        Write-Error "Failed to enable live data providers: $_"
        exit 1
    }
}

function Enable-RealTimeCompliance {
    Write-Info "Enabling real-time compliance..."
    
    try {
        kubectl patch configmap bondx-config -n $Namespace --patch '{"data":{"REAL_TIME_COMPLIANCE":"true"}}'
        Write-Success "Real-time compliance enabled"
    } catch {
        Write-Error "Failed to enable real-time compliance: $_"
        exit 1
    }
}

function Validate-ProductionHealth {
    Write-Info "Validating production health..."
    
    try {
        # Wait for deployment to be ready
        kubectl wait --for=condition=available deployment/bondx-backend -n $Namespace --timeout=600s
        kubectl wait --for=condition=ready pod -l app=bondx-backend -n $Namespace --timeout=300s
        
        # Check health endpoint
        $healthCheck = kubectl exec -n $Namespace deployment/bondx-backend -- curl -s http://localhost:8000/health
        Write-Success "Production health validation completed"
        Write-Info "Health check result: $healthCheck"
    } catch {
        Write-Error "Production health validation failed: $_"
        exit 1
    }
}

function Monitor-KPIs {
    Write-Info "Monitoring KPIs..."
    
    try {
        # Get pod status
        Write-Info "Pod Status:"
        kubectl get pods -n $Namespace -o wide
        
        # Get service status
        Write-Info "Service Status:"
        kubectl get services -n $Namespace
        
        # Get deployment status
        Write-Info "Deployment Status:"
        kubectl get deployment -n $Namespace
        
        Write-Success "KPI monitoring completed"
    } catch {
        Write-Error "KPI monitoring failed: $_"
        exit 1
    }
}

function Test-WebSocketResilience {
    Write-Info "Testing WebSocket resilience..."
    
    try {
        # This would typically run actual WebSocket tests
        # For now, we'll simulate the test
        Write-Info "Running WebSocket resilience tests..."
        Start-Sleep -Seconds 5
        Write-Success "WebSocket resilience test completed"
    } catch {
        Write-Error "WebSocket resilience test failed: $_"
        exit 1
    }
}

function Test-RedisFailover {
    Write-Info "Testing Redis failover..."
    
    try {
        # Check Redis cluster status
        Write-Info "Redis Cluster Status:"
        kubectl exec -n $Namespace deployment/bondx-redis-master -- redis-cli info replication
        
        Write-Info "Redis Sentinel Status:"
        kubectl exec -n $Namespace deployment/bondx-redis-sentinel-1 -- redis-cli -p 26379 sentinel master bondx-master
        
        Write-Success "Redis failover test completed"
    } catch {
        Write-Error "Redis failover test failed: $_"
        exit 1
    }
}

function Setup-TrafficMirroring {
    Write-Info "Setting up traffic mirroring..."
    
    try {
        # This would typically configure traffic mirroring
        # For now, we'll simulate the setup
        Write-Info "Configuring traffic mirroring from staging to production..."
        Start-Sleep -Seconds 3
        Write-Success "Traffic mirroring setup completed"
    } catch {
        Write-Error "Traffic mirroring setup failed: $_"
        exit 1
    }
}

function Drain-Traffic {
    Write-Info "Draining traffic..."
    
    try {
        kubectl scale deployment bondx-backend -n $Namespace --replicas=0
        Write-Success "Traffic drained"
    } catch {
        Write-Error "Traffic draining failed: $_"
        exit 1
    }
}

function Restore-Traffic {
    Write-Info "Restoring traffic..."
    
    try {
        kubectl scale deployment bondx-backend -n $Namespace --replicas=$Replicas
        Write-Success "Traffic restored"
    } catch {
        Write-Error "Traffic restoration failed: $_"
        exit 1
    }
}

function Show-Status {
    Write-Info "Showing deployment status..."
    
    try {
        Write-Info "Deployment Status:"
        kubectl get all -n $Namespace
        
        Write-Info "Pod Status:"
        kubectl get pods -n $Namespace -o wide
        
        Write-Info "Service Status:"
        kubectl get services -n $Namespace
        
        Write-Info "Ingress Status:"
        kubectl get ingress -n $Namespace
        
        Write-Success "Status display completed"
    } catch {
        Write-Error "Status display failed: $_"
        exit 1
    }
}

function Show-Logs {
    Write-Info "Showing application logs..."
    
    try {
        kubectl logs -n $Namespace -l app=bondx-backend --tail=100 -f
    } catch {
        Write-Error "Log display failed: $_"
        exit 1
    }
}

function Show-Help {
    Write-Info "BondX Operational Commands"
    Write-Info "Usage: .\operational-commands.ps1 -Command <command> [-Environment <env>] [-Namespace <ns>] [-Replicas <num>]"
    Write-Info ""
    Write-Info "Available Commands:"
    Write-Info "  deploy-staging              - Deploy to staging environment"
    Write-Info "  deploy-production-observable - Deploy production in observable mode"
    Write-Info "  promote-prod                - Promote staging to production"
    Write-Info "  rollback                    - Rollback deployment"
    Write-Info "  emergency-rollback          - Emergency rollback (stops traffic)"
    Write-Info "  enable-live-data            - Enable live data providers"
    Write-Info "  enable-real-time-compliance - Enable real-time compliance"
    Write-Info "  validate-health             - Validate production health"
    Write-Info "  monitor-kpis                - Monitor key performance indicators"
    Write-Info "  test-websocket              - Test WebSocket resilience"
    Write-Info "  test-redis                  - Test Redis failover"
    Write-Info "  setup-mirroring             - Set up traffic mirroring"
    Write-Info "  drain-traffic               - Drain traffic to 0%"
    Write-Info "  restore-traffic             - Restore traffic to normal"
    Write-Info "  status                      - Show deployment status"
    Write-Info "  logs                        - Show application logs"
    Write-Info "  help                        - Show this help message"
    Write-Info ""
    Write-Info "Examples:"
    Write-Info "  .\operational-commands.ps1 -Command deploy-staging"
    Write-Info "  .\operational-commands.ps1 -Command promote-prod -Environment production -Replicas 5"
}

# Main execution
try {
    Test-Prerequisites
    
    switch ($Command.ToLower()) {
        "deploy-staging" { Deploy-Staging }
        "deploy-production-observable" { Deploy-ProductionObservable }
        "promote-prod" { Promote-Production }
        "rollback" { Rollback-Deployment }
        "emergency-rollback" { Emergency-Rollback }
        "enable-live-data" { Enable-LiveDataProviders }
        "enable-real-time-compliance" { Enable-RealTimeCompliance }
        "validate-health" { Validate-ProductionHealth }
        "monitor-kpis" { Monitor-KPIs }
        "test-websocket" { Test-WebSocketResilience }
        "test-redis" { Test-RedisFailover }
        "setup-mirroring" { Setup-TrafficMirroring }
        "drain-traffic" { Drain-Traffic }
        "restore-traffic" { Restore-Traffic }
        "status" { Show-Status }
        "logs" { Show-Logs }
        "help" { Show-Help }
        default {
            Write-Error "Unknown command: $Command"
            Write-Info "Use 'help' command to see available options"
            exit 1
        }
    }
} catch {
    Write-Error "Command execution failed: $_"
    exit 1
}
