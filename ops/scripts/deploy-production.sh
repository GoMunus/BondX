#!/bin/bash

# BondX Production Deployment Script
# This script deploys the BondX application to production

set -euo pipefail

# Configuration
NAMESPACE="bondx"
ENVIRONMENT="production"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
DRY_RUN="${DRY_RUN:-false}"
ROLLBACK_ON_FAILURE="${ROLLBACK_ON_FAILURE:-true}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check if namespace exists
    if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_error "Namespace $NAMESPACE does not exist"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check current deployment status
    if kubectl get deployment -n "$NAMESPACE" bondx-backend &> /dev/null; then
        CURRENT_REPLICAS=$(kubectl get deployment -n "$NAMESPACE" bondx-backend -o jsonpath='{.status.replicas}')
        AVAILABLE_REPLICAS=$(kubectl get deployment -n "$NAMESPACE" bondx-backend -o jsonpath='{.status.availableReplicas}')
        
        if [ "$AVAILABLE_REPLICAS" -lt "$CURRENT_REPLICAS" ]; then
            log_warning "Current deployment has $AVAILABLE_REPLICAS/$CURRENT_REPLICAS pods available"
        fi
    fi
    
    # Check resource availability
    NODE_CPU=$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.cpu}')
    NODE_MEMORY=$(kubectl get nodes -o jsonpath='{.items[0].status.allocatable.memory}')
    
    log_info "Node resources: CPU=$NODE_CPU, Memory=$NODE_MEMORY"
    
    # Check storage availability
    if ! kubectl get pvc -n "$NAMESPACE" &> /dev/null; then
        log_warning "No persistent volume claims found"
    fi
    
    log_success "Pre-deployment checks completed"
}

# Backup current deployment
backup_deployment() {
    log_info "Creating backup of current deployment..."
    
    BACKUP_DIR="backup/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"
    
    # Backup current deployment
    kubectl get deployment -n "$NAMESPACE" bondx-backend -o yaml > "$BACKUP_DIR/deployment.yaml" 2>/dev/null || true
    
    # Backup current service
    kubectl get service -n "$NAMESPACE" bondx-backend-service -o yaml > "$BACKUP_DIR/service.yaml" 2>/dev/null || true
    
    # Backup current configmap
    kubectl get configmap -n "$NAMESPACE" bondx-config -o yaml > "$BACKUP_DIR/configmap.yaml" 2>/dev/null || true
    
    # Backup current secret
    kubectl get secret -n "$NAMESPACE" bondx-secrets -o yaml > "$BACKUP_DIR/secret.yaml" 2>/dev/null || true
    
    log_success "Backup created in $BACKUP_DIR"
}

# Deploy infrastructure components
deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Deploy Redis HA
    if [ "$DRY_RUN" = "false" ]; then
        kubectl apply -f deploy/redis/ -n "$NAMESPACE"
        log_info "Waiting for Redis to be ready..."
        kubectl wait --for=condition=ready pod -l app=redis -n "$NAMESPACE" --timeout=300s
    else
        log_info "DRY RUN: Would deploy Redis HA"
    fi
    
    # Deploy monitoring
    if [ "$DRY_RUN" = "false" ]; then
        kubectl apply -f deploy/monitoring/ -n bondx-monitoring
        log_info "Waiting for monitoring to be ready..."
        kubectl wait --for=condition=ready pod -l app=prometheus -n bondx-monitoring --timeout=300s
    else
        log_info "DRY RUN: Would deploy monitoring"
    fi
    
    log_success "Infrastructure deployment completed"
}

# Deploy application
deploy_application() {
    log_info "Deploying BondX application..."
    
    # Deploy ConfigMap
    if [ "$DRY_RUN" = "false" ]; then
        kubectl apply -f deploy/kubernetes/configmap.yaml
        log_success "ConfigMap deployed"
    else
        log_info "DRY RUN: Would deploy ConfigMap"
    fi
    
    # Deploy Secret
    if [ "$DRY_RUN" = "false" ]; then
        kubectl apply -f deploy/kubernetes/secret.yaml
        log_success "Secret deployed"
    else
        log_info "DRY RUN: Would deploy Secret"
    fi
    
    # Deploy Deployment
    if [ "$DRY_RUN" = "false" ]; then
        kubectl apply -f deploy/kubernetes/deployment.yaml
        log_success "Deployment deployed"
    else
        log_info "DRY RUN: Would deploy Deployment"
    fi
    
    # Deploy Service
    if [ "$DRY_RUN" = "false" ]; then
        kubectl apply -f deploy/kubernetes/service.yaml
        log_success "Service deployed"
    else
        log_info "DRY RUN: Would deploy Service"
    fi
    
    # Deploy HPA
    if [ "$DRY_RUN" = "false" ]; then
        kubectl apply -f deploy/kubernetes/hpa.yaml
        log_success "HPA deployed"
    else
        log_info "DRY RUN: Would deploy HPA"
    fi
    
    # Deploy PDB
    if [ "$DRY_RUN" = "false" ]; then
        kubectl apply -f deploy/kubernetes/pdb.yaml
        log_success "PDB deployed"
    else
        log_info "DRY RUN: Would deploy PDB"
    fi
    
    # Deploy Ingress
    if [ "$DRY_RUN" = "false" ]; then
        kubectl apply -f deploy/kubernetes/ingress.yaml
        log_success "Ingress deployed"
    else
        log_info "DRY RUN: Would deploy Ingress"
    fi
    
    log_success "Application deployment completed"
}

# Wait for deployment to be ready
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    if [ "$DRY_RUN" = "false" ]; then
        # Wait for deployment to be available
        kubectl wait --for=condition=available deployment/bondx-backend -n "$NAMESPACE" --timeout=600s
        
        # Wait for all pods to be ready
        kubectl wait --for=condition=ready pod -l app=bondx-backend -n "$NAMESPACE" --timeout=300s
        
        # Wait for service to have endpoints
        kubectl wait --for=condition=ready endpoints/bondx-backend-service -n "$NAMESPACE" --timeout=300s
        
        log_success "Deployment is ready"
    else
        log_info "DRY RUN: Would wait for deployment"
    fi
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    if [ "$DRY_RUN" = "false" ]; then
        # Get service URL
        SERVICE_URL=$(kubectl get service -n "$NAMESPACE" bondx-backend-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        
        if [ -n "$SERVICE_URL" ]; then
            # Health check
            if curl -f "http://$SERVICE_URL/health" &> /dev/null; then
                log_success "Health check passed"
            else
                log_error "Health check failed"
                return 1
            fi
            
            # API check
            if curl -f "http://$SERVICE_URL/api/v1/health" &> /dev/null; then
                log_success "API health check passed"
            else
                log_error "API health check failed"
                return 1
            fi
        else
            log_warning "Service URL not available yet"
        fi
    else
        log_info "DRY RUN: Would run health checks"
    fi
    
    log_success "Health checks completed"
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    if [ "$DRY_RUN" = "false" ]; then
        # Basic functionality tests
        log_info "Testing basic functionality..."
        
        # Test database connectivity
        if kubectl exec -n "$NAMESPACE" deployment/bondx-backend -- python -c "import asyncio; from bondx.database.base import init_db; asyncio.run(init_db())" &> /dev/null; then
            log_success "Database connectivity test passed"
        else
            log_error "Database connectivity test failed"
            return 1
        fi
        
        # Test Redis connectivity
        if kubectl exec -n "$NAMESPACE" deployment/bondx-backend -- python -c "import redis; r = redis.Redis(host='bondx-redis-master', port=6379); r.ping()" &> /dev/null; then
            log_success "Redis connectivity test passed"
        else
            log_error "Redis connectivity test failed"
            return 1
        fi
        
        log_success "Smoke tests completed"
    else
        log_info "DRY RUN: Would run smoke tests"
    fi
}

# Rollback function
rollback() {
    log_warning "Rolling back deployment..."
    
    if [ -d "backup" ]; then
        LATEST_BACKUP=$(ls -t backup/ | head -1)
        if [ -n "$LATEST_BACKUP" ]; then
            log_info "Rolling back to $LATEST_BACKUP"
            
            if [ "$DRY_RUN" = "false" ]; then
                kubectl apply -f "backup/$LATEST_BACKUP/deployment.yaml"
                kubectl apply -f "backup/$LATEST_BACKUP/service.yaml"
                kubectl apply -f "backup/$LATEST_BACKUP/configmap.yaml"
                kubectl apply -f "backup/$LATEST_BACKUP/secret.yaml"
                
                log_success "Rollback completed"
            else
                log_info "DRY RUN: Would rollback to $LATEST_BACKUP"
            fi
        fi
    else
        log_error "No backup found for rollback"
    fi
}

# Main deployment function
main() {
    log_info "Starting BondX production deployment"
    log_info "Environment: $ENVIRONMENT"
    log_info "Namespace: $NAMESPACE"
    log_info "Dry run: $DRY_RUN"
    
    # Set error handling
    trap 'log_error "Deployment failed. Rolling back..."; rollback; exit 1' ERR
    
    # Run deployment steps
    check_prerequisites
    pre_deployment_checks
    backup_deployment
    deploy_infrastructure
    deploy_application
    wait_for_deployment
    run_health_checks
    run_smoke_tests
    
    log_success "BondX production deployment completed successfully!"
    
    # Clean up old backups (keep last 5)
    if [ "$DRY_RUN" = "false" ] && [ -d "backup" ]; then
        cd backup && ls -t | tail -n +6 | xargs -r rm -rf && cd ..
        log_info "Cleaned up old backups"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE="false"
            shift
            ;;
        --namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dry-run        Run deployment in dry-run mode"
            echo "  --no-rollback    Disable automatic rollback on failure"
            echo "  --namespace      Specify namespace (default: bondx)"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"
