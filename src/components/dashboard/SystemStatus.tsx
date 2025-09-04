import React from 'react'
import { ConnectionStatus } from '@/types'
import { 
  Wifi, 
  WifiOff, 
  Clock, 
  Activity, 
  CheckCircle, 
  AlertTriangle, 
  XCircle,
  Server,
  Database,
  Cpu
} from 'lucide-react'

interface SystemStatusProps {
  wsStatus: ConnectionStatus
}

const SystemStatus: React.FC<SystemStatusProps> = ({ wsStatus }) => {
  const getStatusIcon = (status: ConnectionStatus) => {
    switch (status) {
      case 'connected':
        return <Wifi className="h-4 w-4 text-accent-green" />
      case 'connecting':
        return <Clock className="h-4 w-4 text-accent-amber animate-spin" />
      case 'reconnecting':
        return <Activity className="h-4 w-4 text-accent-amber animate-pulse" />
      case 'disconnected':
        return <WifiOff className="h-4 w-4 text-accent-red" />
      case 'error':
        return <XCircle className="h-4 w-4 text-accent-red" />
      default:
        return <WifiOff className="h-4 w-4 text-accent-red" />
    }
  }

  const getStatusColor = (status: ConnectionStatus) => {
    switch (status) {
      case 'connected':
        return 'text-accent-green'
      case 'connecting':
      case 'reconnecting':
        return 'text-accent-amber'
      case 'disconnected':
      case 'error':
        return 'text-accent-red'
      default:
        return 'text-accent-red'
    }
  }

  const getStatusText = (status: ConnectionStatus) => {
    switch (status) {
      case 'connected':
        return 'Connected'
      case 'connecting':
        return 'Connecting'
      case 'reconnecting':
        return 'Reconnecting'
      case 'disconnected':
        return 'Disconnected'
      case 'error':
        return 'Error'
      default:
        return 'Unknown'
    }
  }

  // Mock system metrics
  const systemMetrics = [
    {
      name: 'API Response Time',
      value: '45ms',
      status: 'healthy',
      icon: Server
    },
    {
      name: 'Database Latency',
      value: '12ms',
      status: 'healthy',
      icon: Database
    },
    {
      name: 'CPU Usage',
      value: '23%',
      status: 'healthy',
      icon: Cpu
    }
  ]

  const getMetricStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="h-4 w-4 text-accent-green" />
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-accent-amber" />
      case 'critical':
        return <XCircle className="h-4 w-4 text-accent-red" />
      default:
        return <CheckCircle className="h-4 w-4 text-accent-green" />
    }
  }

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-text-primary">System Status</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* WebSocket Status */}
        <div className="p-4 bg-background-secondary border border-border rounded-lg">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-sm font-medium text-text-primary">WebSocket Connection</h3>
            {getStatusIcon(wsStatus)}
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm text-text-secondary">Status:</span>
              <span className={`text-sm font-medium ${getStatusColor(wsStatus)}`}>
                {getStatusText(wsStatus)}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-text-secondary">Latency:</span>
              <span className="text-sm font-mono text-text-primary">
                {wsStatus === 'connected' ? '<50ms' : '--'}
              </span>
            </div>
          </div>
        </div>

        {/* System Metrics */}
        {systemMetrics.map((metric, index) => {
          const Icon = metric.icon
          return (
            <div key={index} className="p-4 bg-background-secondary border border-border rounded-lg">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-text-primary">{metric.name}</h3>
                {getMetricStatusIcon(metric.status)}
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-secondary">Value:</span>
                  <span className="text-sm font-mono text-text-primary">{metric.value}</span>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-sm text-text-secondary">Status:</span>
                  <span className="text-sm font-medium text-accent-green capitalize">
                    {metric.status}
                  </span>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Overall System Health */}
      <div className="p-4 bg-background-secondary border border-border rounded-lg">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium text-text-primary">Overall System Health</h3>
          <div className="flex items-center space-x-2">
            <CheckCircle className="h-5 w-5 text-accent-green" />
            <span className="text-sm font-medium text-accent-green">All Systems Operational</span>
          </div>
        </div>
        
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-xs text-text-secondary mb-1">Uptime</p>
            <p className="text-lg font-semibold text-text-primary">99.98%</p>
          </div>
          <div>
            <p className="text-xs text-text-secondary mb-1">Last Incident</p>
            <p className="text-sm text-text-primary">7 days ago</p>
          </div>
          <div>
            <p className="text-xs text-text-secondary mb-1">Performance</p>
            <p className="text-lg font-semibold text-accent-green">Excellent</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SystemStatus
