import React, { useState, useEffect } from 'react'
import { X, Bell, AlertTriangle, Info, CheckCircle, XCircle } from 'lucide-react'
import { Notification, NotificationType, NotificationSeverity } from '@/types'

interface NotificationCenterProps {
  onClose: () => void
}

const NotificationCenter: React.FC<NotificationCenterProps> = ({ onClose }) => {
  const [notifications, setNotifications] = useState<Notification[]>([])
  const [activeTab, setActiveTab] = useState<'all' | 'alerts' | 'trades' | 'system'>('all')

  // Mock notifications for demonstration
  useEffect(() => {
    const mockNotifications: Notification[] = [
      {
        id: '1',
        userId: 'user1',
        type: 'trade_execution',
        title: 'Trade Executed',
        message: 'Your order for 10Y G-Sec has been executed at 98.45',
        severity: 'success',
        isRead: false,
        createdAt: new Date(Date.now() - 5 * 60 * 1000), // 5 minutes ago
        actionUrl: '/trades/123'
      },
      {
        id: '2',
        userId: 'user1',
        type: 'risk_alert',
        title: 'Risk Limit Warning',
        message: 'Portfolio VaR is approaching the 95% confidence limit',
        severity: 'warning',
        isRead: false,
        createdAt: new Date(Date.now() - 15 * 60 * 1000), // 15 minutes ago
        actionUrl: '/risk/dashboard'
      },
      {
        id: '3',
        userId: 'user1',
        type: 'price_alert',
        title: 'Price Alert',
        message: '5Y G-Sec price has moved by 0.15% in the last hour',
        severity: 'info',
        isRead: true,
        createdAt: new Date(Date.now() - 30 * 60 * 1000), // 30 minutes ago
        actionUrl: '/market/5y-gsec'
      },
      {
        id: '4',
        userId: 'user1',
        type: 'system_alert',
        title: 'System Maintenance',
        message: 'Scheduled maintenance will begin at 2:00 AM UTC',
        severity: 'info',
        isRead: false,
        createdAt: new Date(Date.now() - 60 * 60 * 1000), // 1 hour ago
        actionUrl: '/system/status'
      }
    ]

    setNotifications(mockNotifications)
  }, [])

  const getSeverityIcon = (severity: NotificationSeverity) => {
    switch (severity) {
      case 'success':
        return <CheckCircle className="h-4 w-4 text-accent-green" />
      case 'warning':
        return <AlertTriangle className="h-4 w-4 text-accent-amber" />
      case 'error':
        return <XCircle className="h-4 w-4 text-accent-red" />
      case 'critical':
        return <XCircle className="h-4 w-4 text-accent-red" />
      case 'info':
      default:
        return <Info className="h-4 w-4 text-accent-blue" />
    }
  }

  const getSeverityColor = (severity: NotificationSeverity) => {
    switch (severity) {
      case 'success':
        return 'border-accent-green/20 bg-accent-green/5'
      case 'warning':
        return 'border-accent-amber/20 bg-accent-amber/5'
      case 'error':
      case 'critical':
        return 'border-accent-red/20 bg-accent-red/5'
      case 'info':
      default:
        return 'border-accent-blue/20 bg-accent-blue/5'
    }
  }

  const getTypeLabel = (type: NotificationType) => {
    switch (type) {
      case 'trade_execution':
        return 'Trade'
      case 'risk_alert':
        return 'Risk'
      case 'price_alert':
        return 'Price'
      case 'system_alert':
        return 'System'
      case 'news':
        return 'News'
      default:
        return 'Alert'
    }
  }

  const formatTimeAgo = (date: Date) => {
    const now = new Date()
    const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60))
    
    if (diffInMinutes < 1) return 'Just now'
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`
    
    const diffInHours = Math.floor(diffInMinutes / 60)
    if (diffInHours < 24) return `${diffInHours}h ago`
    
    const diffInDays = Math.floor(diffInHours / 24)
    return `${diffInDays}d ago`
  }

  const filteredNotifications = notifications.filter(notification => {
    if (activeTab === 'all') return true
    if (activeTab === 'alerts') return ['risk_alert', 'system_alert'].includes(notification.type)
    if (activeTab === 'trades') return notification.type === 'trade_execution'
    if (activeTab === 'system') return notification.type === 'system_alert'
    return true
  })

  const unreadCount = notifications.filter(n => !n.isRead).length

  const markAsRead = (notificationId: string) => {
    setNotifications(prev => 
      prev.map(n => 
        n.id === notificationId ? { ...n, isRead: true } : n
      )
    )
  }

  const markAllAsRead = () => {
    setNotifications(prev => 
      prev.map(n => ({ ...n, isRead: true }))
    )
  }

  const clearNotification = (notificationId: string) => {
    setNotifications(prev => prev.filter(n => n.id !== notificationId))
  }

  return (
    <div className="fixed right-4 top-20 w-96 max-h-[calc(100vh-6rem)] bg-background-secondary border border-border rounded-lg shadow-2xl z-50">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-border">
        <div className="flex items-center space-x-2">
          <Bell className="h-5 w-5 text-text-primary" />
          <h3 className="text-lg font-semibold text-text-primary">Notifications</h3>
          {unreadCount > 0 && (
            <span className="badge badge-info">{unreadCount}</span>
          )}
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={markAllAsRead}
            className="text-sm text-accent-blue hover:text-accent-blue/80 transition-colors duration-200"
          >
            Mark all read
          </button>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-background-tertiary transition-colors duration-200"
          >
            <X className="h-4 w-4 text-text-secondary" />
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-border">
        {[
          { key: 'all', label: 'All', count: notifications.length },
          { key: 'alerts', label: 'Alerts', count: notifications.filter(n => ['risk_alert', 'system_alert'].includes(n.type)).length },
          { key: 'trades', label: 'Trades', count: notifications.filter(n => n.type === 'trade_execution').length },
          { key: 'system', label: 'System', count: notifications.filter(n => n.type === 'system_alert').length }
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as any)}
            className={`flex-1 px-3 py-2 text-sm font-medium transition-colors duration-200 ${
              activeTab === tab.key
                ? 'text-accent-blue border-b-2 border-accent-blue'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            {tab.label}
            <span className="ml-1 text-xs text-text-muted">({tab.count})</span>
          </button>
        ))}
      </div>

      {/* Notifications List */}
      <div className="max-h-96 overflow-y-auto">
        {filteredNotifications.length === 0 ? (
          <div className="p-6 text-center">
            <Bell className="h-12 w-12 text-text-muted mx-auto mb-3" />
            <p className="text-text-secondary">No notifications</p>
          </div>
        ) : (
          <div className="divide-y divide-border">
            {filteredNotifications.map((notification) => (
              <div
                key={notification.id}
                className={`p-4 transition-all duration-200 ${
                  notification.isRead ? 'opacity-75' : ''
                }`}
              >
                <div className={`border-l-4 pl-3 ${getSeverityColor(notification.severity)}`}>
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      {getSeverityIcon(notification.severity)}
                      <span className="text-xs font-medium text-text-muted uppercase tracking-wide">
                        {getTypeLabel(notification.type)}
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-1">
                      <button
                        onClick={() => markAsRead(notification.id)}
                        className="text-xs text-text-secondary hover:text-text-primary transition-colors duration-200"
                      >
                        {notification.isRead ? 'Mark unread' : 'Mark read'}
                      </button>
                      <button
                        onClick={() => clearNotification(notification.id)}
                        className="text-xs text-text-secondary hover:text-accent-red transition-colors duration-200"
                      >
                        Clear
                      </button>
                    </div>
                  </div>
                  
                  <h4 className="font-medium text-text-primary mb-1">
                    {notification.title}
                  </h4>
                  
                  <p className="text-sm text-text-secondary mb-2">
                    {notification.message}
                  </p>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-text-muted">
                      {formatTimeAgo(notification.createdAt)}
                    </span>
                    
                    {notification.actionUrl && (
                      <button
                        onClick={() => {
                          // TODO: Navigate to action URL
                          console.log('Navigate to:', notification.actionUrl)
                        }}
                        className="text-xs text-accent-blue hover:text-accent-blue/80 transition-colors duration-200"
                      >
                        View Details
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-border bg-background-tertiary">
        <div className="flex items-center justify-between text-xs text-text-secondary">
          <span>Showing {filteredNotifications.length} of {notifications.length} notifications</span>
          <button
            onClick={() => setNotifications([])}
            className="text-accent-red hover:text-accent-red/80 transition-colors duration-200"
          >
            Clear All
          </button>
        </div>
      </div>
    </div>
  )
}

export default NotificationCenter
