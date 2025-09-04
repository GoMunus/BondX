// User types
export interface User {
  id: string
  email: string
  firstName: string
  lastName: string
  role: UserRole
  organization: string
  avatar?: string | null
  permissions?: Permission[]
  lastLogin?: Date
  isActive?: boolean
  mfaEnabled?: boolean
  preferences?: UserPreferences
}

export type UserRole = 'admin' | 'trader' | 'analyst' | 'viewer'

export interface Permission {
  resource: string
  actions: string[]
}

export interface UserPreferences {
  theme: 'light' | 'dark'
  language: string
  timezone: string
  notifications: {
    email: boolean
    push: boolean
    sms: boolean
  }
  dashboard: {
    layout: string
    widgets: string[]
  }
}

// Dashboard types
export interface Widget {
  id: string
  type: WidgetType
  title: string
  size: WidgetSize
  position: WidgetPosition
  config: Record<string, any>
  isVisible: boolean
}

export type WidgetType = 
  | 'portfolio-summary'
  | 'market-overview'
  | 'risk-metrics'
  | 'alerts'
  | 'news-feed'
  | 'liquidity-pulse'
  | 'economic-calendar'
  | 'currency-tracker'

export interface WidgetSize {
  width: number
  height: number
  minWidth?: number
  minHeight?: number
  maxWidth?: number
  maxHeight?: number
}

export interface WidgetPosition {
  x: number
  y: number
  w: number
  h: number
}

export interface DashboardLayout {
  id: string
  name: string
  widgets: Widget[]
  layout: WidgetPosition[]
  isDefault: boolean
  userId: string
}

// Notification types
export interface NotificationSettings {
  email: boolean
  push: boolean
  sms: boolean
  types: {
    trades: boolean
    alerts: boolean
    news: boolean
    system: boolean
  }
}

// API Response types
export interface ApiResponse<T = any> {
  data: T
  message?: string
  status: 'success' | 'error'
  timestamp: string
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number
    limit: number
    total: number
    totalPages: number
  }
}