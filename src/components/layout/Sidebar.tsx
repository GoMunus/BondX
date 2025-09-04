import React, { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { 
  Home, 
  BarChart3, 
  TrendingUp, 
  Shield, 
  Brain, 
  Settings, 
  Sun,
  Moon,
  Monitor,
  ChevronLeft,
  ChevronRight,
  Building2,
  DollarSign,
  Activity,
  Target,
  FileText,
  Bell,
  Search,
  Globe,
  Database,
  Cpu,
  Zap,
  Users
} from 'lucide-react'

interface SidebarProps {
  collapsed: boolean
  onThemeToggle: () => void
  currentTheme: string
}

const Sidebar: React.FC<SidebarProps> = ({
  collapsed,
  onThemeToggle,
  currentTheme
}) => {
  const location = useLocation()
  const [expandedSection, setExpandedSection] = useState<string | null>(null)

  // All navigation items available
  const navigationItems = [
    {
      title: 'Dashboard',
      icon: Home,
      href: '/dashboard',
      badge: null
    },
    {
      title: 'Trading',
      icon: TrendingUp,
      href: '/trading',
      badge: null
    },
    {
      title: 'Portfolio',
      icon: BarChart3,
      href: '/portfolio',
      badge: null
    },
    {
      title: 'Orders',
      icon: FileText,
      href: '/orders',
      badge: null
    },
    {
      title: 'Risk Management',
      icon: Shield,
      href: '/risk',
      badge: null
    },
    {
      title: 'VaR Calculator',
      icon: Target,
      href: '/risk/var',
      badge: null
    },
    {
      title: 'Stress Testing',
      icon: Activity,
      href: '/risk/stress',
      badge: null
    },
    {
      title: 'AI Analytics',
      icon: Brain,
      href: '/analytics',
      badge: 'AI'
    },
    {
      title: 'Yield Prediction',
      icon: TrendingUp,
      href: '/analytics/yield',
      badge: null
    },
    {
      title: 'Liquidity Analysis',
      icon: Activity,
      href: '/analytics/liquidity',
      badge: null
    },
    {
      title: 'Market Data',
      icon: Globe,
      href: '/market',
      badge: null
    },
    {
      title: 'Research',
      icon: FileText,
      href: '/research',
      badge: null
    },
    {
      title: 'Reports',
      icon: Database,
      href: '/reports',
      badge: null
    },
    {
      title: 'System Admin',
      icon: Cpu,
      href: '/admin',
      badge: 'Admin'
    },
    {
      title: 'User Management',
      icon: Users,
      href: '/admin/users',
      badge: null
    },
    {
      title: 'System Health',
      icon: Activity,
      href: '/admin/health',
      badge: null
    }
  ]

  const isActive = (href: string) => {
    if (href === '/dashboard') {
      return location.pathname === '/dashboard'
    }
    return location.pathname.startsWith(href)
  }

  const getThemeIcon = () => {
    switch (currentTheme) {
      case 'light':
        return Sun
      case 'dark':
        return Moon
      case 'auto':
        return Monitor
      default:
        return Moon
    }
  }

  const ThemeIcon = getThemeIcon()

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="flex h-full flex-col">
        {/* Logo and brand */}
        <div className="flex h-16 items-center justify-between border-b border-border px-4">
          <Link to="/dashboard" className="flex items-center space-x-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-blue">
              <Zap className="h-5 w-5 text-white" />
            </div>
            {!collapsed && (
              <span className="text-xl font-bold text-text-primary">BondX</span>
            )}
          </Link>
          {!collapsed && (
            <button
              onClick={onThemeToggle}
              className="rounded-lg p-2 text-text-secondary hover:bg-background-tertiary hover:text-text-primary"
            >
              <ThemeIcon className="h-4 w-4" />
            </button>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 overflow-y-auto p-4">
          {navigationItems.map((item) => {
            const Icon = item.icon
            return (
              <Link
                key={item.href}
                to={item.href}
                className={`nav-item ${isActive(item.href) ? 'active' : ''}`}
              >
                <Icon className="h-4 w-4" />
                {!collapsed && (
                  <>
                    <span className="ml-3 flex-1">{item.title}</span>
                    {item.badge && (
                      <span className="badge badge-info text-xs">
                        {item.badge}
                      </span>
                    )}
                  </>
                )}
              </Link>
            )
          })}
        </nav>

        {/* System info section */}
        {!collapsed && (
          <div className="border-t border-border p-4">
            <div className="flex items-center space-x-3">
              <div className="h-8 w-8 rounded-full bg-accent-green/20 flex items-center justify-center">
                <Cpu className="h-4 w-4 text-accent-green" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-text-primary truncate">
                  BondX System
                </p>
                <p className="text-xs text-text-secondary">
                  Live Trading Platform
                </p>
              </div>
            </div>
            
            <div className="mt-3 space-y-1">
              <div className="text-xs text-text-secondary">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 rounded-full bg-accent-green"></div>
                  <span>System Online</span>
                </div>
                <div className="flex items-center space-x-2 mt-1">
                  <div className="w-2 h-2 rounded-full bg-accent-blue"></div>
                  <span>WebSocket Active</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Collapse toggle button */}
        <div className="border-t border-border p-2">
          <button
            onClick={() => {/* Toggle sidebar logic handled by parent */}}
            className="w-full rounded-lg p-2 text-text-secondary hover:bg-background-tertiary hover:text-text-primary"
          >
            {collapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>
    </aside>
  )
}

export default Sidebar
