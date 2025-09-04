import React, { useState, useRef, useEffect } from 'react'
import { 
  Menu, 
  Search, 
  Bell, 
  Sun, 
  Moon, 
  Monitor,
  ChevronDown,
  Activity,
  Wifi,
  WifiOff,
  Clock
} from 'lucide-react'

interface HeaderProps {
  onMenuToggle: () => void
  sidebarCollapsed: boolean
  wsStatus: string
  wsLatency: number
  onNotificationToggle: () => void
  onThemeToggle: () => void
  currentTheme: string
}

const Header: React.FC<HeaderProps> = ({
  onMenuToggle,
  sidebarCollapsed,
  wsStatus,
  wsLatency,
  onNotificationToggle,
  onThemeToggle,
  currentTheme
}) => {
  const [showThemeMenu, setShowThemeMenu] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const themeMenuRef = useRef<HTMLDivElement>(null)

  // Close menus when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (themeMenuRef.current && !themeMenuRef.current.contains(event.target as Node)) {
        setShowThemeMenu(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

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

  const getThemeLabel = () => {
    switch (currentTheme) {
      case 'light':
        return 'Light'
      case 'dark':
        return 'Dark'
      case 'auto':
        return 'Auto'
      default:
        return 'Dark'
    }
  }

  const ThemeIcon = getThemeIcon()

  const getConnectionStatusIcon = () => {
    switch (wsStatus) {
      case 'connected':
        return <Wifi className="h-4 w-4 text-accent-green" />
      case 'connecting':
        return <Clock className="h-4 w-4 text-accent-amber animate-spin" />
      case 'reconnecting':
        return <Activity className="h-4 w-4 text-accent-amber animate-pulse" />
      default:
        return <WifiOff className="h-4 w-4 text-accent-red" />
    }
  }

  const getConnectionStatusColor = () => {
    switch (wsStatus) {
      case 'connected':
        return 'text-accent-green'
      case 'connecting':
      case 'reconnecting':
        return 'text-accent-amber'
      default:
        return 'text-accent-red'
    }
  }

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault()
    // TODO: Implement global search
    console.log('Search query:', searchQuery)
  }

  return (
    <header className="header">
      <div className="flex h-16 items-center justify-between px-6">
        {/* Left section */}
        <div className="flex items-center space-x-4">
          {/* Menu toggle button */}
          <button
            onClick={onMenuToggle}
            className="rounded-lg p-2 text-text-secondary hover:bg-background-tertiary hover:text-text-primary lg:hidden"
          >
            <Menu className="h-5 w-5" />
          </button>

          {/* Breadcrumb or title */}
          <div className="hidden md:block">
            <h1 className="text-lg font-semibold text-text-primary">
              BondX Platform
            </h1>
          </div>
        </div>

        {/* Center section - Search */}
        <div className="flex-1 max-w-md mx-8 hidden lg:block">
          <form onSubmit={handleSearch} className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted" />
            <input
              type="text"
              placeholder="Search bonds, markets, analytics..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full rounded-lg border border-border bg-background-secondary pl-10 pr-4 py-2 text-sm text-text-primary placeholder:text-text-muted focus:border-accent-blue focus:outline-none focus:ring-1 focus:ring-accent-blue/50"
            />
          </form>
        </div>

        {/* Right section */}
        <div className="flex items-center space-x-4">
          {/* WebSocket status */}
          <div className="hidden sm:flex items-center space-x-2 text-sm">
            <span className="text-text-secondary">WS:</span>
            <div className="flex items-center space-x-1">
              {getConnectionStatusIcon()}
              <span className={getConnectionStatusColor()}>
                {wsStatus === 'connected' ? `${wsLatency}ms` : wsStatus}
              </span>
            </div>
          </div>

          {/* Theme toggle */}
          <div className="relative" ref={themeMenuRef}>
            <button
              onClick={() => setShowThemeMenu(!showThemeMenu)}
              className="flex items-center space-x-2 rounded-lg px-3 py-2 text-sm text-text-secondary hover:bg-background-tertiary hover:text-text-primary"
            >
              <ThemeIcon className="h-4 w-4" />
              <span className="hidden sm:block">{getThemeLabel()}</span>
              <ChevronDown className="h-4 w-4" />
            </button>

            {showThemeMenu && (
              <div className="absolute right-0 top-full mt-1 w-32 rounded-lg border border-border bg-background-secondary py-1 shadow-lg z-50">
                <button
                  onClick={() => {
                    onThemeToggle()
                    setShowThemeMenu(false)
                  }}
                  className="flex w-full items-center space-x-2 px-3 py-2 text-sm text-text-primary hover:bg-background-tertiary"
                >
                  <Sun className="h-4 w-4" />
                  <span>Light</span>
                </button>
                <button
                  onClick={() => {
                    onThemeToggle()
                    setShowThemeMenu(false)
                  }}
                  className="flex w-full items-center space-x-2 px-3 py-2 text-sm text-text-primary hover:bg-background-tertiary"
                >
                  <Moon className="h-4 w-4" />
                  <span>Dark</span>
                </button>
                <button
                  onClick={() => {
                    onThemeToggle()
                    setShowThemeMenu(false)
                  }}
                  className="flex w-full items-center space-x-2 px-3 py-2 text-sm text-text-primary hover:bg-background-tertiary"
                >
                  <Monitor className="h-4 w-4" />
                  <span>Auto</span>
                </button>
              </div>
            )}
          </div>

          {/* Notifications */}
          <button
            onClick={onNotificationToggle}
            className="relative rounded-lg p-2 text-text-secondary hover:bg-background-tertiary hover:text-text-primary"
          >
            <Bell className="h-5 w-5" />
            {/* Notification badge */}
            <span className="absolute -top-1 -right-1 h-3 w-3 rounded-full bg-accent-red text-xs text-white flex items-center justify-center">
              3
            </span>
          </button>

          {/* System Status */}
          <div className="flex items-center space-x-2 text-sm">
            <div className="w-2 h-2 rounded-full bg-accent-green"></div>
            <span className="text-text-secondary">System Online</span>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
