import React, { useState, useEffect } from 'react'
import { useAppDispatch, useAppSelector } from '@/store'
import { selectSidebarCollapsed, toggleSidebar } from '@/store/slices/uiSlice'
import { selectConnectionStatus, selectLatency } from '@/store/slices/websocketSlice'

import Sidebar from './Sidebar'
import Header from './Header'
import Breadcrumbs from './Breadcrumbs'
import NotificationCenter from '../ui/NotificationCenter'
import AnalyticsNav from '../navigation/AnalyticsNav'
import { useTheme } from '../providers/ThemeProvider'

interface MainLayoutProps {
  children: React.ReactNode
}

const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const dispatch = useAppDispatch()
  const sidebarCollapsed = useAppSelector(selectSidebarCollapsed)
  const wsStatus = useAppSelector(selectConnectionStatus)
  const wsLatency = useAppSelector(selectLatency)
  const { theme, setTheme } = useTheme()

  const [isMobile, setIsMobile] = useState(false)
  const [showNotificationCenter, setShowNotificationCenter] = useState(false)

  // Handle responsive behavior
  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768)
      if (window.innerWidth < 768 && !sidebarCollapsed) {
        dispatch(toggleSidebar())
      }
    }

    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [dispatch, sidebarCollapsed])

  // Handle escape key for mobile sidebar
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isMobile && !sidebarCollapsed) {
        dispatch(toggleSidebar())
      }
    }

    document.addEventListener('keydown', handleEscape)
    return () => document.removeEventListener('keydown', handleEscape)
  }, [dispatch, isMobile, sidebarCollapsed])

  const handleThemeToggle = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark')
  }

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <Sidebar
        collapsed={sidebarCollapsed}
        onThemeToggle={handleThemeToggle}
        currentTheme={theme}
      />

      {/* Main content area */}
      <div className={`flex flex-1 flex-col transition-all duration-300 ease-in-out ${
        sidebarCollapsed ? 'ml-0' : 'ml-70'
      }`}>
        {/* Header */}
        <Header
          onMenuToggle={() => dispatch(toggleSidebar())}
          sidebarCollapsed={sidebarCollapsed}
          wsStatus={wsStatus}
          wsLatency={wsLatency}
          onNotificationToggle={() => setShowNotificationCenter(!showNotificationCenter)}
          onThemeToggle={handleThemeToggle}
          currentTheme={theme}
        />

        {/* Analytics Navigation */}
        <AnalyticsNav />

        {/* Breadcrumbs */}
        <Breadcrumbs />

        {/* Main content */}
        <main className="flex-1 overflow-auto p-6">
          <div className="mx-auto max-w-7xl">
            {children}
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-border bg-background-secondary px-6 py-4">
          <div className="mx-auto max-w-7xl">
            <div className="flex items-center justify-between text-sm text-text-secondary">
              <div className="flex items-center space-x-4">
                <span>© 2024 BondX. All rights reserved.</span>
                <span>•</span>
                <span>Version 1.0.0</span>
              </div>
              <div className="flex items-center space-x-4">
                <span className="flex items-center space-x-2">
                  <span>WebSocket:</span>
                  <span className={`status-indicator ${
                    wsStatus === 'connected' ? 'status-online' :
                    wsStatus === 'connecting' ? 'status-connecting' : 'status-offline'
                  }`} />
                  <span className="text-xs">
                    {wsStatus === 'connected' ? `${wsLatency}ms` : wsStatus}
                  </span>
                </span>
                <span>•</span>
                <span>Theme: {theme}</span>
              </div>
            </div>
          </div>
        </footer>
      </div>

      {/* Notification Center */}
      {showNotificationCenter && (
        <NotificationCenter
          onClose={() => setShowNotificationCenter(false)}
        />
      )}

      {/* Mobile overlay */}
      {isMobile && !sidebarCollapsed && (
        <div
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={() => dispatch(toggleSidebar())}
        />
      )}
    </div>
  )
}

export default MainLayout
