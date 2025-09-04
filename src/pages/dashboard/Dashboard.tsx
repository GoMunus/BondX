import React, { useEffect, useState } from 'react'
import { useAppDispatch, useAppSelector } from '@/store'
import { selectUserRole } from '@/store/slices/authSlice'
import { selectCurrentLayout, selectWidgets, fetchDashboardLayouts } from '@/store/slices/uiSlice'
import { selectConnectionStatus } from '@/store/slices/websocketSlice'

import DashboardGrid from '@/components/dashboard/DashboardGrid'
import DashboardToolbar from '@/components/dashboard/DashboardToolbar'
import WelcomeMessage from '@/components/dashboard/WelcomeMessage'
import QuickActions from '@/components/dashboard/QuickActions'
import SystemStatus from '@/components/dashboard/SystemStatus'

const Dashboard: React.FC = () => {
  const dispatch = useAppDispatch()
  const userRole = useAppSelector(selectUserRole)
  const currentLayout = useAppSelector(selectCurrentLayout)
  const widgets = useAppSelector(selectWidgets)
  const wsStatus = useAppSelector(selectConnectionStatus)

  const [isLoading, setIsLoading] = useState(true)
  const [showWelcome, setShowWelcome] = useState(true)

  useEffect(() => {
    const initializeDashboard = async () => {
      try {
        setIsLoading(true)
        await dispatch(fetchDashboardLayouts()).unwrap()
      } catch (error) {
        console.error('Failed to fetch dashboard layouts:', error)
      } finally {
        setIsLoading(false)
      }
    }

    initializeDashboard()
  }, [dispatch])

  useEffect(() => {
    // Hide welcome message after 5 seconds
    const timer = setTimeout(() => {
      setShowWelcome(false)
    }, 5000)

    return () => clearTimeout(timer)
  }, [])

  if (isLoading) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="text-center">
          <div className="spinner h-8 w-8 mx-auto mb-4"></div>
          <p className="text-text-secondary">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Welcome message */}
      {showWelcome && (
        <WelcomeMessage userRole={userRole} />
      )}

      {/* Quick actions */}
      <QuickActions userRole={userRole} />

      {/* System status */}
      <SystemStatus wsStatus={wsStatus} />

      {/* Dashboard toolbar */}
      <DashboardToolbar />

      {/* Main dashboard grid */}
      <DashboardGrid
        widgets={widgets}
        layout={currentLayout}
        userRole={userRole}
      />

      {/* Empty state if no widgets */}
      {(!widgets || widgets.length === 0) && (
        <div className="text-center py-12">
          <div className="mx-auto h-24 w-24 text-text-muted mb-4">
            <svg
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              className="w-full h-full"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-text-primary mb-2">
            No dashboard widgets configured
          </h3>
          <p className="text-text-secondary mb-4">
            Get started by adding widgets to your dashboard
          </p>
          <button className="btn btn-primary">
            Add Widget
          </button>
        </div>
      )}
    </div>
  )
}

export default Dashboard
