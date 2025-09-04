import React, { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Helmet } from 'react-helmet-async'
import { useAppDispatch } from '@/store'
import { connectWebSocket } from '@/store/slices/websocketSlice'

// Layout components
import MainLayout from '@/components/layout/MainLayout'

// Page components
import Dashboard from '@/pages/dashboard/Dashboard'
import AnalyticsDashboard from '@/pages/analytics/AnalyticsDashboard'
import DemoShowcase from '@/components/demo/DemoShowcase'
import NotFound from '@/pages/NotFound'

const App: React.FC = () => {
  const dispatch = useAppDispatch()

  // Initialize app on mount
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Connect to WebSocket immediately
        await dispatch(connectWebSocket()).unwrap()
      } catch (error) {
        console.error('Failed to initialize app:', error)
      }
    }

    initializeApp()
  }, [dispatch])

  return (
    <>
      <Helmet>
        <title>BondX - Institutional Bond Trading Platform</title>
        <meta name="description" content="Next-generation institutional bond trading platform with AI-powered analytics and real-time risk management" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link rel="icon" href="/bondx-icon.svg" />
      </Helmet>

      <Routes>
        {/* Main dashboard route */}
        <Route
          path="/"
          element={
            <MainLayout>
              <Navigate to="/dashboard" replace />
            </MainLayout>
          }
        />
        <Route
          path="/dashboard"
          element={
            <MainLayout>
              <Dashboard />
            </MainLayout>
          }
        />

        {/* Trading routes */}
        <Route
          path="/trading/*"
          element={
            <MainLayout>
              <div>Trading Interface (Coming Soon)</div>
            </MainLayout>
          }
        />

        {/* Risk management routes */}
        <Route
          path="/risk/*"
          element={
            <MainLayout>
              <div>Risk Management (Coming Soon)</div>
            </MainLayout>
          }
        />

        {/* Analytics routes */}
        <Route
          path="/analytics"
          element={
            <MainLayout>
              <AnalyticsDashboard />
            </MainLayout>
          }
        />

        {/* Demo showcase route */}
        <Route
          path="/demo"
          element={<DemoShowcase />}
        />

        {/* 404 route */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </>
  )
}

export default App
