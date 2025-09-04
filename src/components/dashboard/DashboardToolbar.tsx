import React, { useState } from 'react'
import { useAppDispatch, useAppSelector } from '@/store'
import { selectAvailableLayouts, selectCurrentLayout, setCurrentLayout, saveDashboardLayout } from '@/store/slices/uiSlice'
import { Plus, Save, Layout, Settings, Download, Upload } from 'lucide-react'

const DashboardToolbar: React.FC = () => {
  const dispatch = useAppDispatch()
  const availableLayouts = useAppSelector(selectAvailableLayouts)
  const currentLayout = useAppSelector(selectCurrentLayout)
  const [showLayoutMenu, setShowLayoutMenu] = useState(false)
  const [showAddWidget, setShowAddWidget] = useState(false)

  const handleLayoutChange = (layoutId: string) => {
    const layout = availableLayouts.find(l => l.id === layoutId)
    if (layout) {
      dispatch(setCurrentLayout(layout))
    }
    setShowLayoutMenu(false)
  }

  const handleSaveLayout = async () => {
    if (currentLayout) {
      try {
        await dispatch(saveDashboardLayout(currentLayout)).unwrap()
        // Show success message
      } catch (error) {
        // Show error message
        console.error('Failed to save layout:', error)
      }
    }
  }

  return (
    <div className="flex items-center justify-between p-4 bg-background-secondary border border-border rounded-lg">
      {/* Left section */}
      <div className="flex items-center space-x-4">
        {/* Layout selector */}
        <div className="relative">
          <button
            onClick={() => setShowLayoutMenu(!showLayoutMenu)}
            className="flex items-center space-x-2 btn btn-secondary"
          >
            <Layout className="h-4 w-4" />
            <span>Layout</span>
          </button>

          {showLayoutMenu && (
            <div className="absolute top-full left-0 mt-2 w-48 rounded-lg border border-border bg-background-secondary py-1 shadow-lg z-50">
              {availableLayouts.map((layout) => (
                <button
                  key={layout.id}
                  onClick={() => handleLayoutChange(layout.id)}
                  className={`flex w-full items-center space-x-2 px-3 py-2 text-sm hover:bg-background-tertiary ${
                    currentLayout?.id === layout.id
                      ? 'text-accent-blue bg-accent-blue/10'
                      : 'text-text-primary'
                  }`}
                >
                  <span>{layout.name}</span>
                  {layout.isDefault && (
                    <span className="text-xs text-text-secondary">(Default)</span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Add widget button */}
        <button
          onClick={() => setShowAddWidget(!showAddWidget)}
          className="btn btn-primary"
        >
          <Plus className="h-4 w-4" />
          <span>Add Widget</span>
        </button>
      </div>

      {/* Right section */}
      <div className="flex items-center space-x-2">
        {/* Save layout */}
        <button
          onClick={handleSaveLayout}
          className="btn btn-secondary"
        >
          <Save className="h-4 w-4" />
          <span>Save Layout</span>
        </button>

        {/* Export layout */}
        <button className="btn btn-outline">
          <Download className="h-4 w-4" />
          <span>Export</span>
        </button>

        {/* Import layout */}
        <button className="btn btn-outline">
          <Upload className="h-4 w-4" />
          <span>Import</span>
        </button>

        {/* Dashboard settings */}
        <button className="btn btn-ghost">
          <Settings className="h-4 w-4" />
          <span>Settings</span>
        </button>
      </div>

      {/* Add widget menu */}
      {showAddWidget && (
        <div className="absolute top-full left-0 right-0 mt-2 p-4 bg-background-secondary border border-border rounded-lg shadow-lg z-50">
          <h4 className="text-sm font-medium text-text-primary mb-3">Add Widget</h4>
          
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {[
              { type: 'market_overview', name: 'Market Overview', icon: '📊' },
              { type: 'yield_curve', name: 'Yield Curve', icon: '📈' },
              { type: 'liquidity_pulse', name: 'Liquidity Pulse', icon: '💓' },
              { type: 'portfolio_summary', name: 'Portfolio Summary', icon: '💼' },
              { type: 'risk_metrics', name: 'Risk Metrics', icon: '⚠️' },
              { type: 'trading_activity', name: 'Trading Activity', icon: '🔄' },
              { type: 'economic_calendar', name: 'Economic Calendar', icon: '📅' },
              { type: 'currency_tracker', name: 'Currency Tracker', icon: '💱' },
              { type: 'news_feed', name: 'News Feed', icon: '📰' },
              { type: 'alerts', name: 'Alerts', icon: '🔔' }
            ].map((widget) => (
              <button
                key={widget.type}
                className="flex flex-col items-center p-3 rounded-lg border border-border hover:border-accent-blue hover:bg-accent-blue/5 transition-all duration-200"
                onClick={() => {
                  // TODO: Implement add widget logic
                  setShowAddWidget(false)
                }}
              >
                <span className="text-2xl mb-2">{widget.icon}</span>
                <span className="text-xs text-text-primary text-center">{widget.name}</span>
              </button>
            ))}
          </div>

          <button
            onClick={() => setShowAddWidget(false)}
            className="mt-4 w-full btn btn-secondary"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  )
}

export default DashboardToolbar
