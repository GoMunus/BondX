import React, { useState, useCallback } from 'react'
import { Widget } from '@/types'
import { useAppDispatch } from '@/store'
import { selectWidget, removeWidget, updateWidgetConfig } from '@/store/slices/uiSlice'
import { 
  MoreVertical, 
  X, 
  Settings, 
  RefreshCw, 
  Maximize2,
  Minimize2,
  GripVertical
} from 'lucide-react'

import MarketOverviewWidget from './widgets/MarketOverviewWidget'
import YieldCurveWidget from './widgets/YieldCurveWidget'
import LiquidityPulseWidget from './widgets/LiquidityPulseWidget'
import PortfolioSummaryWidget from './widgets/PortfolioSummaryWidget'
import RiskMetricsWidget from './widgets/RiskMetricsWidget'
import TradingActivityWidget from './widgets/TradingActivityWidget'
import CorporateBondsWidget from './widgets/CorporateBondsWidget'
import EconomicCalendarWidget from './widgets/EconomicCalendarWidget'
import CurrencyTrackerWidget from './widgets/CurrencyTrackerWidget'
import NewsFeedWidget from './widgets/NewsFeedWidget'
import AlertsWidget from './widgets/AlertsWidget'
import FractionalOwnershipWidget from './widgets/FractionalOwnershipWidget'
import AuctionWidget from './widgets/AuctionWidget'

interface WidgetComponentProps {
  widget: Widget
  userRole: string
}

const WidgetComponent: React.FC<WidgetComponentProps> = ({ widget, userRole }) => {
  const dispatch = useAppDispatch()
  const [isExpanded, setIsExpanded] = useState(false)
  const [showSettings, setShowSettings] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)

  const handleSelect = useCallback(() => {
    dispatch(selectWidget(widget))
  }, [dispatch, widget])

  const handleRemove = useCallback(() => {
    dispatch(removeWidget(widget.id))
  }, [dispatch, widget.id])

  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true)
    try {
      // Simulate refresh delay
      await new Promise(resolve => setTimeout(resolve, 1000))
      // TODO: Implement actual widget refresh logic
    } finally {
      setIsRefreshing(false)
    }
  }, [])

  const handleToggleExpand = useCallback(() => {
    setIsExpanded(!isExpanded)
  }, [isExpanded])

  const handleToggleSettings = useCallback(() => {
    setShowSettings(!showSettings)
  }, [showSettings])

  // Render widget content based on type
  const renderWidgetContent = () => {
    const commonProps = {
      widget,
      userRole,
      isExpanded
    }

    switch (widget.type) {
      case 'market_overview':
        return <MarketOverviewWidget {...commonProps} />
      case 'yield_curve':
        return <YieldCurveWidget {...commonProps} />
      case 'liquidity_pulse':
        return <LiquidityPulseWidget {...commonProps} />
      case 'portfolio_summary':
        return <PortfolioSummaryWidget {...commonProps} />
      case 'risk_metrics':
        return <RiskMetricsWidget {...commonProps} />
      case 'trading_activity':
        return <TradingActivityWidget {...commonProps} />
      case 'corporate_bonds':
        return <CorporateBondsWidget {...commonProps} />
      case 'economic_calendar':
        return <EconomicCalendarWidget {...commonProps} />
      case 'currency_tracker':
        return <CurrencyTrackerWidget {...commonProps} />
      case 'news_feed':
        return <NewsFeedWidget {...commonProps} />
      case 'alerts':
        return <AlertsWidget {...commonProps} />
      case 'fractional_ownership':
        return <FractionalOwnershipWidget />
      case 'auction':
        return <AuctionWidget />
      default:
        return (
          <div className="flex h-full items-center justify-center text-text-secondary">
            <p>Unknown widget type: {widget.type}</p>
          </div>
        )
    }
  }

  return (
    <div 
      className={`widget ${isExpanded ? 'expanded' : ''}`}
      onClick={handleSelect}
    >
      {/* Widget header */}
      <div className="widget-header">
        <div className="flex items-center space-x-2">
          {/* Drag handle */}
          <div className="widget-drag-handle cursor-move">
            <GripVertical className="h-4 w-4 text-text-muted" />
          </div>
          
          {/* Title */}
          <h3 className="card-title">{widget.config.title}</h3>
        </div>

        {/* Header actions */}
        <div className="flex items-center space-x-1">
          {/* Refresh button */}
          <button
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="rounded p-1 text-text-secondary hover:bg-background-tertiary hover:text-text-primary transition-colors duration-200"
            title="Refresh widget"
          >
            <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          </button>

          {/* Expand/collapse button */}
          <button
            onClick={handleToggleExpand}
            className="rounded p-1 text-text-secondary hover:bg-background-tertiary hover:text-text-primary transition-colors duration-200"
            title={isExpanded ? 'Collapse' : 'Expand'}
          >
            {isExpanded ? (
              <Minimize2 className="h-4 w-4" />
            ) : (
              <Maximize2 className="h-4 w-4" />
            )}
          </button>

          {/* Settings button */}
          <button
            onClick={handleToggleSettings}
            className="rounded p-1 text-text-secondary hover:bg-background-tertiary hover:text-text-primary transition-colors duration-200"
            title="Widget settings"
          >
            <Settings className="h-4 w-4" />
          </button>

          {/* Remove button */}
          <button
            onClick={handleRemove}
            className="rounded p-1 text-text-secondary hover:bg-background-tertiary hover:text-accent-red transition-colors duration-200"
            title="Remove widget"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Widget content */}
      <div className="widget-content">
        {renderWidgetContent()}
      </div>

      {/* Settings panel */}
      {showSettings && (
        <div className="absolute top-full left-0 right-0 z-10 mt-2 rounded-lg border border-border bg-background-secondary p-4 shadow-lg">
          <h4 className="text-sm font-medium text-text-primary mb-3">Widget Settings</h4>
          
          <div className="space-y-3">
            {/* Title input */}
            <div>
              <label className="block text-xs font-medium text-text-secondary mb-1">
                Title
              </label>
              <input
                type="text"
                value={widget.config.title}
                onChange={(e) => dispatch(updateWidgetConfig({
                  widgetId: widget.id,
                  config: { title: e.target.value }
                }))}
                className="input text-sm"
              />
            </div>

            {/* Refresh interval */}
            <div>
              <label className="block text-xs font-medium text-text-secondary mb-1">
                Refresh Interval (seconds)
              </label>
              <input
                type="number"
                min="0"
                value={widget.config.refreshInterval || 0}
                onChange={(e) => dispatch(updateWidgetConfig({
                  widgetId: widget.id,
                  config: { refreshInterval: parseInt(e.target.value) || 0 }
                }))}
                className="input text-sm"
              />
            </div>

            {/* Show header toggle */}
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id={`show-header-${widget.id}`}
                checked={widget.config.showHeader !== false}
                onChange={(e) => dispatch(updateWidgetConfig({
                  widgetId: widget.id,
                  config: { showHeader: e.target.checked }
                }))}
                className="rounded border-border bg-background-secondary text-accent-blue focus:ring-accent-blue/50"
              />
              <label htmlFor={`show-header-${widget.id}`} className="text-xs text-text-secondary">
                Show header
              </label>
            </div>

            {/* Show footer toggle */}
            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                id={`show-footer-${widget.id}`}
                checked={widget.config.showFooter !== false}
                onChange={(e) => dispatch(updateWidgetConfig({
                  widgetId: widget.id,
                  config: { showFooter: e.target.checked }
                }))}
                className="rounded border-border bg-background-secondary text-accent-blue focus:ring-accent-blue/50"
              />
              <label htmlFor={`show-footer-${widget.id}`} className="text-xs text-text-secondary">
                Show footer
              </label>
            </div>
          </div>

          {/* Close settings */}
          <button
            onClick={handleToggleSettings}
            className="mt-4 w-full btn btn-secondary text-sm"
          >
            Close
          </button>
        </div>
      )}

      {/* Resize handle */}
      <div className="widget-resize-handle">
        <div className="h-4 w-4 cursor-se-resize opacity-50 hover:opacity-100">
          <svg viewBox="0 0 24 24" fill="currentColor" className="h-full w-full">
            <path d="M22 22H20V20H22V22ZM22 18H20V16H22V18ZM18 22H16V20H18V22ZM18 18H16V16H18V18ZM14 22H12V20H14V22ZM22 14H20V12H22V14Z" />
          </svg>
        </div>
      </div>
    </div>
  )
}

export default WidgetComponent
