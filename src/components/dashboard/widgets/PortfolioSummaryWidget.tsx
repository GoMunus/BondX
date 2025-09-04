import React, { useEffect, useState } from 'react'
import { Widget } from '@/types'
import { apiService, PortfolioSummary, formatCurrency, formatPercentage } from '@/services/api'

interface PortfolioSummaryWidgetProps {
  widget: Widget
  userRole: string
  isExpanded: boolean
}

const PortfolioSummaryWidget: React.FC<PortfolioSummaryWidgetProps> = ({ widget, userRole, isExpanded }) => {
  const [portfolioData, setPortfolioData] = useState<PortfolioSummary | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const fetchPortfolioData = async () => {
    try {
      setError(null)
      const data = await apiService.getPortfolioSummary()
      setPortfolioData(data)
      setLastUpdated(new Date())
    } catch (err) {
      console.error('Error fetching portfolio data:', err)
      setError('Failed to fetch portfolio data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchPortfolioData()
    
    // Set up periodic refresh every 30 seconds
    const interval = setInterval(fetchPortfolioData, 30000)
    return () => clearInterval(interval)
  }, [])

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-text-secondary">
        <div className="text-red-400 mb-2">⚠️ Error</div>
        <p className="text-sm">{error}</p>
        <button 
          onClick={fetchPortfolioData}
          className="mt-2 px-3 py-1 text-xs bg-primary-600 hover:bg-primary-700 rounded transition-colors"
        >
          Retry
        </button>
      </div>
    )
  }

  if (!portfolioData) {
    return (
      <div className="h-full flex items-center justify-center text-text-secondary">
        <p>No portfolio data available</p>
      </div>
    )
  }

  const { summary, performance, allocation } = portfolioData

  return (
    <div className="h-full p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-text-primary mb-1">Portfolio Summary</h3>
          <p className="text-sm text-text-secondary">
            Portfolio {summary.portfolio_id}
          </p>
        </div>
        <div className="text-right">
          <div className="text-sm text-text-secondary">Last Updated</div>
          <div className="text-xs text-text-muted">
            {lastUpdated?.toLocaleTimeString() || 'Loading...'}
          </div>
        </div>
      </div>

      {/* Main Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Total AUM */}
        <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
          <div className="text-sm text-text-secondary mb-1">Total AUM</div>
          <div className="text-xl font-bold text-text-primary">
            {formatCurrency(summary.total_aum)}
          </div>
        </div>

        {/* Daily P&L */}
        <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
          <div className="text-sm text-text-secondary mb-1">Daily P&L</div>
          <div className={`text-xl font-bold ${summary.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatCurrency(summary.daily_pnl)}
          </div>
          <div className={`text-sm ${summary.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {formatPercentage(summary.daily_pnl_percent)}
          </div>
        </div>

        {/* Active Positions */}
        <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
          <div className="text-sm text-text-secondary mb-1">Active Positions</div>
          <div className="text-xl font-bold text-text-primary">
            {summary.active_positions}
          </div>
          <div className="text-sm text-text-secondary">
            {summary.total_bonds} bonds
          </div>
        </div>

        {/* Average Rating */}
        <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
          <div className="text-sm text-text-secondary mb-1">Avg Rating</div>
          <div className="text-xl font-bold text-text-primary">
            {summary.average_rating}
          </div>
          <div className="text-sm text-text-secondary">
            Duration: {summary.weighted_duration}Y
          </div>
        </div>
      </div>

      {isExpanded && (
        <>
          {/* Performance Metrics */}
          <div className="mb-6">
            <h4 className="text-md font-semibold text-text-primary mb-3">Performance</h4>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <div className="bg-card-background/30 rounded p-3">
                <div className="text-xs text-text-secondary mb-1">MTD Return</div>
                <div className="text-lg font-semibold text-green-400">
                  {formatPercentage(performance.mtd_return)}
                </div>
              </div>
              <div className="bg-card-background/30 rounded p-3">
                <div className="text-xs text-text-secondary mb-1">QTD Return</div>
                <div className="text-lg font-semibold text-green-400">
                  {formatPercentage(performance.qtd_return)}
                </div>
              </div>
              <div className="bg-card-background/30 rounded p-3">
                <div className="text-xs text-text-secondary mb-1">YTD Return</div>
                <div className="text-lg font-semibold text-green-400">
                  {formatPercentage(performance.ytd_return)}
                </div>
              </div>
              <div className="bg-card-background/30 rounded p-3">
                <div className="text-xs text-text-secondary mb-1">Volatility</div>
                <div className="text-lg font-semibold text-text-primary">
                  {formatPercentage(performance.trailing_volatility)}
                </div>
              </div>
            </div>
          </div>

          {/* Allocation Breakdown */}
          <div className="mb-6">
            <h4 className="text-md font-semibold text-text-primary mb-3">Allocation</h4>
            <div className="space-y-3">
              {/* Government Bonds */}
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-secondary">Government</span>
                <div className="flex items-center gap-2">
                  <div className="w-20 h-2 bg-background rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-500"
                      style={{ width: `${allocation.government}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium text-text-primary w-12 text-right">
                    {formatPercentage(allocation.government)}
                  </span>
                </div>
              </div>

              {/* Corporate Bonds */}
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-secondary">Corporate</span>
                <div className="flex items-center gap-2">
                  <div className="w-20 h-2 bg-background rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-green-500"
                      style={{ width: `${allocation.corporate}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium text-text-primary w-12 text-right">
                    {formatPercentage(allocation.corporate)}
                  </span>
                </div>
              </div>

              {/* PSU Bonds */}
              <div className="flex items-center justify-between">
                <span className="text-sm text-text-secondary">PSU</span>
                <div className="flex items-center gap-2">
                  <div className="w-20 h-2 bg-background rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-purple-500"
                      style={{ width: `${allocation.psu}%` }}
                    />
                  </div>
                  <span className="text-sm font-medium text-text-primary w-12 text-right">
                    {formatPercentage(allocation.psu)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Additional Metrics */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-card-background/30 rounded p-3">
              <div className="text-xs text-text-secondary mb-1">Cash Balance</div>
              <div className="text-sm font-semibold text-text-primary">
                {formatCurrency(summary.cash_balance)}
              </div>
            </div>
            <div className="bg-card-background/30 rounded p-3">
              <div className="text-xs text-text-secondary mb-1">Leverage Ratio</div>
              <div className="text-sm font-semibold text-text-primary">
                {summary.leverage_ratio.toFixed(2)}x
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default PortfolioSummaryWidget
