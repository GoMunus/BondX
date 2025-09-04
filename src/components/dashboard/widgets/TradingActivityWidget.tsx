import React, { useEffect, useState } from 'react'
import { Widget } from '@/types'
import { apiService, TradingActivity, formatCurrency, formatNumber } from '@/services/api'

interface TradingActivityWidgetProps {
  widget: Widget
  userRole: string
  isExpanded: boolean
}

const TradingActivityWidget: React.FC<TradingActivityWidgetProps> = ({ widget, userRole, isExpanded }) => {
  const [tradingData, setTradingData] = useState<TradingActivity | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const fetchTradingData = async () => {
    try {
      setError(null)
      const data = await apiService.getTradingActivity(20)
      setTradingData(data)
      setLastUpdated(new Date())
    } catch (err) {
      console.error('Error fetching trading data:', err)
      setError('Failed to fetch trading data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchTradingData()
    
    // Set up periodic refresh every 10 seconds for trading activity
    const interval = setInterval(fetchTradingData, 10000)
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
          onClick={fetchTradingData}
          className="mt-2 px-3 py-1 text-xs bg-primary-600 hover:bg-primary-700 rounded transition-colors"
        >
          Retry
        </button>
      </div>
    )
  }

  if (!tradingData) {
    return (
      <div className="h-full flex items-center justify-center text-text-secondary">
        <p>No trading data available</p>
      </div>
    )
  }

  const { recent_trades, summary } = tradingData

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('en-IN', { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    })
  }

  const getSideColor = (side: string) => {
    return side === 'BUY' ? 'text-green-400' : 'text-red-400'
  }

  const getSideIcon = (side: string) => {
    return side === 'BUY' ? '↗' : '↘'
  }

  return (
    <div className="h-full p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-text-primary mb-1">Trading Activity</h3>
          <p className="text-sm text-text-secondary">
            Recent transactions
          </p>
        </div>
        <div className="text-right">
          <div className="text-sm text-text-secondary">Last Updated</div>
          <div className="text-xs text-text-muted">
            {lastUpdated?.toLocaleTimeString() || 'Loading...'}
          </div>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
          <div className="text-sm text-text-secondary mb-1">Total Trades</div>
          <div className="text-xl font-bold text-text-primary">
            {summary.total_trades}
          </div>
        </div>
        <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
          <div className="text-sm text-text-secondary mb-1">Total Volume</div>
          <div className="text-xl font-bold text-text-primary">
            {formatCurrency(summary.total_volume)}
          </div>
        </div>
        <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
          <div className="text-sm text-text-secondary mb-1">Buy Trades</div>
          <div className="text-xl font-bold text-green-400">
            {summary.buy_trades}
          </div>
          <div className="text-sm text-green-400">
            {formatCurrency(summary.buy_volume)}
          </div>
        </div>
        <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
          <div className="text-sm text-text-secondary mb-1">Sell Trades</div>
          <div className="text-xl font-bold text-red-400">
            {summary.sell_trades}
          </div>
          <div className="text-sm text-red-400">
            {formatCurrency(summary.sell_volume)}
          </div>
        </div>
      </div>

      {/* Recent Trades List */}
      <div className="mb-4">
        <h4 className="text-md font-semibold text-text-primary mb-3">Recent Trades</h4>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {recent_trades.map((trade) => (
            <div 
              key={trade.trade_id} 
              className="flex items-center justify-between p-3 bg-card-background/30 rounded-lg hover:bg-card-background/50 transition-colors"
            >
              <div className="flex items-center gap-3">
                <div className={`text-lg ${getSideColor(trade.side)}`}>
                  {getSideIcon(trade.side)}
                </div>
                <div>
                  <div className="text-sm font-medium text-text-primary">
                    {trade.bond_name}
                  </div>
                  <div className="text-xs text-text-secondary">
                    {trade.bond_id} • {trade.venue}
                    {(trade as any).sector && (
                      <span className="ml-2 px-2 py-1 bg-blue-500/20 text-blue-400 rounded text-xs">
                        {(trade as any).sector}
                      </span>
                    )}
                  </div>
                  {(trade as any).issuer && (
                    <div className="text-xs text-text-muted">
                      {(trade as any).issuer}
                    </div>
                  )}
                </div>
              </div>
              
              <div className="text-right">
                <div className="flex items-center gap-2">
                  <span className={`text-sm font-medium ${getSideColor(trade.side)}`}>
                    {trade.side}
                  </span>
                  <span className="text-sm text-text-primary">
                    {formatNumber(trade.quantity, 0)}
                  </span>
                </div>
                <div className="text-xs text-text-secondary">
                  @ {formatNumber(trade.price, 2)}
                </div>
              </div>
              
              <div className="text-right">
                <div className="text-sm font-medium text-text-primary">
                  {formatCurrency(trade.trade_value)}
                </div>
                <div className="text-xs text-text-secondary">
                  {formatTime(trade.timestamp)}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {isExpanded && (
        <>
          {/* Trading Statistics */}
          <div className="mb-6">
            <h4 className="text-md font-semibold text-text-primary mb-3">Trading Statistics</h4>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-card-background/30 rounded p-4">
                <div className="text-sm text-text-secondary mb-1">Average Trade Size</div>
                <div className="text-lg font-semibold text-text-primary">
                  {formatCurrency(summary.avg_trade_size)}
                </div>
              </div>
              <div className="bg-card-background/30 rounded p-4">
                <div className="text-sm text-text-secondary mb-1">Buy/Sell Ratio</div>
                <div className="text-lg font-semibold text-text-primary">
                  {summary.sell_trades > 0 ? (summary.buy_trades / summary.sell_trades).toFixed(2) : 'N/A'}
                </div>
              </div>
            </div>
          </div>

          {/* Market Activity */}
          <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <span className="text-sm font-semibold text-blue-400">Market Activity</span>
            </div>
            <p className="text-sm text-text-secondary">
              {summary.total_trades > 50 ? 'High' : summary.total_trades > 20 ? 'Moderate' : 'Low'} trading activity detected. 
              Recent trades show {summary.buy_trades > summary.sell_trades ? 'buying' : 'selling'} pressure in the market.
            </p>
          </div>
        </>
      )}
    </div>
  )
}

export default TradingActivityWidget
