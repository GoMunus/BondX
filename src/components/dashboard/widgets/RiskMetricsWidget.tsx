import React, { useEffect, useState } from 'react'
import { Widget } from '@/types'
import { apiService, RiskMetrics, formatCurrency, formatPercentage, formatNumber } from '@/services/api'

interface RiskMetricsWidgetProps {
  widget: Widget
  userRole: string
  isExpanded: boolean
}

const RiskMetricsWidget: React.FC<RiskMetricsWidgetProps> = ({ widget, userRole, isExpanded }) => {
  const [riskData, setRiskData] = useState<RiskMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const fetchRiskData = async () => {
    try {
      setError(null)
      const data = await apiService.getRiskMetrics()
      setRiskData(data)
      setLastUpdated(new Date())
    } catch (err) {
      console.error('Error fetching risk data:', err)
      setError('Failed to fetch risk data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchRiskData()
    
    // Set up periodic refresh every 60 seconds (risk calculations are more expensive)
    const interval = setInterval(fetchRiskData, 60000)
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
          onClick={fetchRiskData}
          className="mt-2 px-3 py-1 text-xs bg-primary-600 hover:bg-primary-700 rounded transition-colors"
        >
          Retry
        </button>
      </div>
    )
  }

  if (!riskData) {
    return (
      <div className="h-full flex items-center justify-center text-text-secondary">
        <p>No risk data available</p>
      </div>
    )
  }

  const { var_metrics, duration_risk, concentration_risk, liquidity_metrics, stress_scenarios } = riskData

  const getRiskColor = (value: number, thresholds: { low: number; medium: number }) => {
    if (value <= thresholds.low) return 'text-green-400'
    if (value <= thresholds.medium) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getLiquidityRiskColor = (risk: string) => {
    switch (risk.toUpperCase()) {
      case 'LOW': return 'text-green-400'
      case 'MEDIUM': return 'text-yellow-400'
      case 'HIGH': return 'text-red-400'
      default: return 'text-text-primary'
    }
  }

  return (
    <div className="h-full p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-text-primary mb-1">Risk Metrics</h3>
          <p className="text-sm text-text-secondary">
            Portfolio {riskData.portfolio_id}
          </p>
        </div>
        <div className="text-right">
          <div className="text-sm text-text-secondary">Last Updated</div>
          <div className="text-xs text-text-muted">
            {lastUpdated?.toLocaleTimeString() || 'Loading...'}
          </div>
        </div>
      </div>

      {/* VaR Metrics */}
      <div className="mb-6">
        <h4 className="text-md font-semibold text-text-primary mb-3">Value at Risk</h4>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
            <div className="text-sm text-text-secondary mb-1">VaR 95% (1D)</div>
            <div className="text-xl font-bold text-red-400">
              {formatCurrency(var_metrics.var_95_1d)}
            </div>
          </div>
          <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
            <div className="text-sm text-text-secondary mb-1">VaR 99% (1D)</div>
            <div className="text-xl font-bold text-red-500">
              {formatCurrency(var_metrics.var_99_1d)}
            </div>
          </div>
          <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
            <div className="text-sm text-text-secondary mb-1">CVaR 95% (1D)</div>
            <div className="text-xl font-bold text-red-600">
              {formatCurrency(var_metrics.cvar_95_1d)}
            </div>
          </div>
        </div>
      </div>

      {/* Duration Risk */}
      <div className="mb-6">
        <h4 className="text-md font-semibold text-text-primary mb-3">Duration Risk</h4>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
            <div className="text-sm text-text-secondary mb-1">Modified Duration</div>
            <div className="text-xl font-bold text-text-primary">
              {formatNumber(duration_risk.modified_duration, 1)} years
            </div>
          </div>
          <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
            <div className="text-sm text-text-secondary mb-1">Effective Duration</div>
            <div className="text-xl font-bold text-text-primary">
              {formatNumber(duration_risk.effective_duration, 1)} years
            </div>
          </div>
          <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
            <div className="text-sm text-text-secondary mb-1">Convexity</div>
            <div className="text-xl font-bold text-text-primary">
              {formatNumber(duration_risk.convexity, 1)}
            </div>
          </div>
        </div>
      </div>

      {/* Concentration Risk */}
      <div className="mb-6">
        <h4 className="text-md font-semibold text-text-primary mb-3">Concentration Risk</h4>
        <div className="space-y-3">
          <div className="flex items-center justify-between p-3 bg-card-background/30 rounded">
            <span className="text-sm text-text-secondary">Issuer Concentration</span>
            <div className="text-right">
              <div className={`text-lg font-semibold ${getRiskColor(concentration_risk.issuer_concentration * 100, { low: 10, medium: 20 })}`}>
                {formatPercentage(concentration_risk.issuer_concentration * 100)}
              </div>
              <div className="text-xs text-text-muted">Max single issuer</div>
            </div>
          </div>
          <div className="flex items-center justify-between p-3 bg-card-background/30 rounded">
            <span className="text-sm text-text-secondary">Sector Concentration</span>
            <div className="text-right">
              <div className={`text-lg font-semibold ${getRiskColor(concentration_risk.sector_concentration * 100, { low: 25, medium: 40 })}`}>
                {formatPercentage(concentration_risk.sector_concentration * 100)}
              </div>
              <div className="text-xs text-text-muted">Max single sector</div>
            </div>
          </div>
          <div className="flex items-center justify-between p-3 bg-card-background/30 rounded">
            <span className="text-sm text-text-secondary">Rating Concentration</span>
            <div className="text-right">
              <div className={`text-lg font-semibold ${getRiskColor(concentration_risk.rating_concentration * 100, { low: 15, medium: 30 })}`}>
                {formatPercentage(concentration_risk.rating_concentration * 100)}
              </div>
              <div className="text-xs text-text-muted">Below investment grade</div>
            </div>
          </div>
        </div>
      </div>

      {/* Liquidity Metrics */}
      <div className="mb-6">
        <h4 className="text-md font-semibold text-text-primary mb-3">Liquidity Risk</h4>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
            <div className="text-sm text-text-secondary mb-1">Liquidity Score</div>
            <div className="text-xl font-bold text-text-primary">
              {formatNumber(liquidity_metrics.liquidity_score, 1)}/10
            </div>
          </div>
          <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
            <div className="text-sm text-text-secondary mb-1">Risk Level</div>
            <div className={`text-xl font-bold ${getLiquidityRiskColor(liquidity_metrics.liquidity_risk)}`}>
              {liquidity_metrics.liquidity_risk}
            </div>
          </div>
          <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
            <div className="text-sm text-text-secondary mb-1">Days to Liquidate</div>
            <div className="text-xl font-bold text-text-primary">
              {formatNumber(liquidity_metrics.days_to_liquidate, 1)} days
            </div>
          </div>
        </div>
      </div>

      {isExpanded && (
        <>
          {/* Stress Test Scenarios */}
          <div className="mb-6">
            <h4 className="text-md font-semibold text-text-primary mb-3">Stress Test Results</h4>
            <div className="space-y-2">
              {Object.entries(stress_scenarios).map(([scenario, impact]) => (
                <div key={scenario} className="flex items-center justify-between p-3 bg-card-background/30 rounded">
                  <span className="text-sm text-text-secondary capitalize">
                    {scenario.replace(/_/g, ' ')}
                  </span>
                  <div className={`text-lg font-semibold ${impact >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {impact >= 0 ? '+' : ''}{formatPercentage(impact)}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Risk Alerts */}
          <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
              <span className="text-sm font-semibold text-yellow-400">Risk Monitoring</span>
            </div>
            <p className="text-sm text-text-secondary">
              All risk metrics are within acceptable limits. Portfolio concentration and liquidity levels are being monitored.
            </p>
          </div>
        </>
      )}
    </div>
  )
}

export default RiskMetricsWidget
