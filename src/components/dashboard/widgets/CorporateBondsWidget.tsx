import React, { useEffect, useState } from 'react'
import { Widget } from '@/types'
import { apiService, formatCurrency, formatPercentage, formatNumber } from '@/services/api'

interface CorporateBond {
  isin: string
  descriptor: string
  issuer_name: string
  sector: string
  bond_type: string
  coupon_rate: number | null
  maturity_date: string | null
  weighted_avg_price: number
  last_trade_price: number
  weighted_avg_yield: number
  last_trade_yield: number
  value_lakhs: number
  num_trades: number
  face_value: number | null
}

interface CorporateBondsData {
  timestamp: string
  total_bonds: number
  bonds: CorporateBond[]
  market_summary: {
    total_bonds: number
    total_value_lakhs: number
    total_trades: number
    average_yield: number
    sector_breakdown: {
      counts: Record<string, number>
      values: Record<string, number>
    }
  }
  filters_applied: {
    sector: string | null
    min_yield: number | null
    max_yield: number | null
    sort_by: string
    limit: number
  }
}

interface CorporateBondsWidgetProps {
  widget: Widget
  userRole: string
  isExpanded: boolean
}

const CorporateBondsWidget: React.FC<CorporateBondsWidgetProps> = ({ widget, userRole, isExpanded }) => {
  const [bondsData, setBondsData] = useState<CorporateBondsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [selectedSector, setSelectedSector] = useState<string | null>(null)
  const [sortBy, setSortBy] = useState('volume')

  const fetchCorporateBonds = async () => {
    try {
      setError(null)
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/dashboard/corporate-bonds?${new URLSearchParams({
        ...(selectedSector && { sector: selectedSector }),
        sort_by: sortBy,
        limit: isExpanded ? '50' : '20'
      })}`)
      
      if (!response.ok) {
        throw new Error('Failed to fetch corporate bonds data')
      }
      
      const data = await response.json()
      setBondsData(data)
      setLastUpdated(new Date())
    } catch (err) {
      console.error('Error fetching corporate bonds:', err)
      setError('Failed to fetch corporate bonds data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchCorporateBonds()
    
    // Set up periodic refresh every 60 seconds
    const interval = setInterval(fetchCorporateBonds, 60000)
    return () => clearInterval(interval)
  }, [selectedSector, sortBy, isExpanded])

  const getSectorColor = (sector: string) => {
    const colors: Record<string, string> = {
      'Financial Services': 'text-blue-400',
      'Energy': 'text-green-400',
      'Infrastructure': 'text-purple-400',
      'Automotive': 'text-yellow-400',
      'Metals': 'text-red-400',
      'Corporate': 'text-gray-400'
    }
    return colors[sector] || 'text-gray-400'
  }

  const formatMaturityDate = (dateStr: string | null) => {
    if (!dateStr) return 'N/A'
    try {
      return new Date(dateStr).toLocaleDateString('en-IN', {
        day: '2-digit',
        month: 'short',
        year: '2-digit'
      })
    } catch {
      return 'N/A'
    }
  }

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
          onClick={fetchCorporateBonds}
          className="mt-2 px-3 py-1 text-xs bg-primary-600 hover:bg-primary-700 rounded transition-colors"
        >
          Retry
        </button>
      </div>
    )
  }

  if (!bondsData) {
    return (
      <div className="h-full flex items-center justify-center text-text-secondary">
        <p>No corporate bonds data available</p>
      </div>
    )
  }

  const { bonds, market_summary } = bondsData
  const sectors = Object.keys(market_summary.sector_breakdown.counts)

  return (
    <div className="h-full p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-text-primary mb-1">Corporate Bonds</h3>
          <p className="text-sm text-text-secondary">
            {bondsData.total_bonds} bonds • ₹{formatNumber(market_summary.total_value_lakhs)} Lakhs
          </p>
        </div>
        <div className="text-right">
          <div className="text-sm text-text-secondary">Last Updated</div>
          <div className="text-xs text-text-muted">
            {lastUpdated?.toLocaleTimeString() || 'Loading...'}
          </div>
        </div>
      </div>

      {/* Market Summary */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
          <div className="text-sm text-text-secondary mb-1">Total Volume</div>
          <div className="text-xl font-bold text-text-primary">
            ₹{formatNumber(market_summary.total_value_lakhs)} L
          </div>
        </div>
        <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
          <div className="text-sm text-text-secondary mb-1">Total Trades</div>
          <div className="text-xl font-bold text-text-primary">
            {formatNumber(market_summary.total_trades, 0)}
          </div>
        </div>
        <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
          <div className="text-sm text-text-secondary mb-1">Avg Yield</div>
          <div className="text-xl font-bold text-text-primary">
            {formatPercentage(market_summary.average_yield)}
          </div>
        </div>
        <div className="bg-card-background/50 rounded-lg p-4 border border-border/50">
          <div className="text-sm text-text-secondary mb-1">Active Bonds</div>
          <div className="text-xl font-bold text-text-primary">
            {market_summary.total_bonds}
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-4 mb-6">
        {/* Sector Filter */}
        <div className="flex gap-2">
          <button
            onClick={() => setSelectedSector(null)}
            className={`px-3 py-1 text-sm rounded transition-colors ${
              selectedSector === null
                ? 'bg-primary-600 text-white'
                : 'bg-card-background/50 text-text-secondary hover:text-text-primary'
            }`}
          >
            All Sectors
          </button>
          {sectors.map((sector) => (
            <button
              key={sector}
              onClick={() => setSelectedSector(sector)}
              className={`px-3 py-1 text-sm rounded transition-colors ${
                selectedSector === sector
                  ? 'bg-primary-600 text-white'
                  : 'bg-card-background/50 text-text-secondary hover:text-text-primary'
              }`}
            >
              {sector} ({market_summary.sector_breakdown.counts[sector]})
            </button>
          ))}
        </div>

        {/* Sort Filter */}
        <div className="flex gap-2">
          {['volume', 'yield', 'price'].map((sort) => (
            <button
              key={sort}
              onClick={() => setSortBy(sort)}
              className={`px-3 py-1 text-sm rounded transition-colors capitalize ${
                sortBy === sort
                  ? 'bg-secondary-600 text-white'
                  : 'bg-card-background/50 text-text-secondary hover:text-text-primary'
              }`}
            >
              {sort}
            </button>
          ))}
        </div>
      </div>

      {/* Bonds Table */}
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {bonds.map((bond) => (
          <div 
            key={bond.isin} 
            className="flex items-center justify-between p-4 bg-card-background/30 rounded-lg hover:bg-card-background/50 transition-colors"
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <div className="text-sm font-medium text-text-primary truncate">
                  {bond.issuer_name || bond.descriptor}
                </div>
                <span className={`text-xs px-2 py-1 rounded ${getSectorColor(bond.sector)} bg-current bg-opacity-20`}>
                  {bond.sector}
                </span>
              </div>
              <div className="text-xs text-text-secondary">
                {bond.isin} • {bond.bond_type}
                {bond.coupon_rate && ` • ${formatPercentage(bond.coupon_rate)} coupon`}
              </div>
              {bond.maturity_date && (
                <div className="text-xs text-text-muted">
                  Maturity: {formatMaturityDate(bond.maturity_date)}
                </div>
              )}
            </div>
            
            <div className="text-right">
              <div className="text-sm font-medium text-text-primary">
                ₹{formatNumber(bond.last_trade_price, 4)}
              </div>
              <div className="text-sm text-primary-400">
                {formatPercentage(bond.last_trade_yield)} YTM
              </div>
              <div className="text-xs text-text-secondary">
                ₹{formatNumber(bond.value_lakhs)}L • {bond.num_trades} trades
              </div>
            </div>
          </div>
        ))}
      </div>

      {isExpanded && (
        <>
          {/* Sector Breakdown */}
          <div className="mt-6">
            <h4 className="text-md font-semibold text-text-primary mb-3">Sector Breakdown</h4>
            <div className="space-y-2">
              {Object.entries(market_summary.sector_breakdown.values).map(([sector, value]) => {
                const percentage = (value / market_summary.total_value_lakhs) * 100
                return (
                  <div key={sector} className="flex items-center justify-between p-2 bg-card-background/20 rounded">
                    <span className={`text-sm ${getSectorColor(sector)}`}>{sector}</span>
                    <div className="text-right">
                      <div className="text-sm font-medium text-text-primary">
                        ₹{formatNumber(value)}L
                      </div>
                      <div className="text-xs text-text-secondary">
                        {formatPercentage(percentage)}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Market Insights */}
          <div className="mt-6 bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <span className="text-sm font-semibold text-blue-400">Market Insights</span>
            </div>
            <p className="text-sm text-text-secondary">
              Corporate bonds market shows {market_summary.total_trades} trades with average yield of {formatPercentage(market_summary.average_yield)}. 
              {market_summary.sector_breakdown.counts['Financial Services'] ? 
                ` Financial Services leads with ${market_summary.sector_breakdown.counts['Financial Services']} bonds.` : 
                ''
              }
            </p>
          </div>
        </>
      )}
    </div>
  )
}

export default CorporateBondsWidget
