import React, { useState, useEffect } from 'react'
import { Widget } from '@/types'
import { BarChart3, TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { apiService, MarketStatus, formatNumber, formatPercentage } from '@/services/api'

interface MarketOverviewWidgetProps {
  widget: Widget
  userRole: string
  isExpanded: boolean
}

interface MarketData {
  region: string
  bonds: BondData[]
  totalVolume: number
  changePercent: number
}

interface BondData {
  name: string
  price: number
  change: number
  changePercent: number
  volume: number
  yield: number
}

const MarketOverviewWidget: React.FC<MarketOverviewWidgetProps> = ({
  widget,
  userRole,
  isExpanded
}) => {
  const [marketData, setMarketData] = useState<MarketData[]>([])
  const [realMarketData, setRealMarketData] = useState<MarketStatus | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const fetchMarketData = async () => {
    try {
      setError(null)
      const data = await apiService.getMarketStatus()
      setRealMarketData(data)
      
      // Convert real market data to display format
      const convertedData: MarketData[] = [
        {
          region: 'India - NSE',
          totalVolume: data.markets.nse.total_volume * 10000000, // Convert Cr to actual number
          changePercent: 2.3, // Mock for now
          bonds: [
            { 
              name: 'Corporate Bonds', 
              price: 98.45, 
              change: 0.15, 
              changePercent: 0.15, 
              volume: data.markets.nse.total_volume * 10000000, 
              yield: data.markets.nse.avg_yield 
            }
          ]
        },
        {
          region: 'India - BSE',
          totalVolume: data.markets.bse.total_volume * 10000000,
          changePercent: 1.8,
          bonds: [
            { 
              name: 'Corporate Bonds', 
              price: 99.12, 
              change: -0.08, 
              changePercent: -0.08, 
              volume: data.markets.bse.total_volume * 10000000, 
              yield: data.markets.bse.avg_yield 
            }
          ]
        },
        {
          region: 'India - RBI',
          totalVolume: data.markets.rbi.total_volume * 10000000,
          changePercent: 0.5,
          bonds: [
            { 
              name: 'Government Securities', 
              price: 95.67, 
              change: 0.32, 
              changePercent: 0.34, 
              volume: data.markets.rbi.total_volume * 10000000, 
              yield: data.markets.rbi.avg_yield 
            }
          ]
        }
      ]
      setMarketData(convertedData)
    } catch (err) {
      console.error('Error fetching market data:', err)
      setError('Failed to fetch market data')
      // Fallback to mock data
      const mockData: MarketData[] = [
      {
        region: 'India',
        totalVolume: 1250000000,
        changePercent: 2.3,
        bonds: [
          { name: '10Y G-Sec', price: 98.45, change: 0.15, changePercent: 0.15, volume: 450000000, yield: 6.85 },
          { name: '5Y G-Sec', price: 99.12, change: -0.08, changePercent: -0.08, volume: 320000000, yield: 6.45 },
          { name: '30Y G-Sec', price: 95.67, change: 0.32, changePercent: 0.34, volume: 180000000, yield: 7.25 }
        ]
      },
      {
        region: 'US',
        totalVolume: 8900000000,
        changePercent: -1.2,
        bonds: [
          { name: '10Y Treasury', price: 101.23, change: -0.45, changePercent: -0.44, volume: 3200000000, yield: 4.25 },
          { name: '5Y Treasury', price: 102.67, change: -0.23, changePercent: -0.22, volume: 2800000000, yield: 4.05 },
          { name: '30Y Treasury', price: 98.45, change: -0.67, changePercent: -0.68, volume: 1200000000, yield: 4.85 }
        ]
      },
      {
        region: 'EU',
        totalVolume: 3450000000,
        changePercent: 0.8,
        bonds: [
          { name: '10Y Bund', price: 103.45, change: 0.12, changePercent: 0.12, volume: 1200000000, yield: 2.85 },
          { name: '5Y Bund', price: 104.23, change: 0.08, changePercent: 0.08, volume: 980000000, yield: 2.45 },
          { name: '30Y Bund', price: 99.87, change: 0.25, changePercent: 0.25, volume: 450000000, yield: 3.25 }
        ]
      },
      {
        region: 'Asia',
        totalVolume: 890000000,
        changePercent: 1.5,
        bonds: [
          { name: '10Y JGB', price: 102.34, change: 0.18, changePercent: 0.18, volume: 320000000, yield: 0.85 },
          { name: '5Y JGB', price: 103.12, change: 0.12, changePercent: 0.12, volume: 280000000, yield: 0.45 },
          { name: '30Y JGB', price: 98.76, change: 0.28, changePercent: 0.28, volume: 120000000, yield: 1.25 }
        ]
      }
    ]
    setMarketData(mockData)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchMarketData()
    
    // Set up periodic refresh every 30 seconds
    const interval = setInterval(fetchMarketData, 30000)
    return () => clearInterval(interval)
  }, [])

  const getChangeIcon = (change: number) => {
    if (change > 0) return <TrendingUp className="h-4 w-4 text-accent-green" />
    if (change < 0) return <TrendingDown className="h-4 w-4 text-accent-red" />
    return <Minus className="h-4 w-4 text-text-muted" />
  }

  const getChangeColor = (change: number) => {
    if (change > 0) return 'text-accent-green'
    if (change < 0) return 'text-accent-red'
    return 'text-text-muted'
  }

  const formatVolume = (volume: number) => {
    if (volume >= 1000000000) return `${(volume / 1000000000).toFixed(1)}B`
    if (volume >= 1000000) return `${(volume / 1000000).toFixed(1)}M`
    if (volume >= 1000) return `${(volume / 1000).toFixed(1)}K`
    return volume.toString()
  }

  if (isLoading) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <div className="spinner h-6 w-6 mx-auto mb-2"></div>
          <p className="text-sm text-text-secondary">Loading market data...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full space-y-4">
      {/* Market summary */}
      <div className="grid grid-cols-2 gap-4">
        <div className="text-center">
          <p className="text-xs text-text-secondary mb-1">Total Volume</p>
          <p className="text-lg font-semibold text-text-primary">
            ${formatVolume(marketData.reduce((sum, region) => sum + region.totalVolume, 0))}
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-text-secondary mb-1">Active Markets</p>
          <p className="text-lg font-semibold text-text-primary">{marketData.length}</p>
        </div>
      </div>

      {/* Region tabs */}
      <div className="flex space-x-1">
        {marketData.map((region) => (
          <button
            key={region.region}
            onClick={() => setSelectedRegion(selectedRegion === region.region ? null : region.region)}
            className={`flex-1 px-2 py-1 text-xs rounded ${
              selectedRegion === region.region
                ? 'bg-accent-blue text-white'
                : 'bg-background-tertiary text-text-secondary hover:text-text-primary'
            }`}
          >
            {region.region}
          </button>
        ))}
      </div>

      {/* Market data */}
      <div className="space-y-3">
        {marketData.map((region) => (
          <div
            key={region.region}
            className={`border rounded-lg p-3 transition-all duration-200 ${
              selectedRegion === region.region
                ? 'border-accent-blue bg-accent-blue/5'
                : 'border-border hover:border-border-secondary'
            }`}
          >
            {/* Region header */}
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-medium text-text-primary">{region.region}</h4>
              <div className="flex items-center space-x-2">
                <span className="text-xs text-text-secondary">
                  Vol: ${formatVolume(region.totalVolume)}
                </span>
                <div className={`flex items-center space-x-1 ${getChangeColor(region.changePercent)}`}>
                  {getChangeIcon(region.changePercent)}
                  <span className="text-xs font-medium">
                    {region.changePercent > 0 ? '+' : ''}{region.changePercent.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* Bond list */}
            <div className="space-y-2">
              {region.bonds.map((bond) => (
                <div key={bond.name} className="flex items-center justify-between text-sm">
                  <div className="flex-1 min-w-0">
                    <p className="text-text-primary truncate">{bond.name}</p>
                    <p className="text-xs text-text-secondary">
                      Yield: {bond.yield.toFixed(2)}%
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-text-primary font-mono">{bond.price.toFixed(2)}</p>
                    <div className={`flex items-center space-x-1 ${getChangeColor(bond.changePercent)}`}>
                      {getChangeIcon(bond.changePercent)}
                      <span className="text-xs">
                        {bond.changePercent > 0 ? '+' : ''}{bond.changePercent.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Expanded view content */}
      {isExpanded && (
        <div className="border-t border-border pt-4">
          <h5 className="text-sm font-medium text-text-primary mb-3">Market Sentiment</h5>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="h-16 w-16 mx-auto rounded-full bg-accent-green/20 flex items-center justify-center mb-2">
                <BarChart3 className="h-8 w-8 text-accent-green" />
              </div>
              <p className="text-xs text-text-secondary">Bullish</p>
              <p className="text-sm font-semibold text-text-primary">65%</p>
            </div>
            <div>
              <div className="h-16 w-16 mx-auto rounded-full bg-accent-red/20 flex items-center justify-center mb-2">
                <BarChart3 className="h-8 w-8 text-accent-red" />
              </div>
              <p className="text-xs text-text-secondary">Bearish</p>
              <p className="text-sm font-semibold text-text-primary">25%</p>
            </div>
            <div>
              <div className="h-16 w-16 mx-auto rounded-full bg-text-muted/20 flex items-center justify-center mb-2">
                <BarChart3 className="h-8 w-8 text-text-muted" />
              </div>
              <p className="text-xs text-text-secondary">Neutral</p>
              <p className="text-sm font-semibold text-text-primary">10%</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default MarketOverviewWidget
