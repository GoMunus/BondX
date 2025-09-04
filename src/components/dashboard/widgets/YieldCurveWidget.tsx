import React, { useEffect, useState } from 'react'
import { Widget } from '@/types'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { apiService, YieldCurveData } from '@/services/api'

interface YieldCurveWidgetProps {
  widget: Widget
  userRole: string
  isExpanded: boolean
}

const YieldCurveWidget: React.FC<YieldCurveWidgetProps> = ({ widget, userRole, isExpanded }) => {
  const [yieldData, setYieldData] = useState<YieldCurveData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedCurrency, setSelectedCurrency] = useState('INR')
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)

  const fetchYieldCurveData = async (currency: string = 'INR') => {
    try {
      setError(null)
      const data = await apiService.getYieldCurve(currency)
      setYieldData(data)
      setLastUpdated(new Date())
    } catch (err) {
      console.error('Error fetching yield curve data:', err)
      setError('Failed to fetch yield curve data')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchYieldCurveData(selectedCurrency)
    
    // Set up periodic refresh every 60 seconds
    const interval = setInterval(() => fetchYieldCurveData(selectedCurrency), 60000)
    return () => clearInterval(interval)
  }, [selectedCurrency])

  const handleCurrencyChange = (currency: string) => {
    setSelectedCurrency(currency)
    setLoading(true)
    fetchYieldCurveData(currency)
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
          onClick={() => fetchYieldCurveData(selectedCurrency)}
          className="mt-2 px-3 py-1 text-xs bg-primary-600 hover:bg-primary-700 rounded transition-colors"
        >
          Retry
        </button>
      </div>
    )
  }

  if (!yieldData) {
    return (
      <div className="h-full flex items-center justify-center text-text-secondary">
        <p>No yield curve data available</p>
      </div>
    )
  }

  // Custom tooltip for the chart
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-card-background border border-border rounded-lg p-3 shadow-lg">
          <p className="text-sm font-medium text-text-primary">
            Tenor: {label}
          </p>
          <p className="text-sm text-text-secondary">
            Yield: <span className="text-primary-400">{payload[0].value.toFixed(2)}%</span>
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <div className="h-full p-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-text-primary mb-1">Yield Curve</h3>
          <p className="text-sm text-text-secondary">
            {yieldData.currency} Government Securities
          </p>
        </div>
        <div className="text-right">
          <div className="text-sm text-text-secondary">Last Updated</div>
          <div className="text-xs text-text-muted">
            {lastUpdated?.toLocaleTimeString() || 'Loading...'}
          </div>
        </div>
      </div>

      {/* Currency Selector */}
      <div className="flex gap-2 mb-6">
        {['INR', 'USD', 'EUR'].map((currency) => (
          <button
            key={currency}
            onClick={() => handleCurrencyChange(currency)}
            className={`px-3 py-1 text-sm rounded transition-colors ${
              selectedCurrency === currency
                ? 'bg-primary-600 text-white'
                : 'bg-card-background/50 text-text-secondary hover:text-text-primary'
            }`}
          >
            {currency}
          </button>
        ))}
      </div>

      {/* Yield Curve Chart */}
      <div className="mb-6" style={{ height: isExpanded ? '400px' : '200px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={yieldData.data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <XAxis 
              dataKey="tenor_label" 
              stroke="#9CA3AF"
              fontSize={12}
              tickLine={false}
            />
            <YAxis 
              stroke="#9CA3AF"
              fontSize={12}
              tickLine={false}
              domain={['dataMin - 0.1', 'dataMax + 0.1']}
              tickFormatter={(value) => `${value.toFixed(1)}%`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line 
              type="monotone" 
              dataKey="rate" 
              stroke="#3B82F6" 
              strokeWidth={2}
              dot={{ fill: '#3B82F6', strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, stroke: '#3B82F6', strokeWidth: 2, fill: '#1E40AF' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Yield Table */}
      <div className="mb-6">
        <h4 className="text-md font-semibold text-text-primary mb-3">Current Yields</h4>
        <div className="grid grid-cols-3 lg:grid-cols-6 gap-2">
          {yieldData.data.map((point) => (
            <div key={point.tenor} className="bg-card-background/30 rounded p-2 text-center">
              <div className="text-xs text-text-secondary mb-1">{point.tenor_label}</div>
              <div className="text-sm font-semibold text-text-primary">
                {point.rate.toFixed(2)}%
              </div>
            </div>
          ))}
        </div>
      </div>

      {isExpanded && (
        <>
          {/* Curve Statistics */}
          <div className="mb-6">
            <h4 className="text-md font-semibold text-text-primary mb-3">Curve Statistics</h4>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-card-background/30 rounded p-3">
                <div className="text-xs text-text-secondary mb-1">2Y-10Y Spread</div>
                <div className="text-lg font-semibold text-text-primary">
                  {yieldData.data.find(p => p.tenor === 10) && yieldData.data.find(p => p.tenor === 2) 
                    ? ((yieldData.data.find(p => p.tenor === 10)?.rate || 0) - (yieldData.data.find(p => p.tenor === 2)?.rate || 0)).toFixed(2)
                    : 'N/A'}%
                </div>
              </div>
              <div className="bg-card-background/30 rounded p-3">
                <div className="text-xs text-text-secondary mb-1">10Y-30Y Spread</div>
                <div className="text-lg font-semibold text-text-primary">
                  {yieldData.data.find(p => p.tenor === 30) && yieldData.data.find(p => p.tenor === 10) 
                    ? ((yieldData.data.find(p => p.tenor === 30)?.rate || 0) - (yieldData.data.find(p => p.tenor === 10)?.rate || 0)).toFixed(2)
                    : 'N/A'}%
                </div>
              </div>
              <div className="bg-card-background/30 rounded p-3">
                <div className="text-xs text-text-secondary mb-1">Average Yield</div>
                <div className="text-lg font-semibold text-text-primary">
                  {(yieldData.data.reduce((sum, point) => sum + point.rate, 0) / yieldData.data.length).toFixed(2)}%
                </div>
              </div>
              <div className="bg-card-background/30 rounded p-3">
                <div className="text-xs text-text-secondary mb-1">Curve Slope</div>
                <div className="text-lg font-semibold text-text-primary">
                  {yieldData.data.length > 1 
                    ? ((yieldData.data[yieldData.data.length - 1].rate - yieldData.data[0].rate) / 
                       (yieldData.data[yieldData.data.length - 1].tenor - yieldData.data[0].tenor) * 100).toFixed(2)
                    : 'N/A'} bps/yr
                </div>
              </div>
            </div>
          </div>

          {/* Curve Analysis */}
          <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <span className="text-sm font-semibold text-blue-400">Curve Analysis</span>
            </div>
            <p className="text-sm text-text-secondary">
              The {yieldData.currency} yield curve shows a {
                yieldData.data.length > 1 && yieldData.data[yieldData.data.length - 1].rate > yieldData.data[0].rate 
                  ? 'normal upward' 
                  : 'inverted'
              } slope pattern. Current 10-year yield is at {
                yieldData.data.find(p => p.tenor === 10)?.rate.toFixed(2) || 'N/A'
              }%, indicating {
                (yieldData.data.find(p => p.tenor === 10)?.rate || 0) > 6 
                  ? 'elevated interest rate environment' 
                  : 'moderate interest rate levels'
              }.
            </p>
          </div>
        </>
      )}
    </div>
  )
}

export default YieldCurveWidget
