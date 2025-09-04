import React, { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { TrendingUp, TrendingDown, AlertTriangle, Calculator, Target } from 'lucide-react'

interface PortfolioMetrics {
  totalValue: number
  duration: number
  convexity: number
  modifiedDuration: number
  pvbp: number // Price Value of Basis Point
}

interface ShockSimulatorProps {
  initialMetrics: PortfolioMetrics
  onMetricsChange?: (metrics: PortfolioMetrics) => void
  className?: string
}

interface ShockScenario {
  id: string
  name: string
  value: number // basis points
  description: string
  color: string
}

const SHOCK_SCENARIOS: ShockScenario[] = [
  {
    id: 'mild',
    name: 'Mild Shock',
    value: 25,
    description: 'RBI repo rate hike +25bps',
    color: 'bg-yellow-600'
  },
  {
    id: 'moderate',
    name: 'Moderate Shock',
    value: 50,
    description: 'RBI repo rate hike +50bps',
    color: 'bg-orange-600'
  },
  {
    id: 'severe',
    name: 'Severe Shock',
    value: 100,
    description: 'RBI repo rate hike +100bps',
    color: 'bg-red-600'
  },
  {
    id: 'crisis',
    name: 'Crisis Shock',
    value: 200,
    description: 'Crisis scenario +200bps',
    color: 'bg-red-800'
  }
]

const ShockSimulator: React.FC<ShockSimulatorProps> = ({
  initialMetrics,
  onMetricsChange,
  className = ''
}) => {
  const [selectedShock, setSelectedShock] = useState<number>(0)
  const [customShock, setCustomShock] = useState<string>('')
  const [isAnimating, setIsAnimating] = useState(false)

  // Calculate shocked metrics
  const shockedMetrics = useMemo(() => {
    const shockBps = selectedShock
    const shockDecimal = shockBps / 10000 // Convert bps to decimal

    // Duration impact: Price change = -Duration * Yield change
    const priceChange = -initialMetrics.duration * shockDecimal
    
    // Convexity adjustment: Additional price change = 0.5 * Convexity * (Yield change)^2
    const convexityAdjustment = 0.5 * initialMetrics.convexity * Math.pow(shockDecimal, 2)
    
    const totalPriceChange = priceChange + convexityAdjustment
    const newValue = initialMetrics.totalValue * (1 + totalPriceChange)
    
    // PVBP impact
    const pvbpImpact = initialMetrics.pvbp * shockBps

    return {
      totalValue: newValue,
      duration: initialMetrics.duration,
      convexity: initialMetrics.convexity,
      modifiedDuration: initialMetrics.modifiedDuration,
      pvbp: initialMetrics.pvbp,
      priceChange: totalPriceChange * 100,
      pvbpImpact,
      shockBps
    }
  }, [initialMetrics, selectedShock])

  // Handle shock selection
  const handleShockSelect = (shockValue: number) => {
    setIsAnimating(true)
    setSelectedShock(shockValue)
    setCustomShock('')
    
    setTimeout(() => {
      setIsAnimating(false)
      onMetricsChange?.(shockedMetrics)
    }, 300)
  }

  // Handle custom shock input
  const handleCustomShock = (value: string) => {
    setCustomShock(value)
    const numValue = parseFloat(value)
    if (!isNaN(numValue) && numValue >= 0 && numValue <= 500) {
      setSelectedShock(numValue)
      onMetricsChange?.(shockedMetrics)
    }
  }

  // Reset shock
  const resetShock = () => {
    setIsAnimating(true)
    setSelectedShock(0)
    setCustomShock('')
    
    setTimeout(() => {
      setIsAnimating(false)
      onMetricsChange?.(initialMetrics)
    }, 300)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={`bg-gray-900 rounded-lg overflow-hidden ${className}`}
    >
      {/* Header */}
      <div className="p-6 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-red-600 rounded-lg">
              <AlertTriangle className="h-6 w-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-white">Shock Simulator</h3>
              <p className="text-gray-400 text-sm">Simulate interest rate shocks on portfolio</p>
            </div>
          </div>
          
          {selectedShock > 0 && (
            <button
              onClick={resetShock}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-md text-sm text-white transition-colors"
            >
              Reset
            </button>
          )}
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Shock Scenarios */}
        <div>
          <h4 className="text-white font-medium mb-4 flex items-center">
            <Target className="h-4 w-4 mr-2" />
            Shock Scenarios
          </h4>
          <div className="grid grid-cols-2 gap-3">
            {SHOCK_SCENARIOS.map((scenario) => (
              <motion.button
                key={scenario.id}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => handleShockSelect(scenario.value)}
                className={`p-4 rounded-lg border-2 transition-all ${
                  selectedShock === scenario.value
                    ? `${scenario.color} border-transparent`
                    : 'bg-gray-800 border-gray-700 hover:border-gray-600'
                }`}
              >
                <div className="text-left">
                  <div className={`font-medium ${
                    selectedShock === scenario.value ? 'text-white' : 'text-white'
                  }`}>
                    {scenario.name}
                  </div>
                  <div className={`text-sm ${
                    selectedShock === scenario.value ? 'text-gray-200' : 'text-gray-400'
                  }`}>
                    +{scenario.value}bps
                  </div>
                  <div className={`text-xs mt-1 ${
                    selectedShock === scenario.value ? 'text-gray-300' : 'text-gray-500'
                  }`}>
                    {scenario.description}
                  </div>
                </div>
              </motion.button>
            ))}
          </div>
        </div>

        {/* Custom Shock Input */}
        <div>
          <h4 className="text-white font-medium mb-3 flex items-center">
            <Calculator className="h-4 w-4 mr-2" />
            Custom Shock
          </h4>
          <div className="flex items-center space-x-3">
            <input
              type="number"
              value={customShock}
              onChange={(e) => handleCustomShock(e.target.value)}
              placeholder="Enter basis points"
              min="0"
              max="500"
              className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <span className="text-gray-400 text-sm">bps</span>
          </div>
        </div>

        {/* Impact Analysis */}
        <AnimatePresence>
          {selectedShock > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
              className="space-y-4"
            >
              <h4 className="text-white font-medium">Impact Analysis</h4>
              
              {/* Portfolio Value Impact */}
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-gray-300">Portfolio Value Impact</span>
                  <div className={`flex items-center space-x-1 ${
                    shockedMetrics.priceChange < 0 ? 'text-red-400' : 'text-green-400'
                  }`}>
                    {shockedMetrics.priceChange < 0 ? (
                      <TrendingDown className="h-4 w-4" />
                    ) : (
                      <TrendingUp className="h-4 w-4" />
                    )}
                    <span className="font-medium">
                      {shockedMetrics.priceChange.toFixed(2)}%
                    </span>
                  </div>
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Initial Value:</span>
                    <span className="text-white">₹{initialMetrics.totalValue.toLocaleString()}L</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Shocked Value:</span>
                    <span className="text-white">₹{shockedMetrics.totalValue.toLocaleString()}L</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Absolute Change:</span>
                    <span className={`${
                      shockedMetrics.totalValue < initialMetrics.totalValue ? 'text-red-400' : 'text-green-400'
                    }`}>
                      ₹{(shockedMetrics.totalValue - initialMetrics.totalValue).toLocaleString()}L
                    </span>
                  </div>
                </div>
              </div>

              {/* Risk Metrics Impact */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-gray-800 rounded-lg p-4">
                  <div className="text-gray-300 text-sm mb-2">Duration Impact</div>
                  <div className="text-white font-medium">
                    {shockedMetrics.duration.toFixed(2)} years
                  </div>
                  <div className="text-gray-400 text-xs mt-1">
                    Modified: {shockedMetrics.modifiedDuration.toFixed(2)}
                  </div>
                </div>
                
                <div className="bg-gray-800 rounded-lg p-4">
                  <div className="text-gray-300 text-sm mb-2">PVBP Impact</div>
                  <div className="text-white font-medium">
                    ₹{shockedMetrics.pvbpImpact.toFixed(2)}L
                  </div>
                  <div className="text-gray-400 text-xs mt-1">
                    Per basis point
                  </div>
                </div>
              </div>

              {/* Convexity Adjustment */}
              <div className="bg-gray-800 rounded-lg p-4">
                <div className="text-gray-300 text-sm mb-2">Convexity Adjustment</div>
                <div className="text-white font-medium">
                  {shockedMetrics.convexity.toFixed(2)}
                </div>
                <div className="text-gray-400 text-xs mt-1">
                  Reduces duration impact by {Math.abs(shockedMetrics.priceChange - (-shockedMetrics.duration * shockedMetrics.shockBps / 10000) * 100).toFixed(2)}%
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Initial Metrics */}
        <div className="border-t border-gray-800 pt-4">
          <h4 className="text-white font-medium mb-3">Current Portfolio Metrics</h4>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="text-center">
              <div className="text-gray-400">Total Value</div>
              <div className="text-white font-medium">₹{initialMetrics.totalValue.toLocaleString()}L</div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">Duration</div>
              <div className="text-white font-medium">{initialMetrics.duration.toFixed(2)}Y</div>
            </div>
            <div className="text-center">
              <div className="text-gray-400">Convexity</div>
              <div className="text-white font-medium">{initialMetrics.convexity.toFixed(2)}</div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default ShockSimulator
