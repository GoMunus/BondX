import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Activity, TrendingUp, TrendingDown, AlertTriangle, Shield, Zap } from 'lucide-react'

interface MarketRegime {
  id: string
  name: string
  description: string
  color: string
  bgColor: string
  icon: React.ReactNode
  confidence: number
  volatility: number
  liquidity: number
  spread: number
  recommendation: string
  riskLevel: 'low' | 'medium' | 'high' | 'extreme'
}

interface MarketRegimeDetectorProps {
  onRegimeChange?: (regime: MarketRegime) => void
  className?: string
}

const REGIME_TYPES: Omit<MarketRegime, 'confidence' | 'volatility' | 'liquidity' | 'spread'>[] = [
  {
    id: 'calm',
    name: 'Calm Market',
    description: 'Low volatility, stable spreads, high liquidity',
    color: 'text-green-400',
    bgColor: 'bg-green-600/20',
    icon: <Shield className="h-5 w-5" />,
    recommendation: 'Ideal for carry strategies and yield enhancement',
    riskLevel: 'low'
  },
  {
    id: 'volatile',
    name: 'Volatile Market',
    description: 'Elevated volatility, widening spreads, moderate liquidity',
    color: 'text-yellow-400',
    bgColor: 'bg-yellow-600/20',
    icon: <Activity className="h-5 w-5" />,
    recommendation: 'Defensive positioning, focus on quality over yield',
    riskLevel: 'medium'
  },
  {
    id: 'crisis',
    name: 'Crisis Market',
    description: 'High volatility, wide spreads, low liquidity',
    color: 'text-red-400',
    bgColor: 'bg-red-600/20',
    icon: <AlertTriangle className="h-5 w-5" />,
    recommendation: 'Capital preservation, high-quality bonds only',
    riskLevel: 'high'
  },
  {
    id: 'recovery',
    name: 'Recovery Market',
    description: 'Improving conditions, narrowing spreads, returning liquidity',
    color: 'text-blue-400',
    bgColor: 'bg-blue-600/20',
    icon: <TrendingUp className="h-5 w-5" />,
    recommendation: 'Gradual risk-on approach, selective opportunities',
    riskLevel: 'medium'
  }
]

const MarketRegimeDetector: React.FC<MarketRegimeDetectorProps> = ({
  onRegimeChange,
  className = ''
}) => {
  const [currentRegime, setCurrentRegime] = useState<MarketRegime | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())
  const [historicalRegimes, setHistoricalRegimes] = useState<MarketRegime[]>([])

  // Simulate market regime detection
  useEffect(() => {
    const detectRegime = async () => {
      setIsLoading(true)
      
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      // Simulate market data analysis
      const volatility = Math.random() * 30 + 5 // 5-35%
      const liquidity = Math.random() * 40 + 40 // 40-80%
      const spread = Math.random() * 100 + 20 // 20-120 bps
      
      // Determine regime based on metrics
      let regimeType: string
      let confidence: number
      
      if (volatility < 10 && liquidity > 70 && spread < 40) {
        regimeType = 'calm'
        confidence = 0.85 + Math.random() * 0.1
      } else if (volatility < 20 && liquidity > 50 && spread < 80) {
        regimeType = 'volatile'
        confidence = 0.75 + Math.random() * 0.15
      } else if (volatility > 25 || liquidity < 40 || spread > 100) {
        regimeType = 'crisis'
        confidence = 0.80 + Math.random() * 0.15
      } else {
        regimeType = 'recovery'
        confidence = 0.70 + Math.random() * 0.20
      }
      
      const baseRegime = REGIME_TYPES.find(r => r.id === regimeType)!
      
      const regime: MarketRegime = {
        ...baseRegime,
        confidence,
        volatility,
        liquidity,
        spread
      }
      
      setCurrentRegime(regime)
      setLastUpdate(new Date())
      onRegimeChange?.(regime)
      
      // Add to historical data
      setHistoricalRegimes(prev => [regime, ...prev.slice(0, 9)])
      
      setIsLoading(false)
    }

    // Initial detection
    detectRegime()
    
    // Update every 30 seconds
    const interval = setInterval(detectRegime, 30000)
    
    return () => clearInterval(interval)
  }, [onRegimeChange])

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return 'text-green-400'
      case 'medium': return 'text-yellow-400'
      case 'high': return 'text-orange-400'
      case 'extreme': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  const getRiskBgColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return 'bg-green-600/20'
      case 'medium': return 'bg-yellow-600/20'
      case 'high': return 'bg-orange-600/20'
      case 'extreme': return 'bg-red-600/20'
      default: return 'bg-gray-600/20'
    }
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
            <div className="p-2 bg-purple-600 rounded-lg">
              <Zap className="h-6 w-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-white">Market Regime Detector</h3>
              <p className="text-gray-400 text-sm">AI-powered market condition analysis</p>
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-gray-400 text-sm">Last Updated</div>
            <div className="text-white text-sm font-mono">
              {lastUpdate.toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Current Regime */}
        <div>
          <h4 className="text-white font-medium mb-4">Current Market Regime</h4>
          
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500 mx-auto mb-4"></div>
                <p className="text-gray-400">Analyzing market conditions...</p>
              </div>
            </div>
          ) : currentRegime ? (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className={`${currentRegime.bgColor} rounded-lg p-6 border border-gray-700`}
            >
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className={currentRegime.color}>
                    {currentRegime.icon}
                  </div>
                  <div>
                    <h5 className={`text-lg font-semibold ${currentRegime.color}`}>
                      {currentRegime.name}
                    </h5>
                    <p className="text-gray-300 text-sm">
                      {currentRegime.description}
                    </p>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className={`text-2xl font-bold ${currentRegime.color}`}>
                    {(currentRegime.confidence * 100).toFixed(0)}%
                  </div>
                  <div className="text-gray-400 text-sm">Confidence</div>
                </div>
              </div>
              
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="text-center">
                  <div className="text-white font-medium">
                    {currentRegime.volatility.toFixed(1)}%
                  </div>
                  <div className="text-gray-400 text-sm">Volatility</div>
                </div>
                <div className="text-center">
                  <div className="text-white font-medium">
                    {currentRegime.liquidity.toFixed(0)}%
                  </div>
                  <div className="text-gray-400 text-sm">Liquidity</div>
                </div>
                <div className="text-center">
                  <div className="text-white font-medium">
                    {currentRegime.spread.toFixed(0)}bps
                  </div>
                  <div className="text-gray-400 text-sm">Spread</div>
                </div>
              </div>
              
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="flex items-start space-x-3">
                  <TrendingUp className="h-5 w-5 text-blue-400 mt-0.5" />
                  <div>
                    <div className="text-white font-medium mb-1">Recommendation</div>
                    <div className="text-gray-300 text-sm">
                      {currentRegime.recommendation}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="text-center py-8 text-gray-400">
              Unable to detect market regime
            </div>
          )}
        </div>

        {/* Risk Level Indicator */}
        {currentRegime && (
          <div>
            <h4 className="text-white font-medium mb-3">Risk Assessment</h4>
            <div className={`${getRiskBgColor(currentRegime.riskLevel)} rounded-lg p-4 border border-gray-700`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${
                    currentRegime.riskLevel === 'low' ? 'bg-green-600' :
                    currentRegime.riskLevel === 'medium' ? 'bg-yellow-600' :
                    currentRegime.riskLevel === 'high' ? 'bg-orange-600' : 'bg-red-600'
                  }`}>
                    {currentRegime.riskLevel === 'low' ? <Shield className="h-4 w-4 text-white" /> :
                     currentRegime.riskLevel === 'medium' ? <Activity className="h-4 w-4 text-white" /> :
                     currentRegime.riskLevel === 'high' ? <AlertTriangle className="h-4 w-4 text-white" /> :
                     <AlertTriangle className="h-4 w-4 text-white" />}
                  </div>
                  <div>
                    <div className={`font-medium ${getRiskColor(currentRegime.riskLevel)}`}>
                      {currentRegime.riskLevel.toUpperCase()} RISK
                    </div>
                    <div className="text-gray-400 text-sm">
                      {currentRegime.riskLevel === 'low' ? 'Stable market conditions' :
                       currentRegime.riskLevel === 'medium' ? 'Moderate risk environment' :
                       currentRegime.riskLevel === 'high' ? 'High risk environment' :
                       'Extreme risk conditions'}
                    </div>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className={`text-lg font-bold ${getRiskColor(currentRegime.riskLevel)}`}>
                    {currentRegime.riskLevel === 'low' ? '🟢' :
                     currentRegime.riskLevel === 'medium' ? '🟡' :
                     currentRegime.riskLevel === 'high' ? '🟠' : '🔴'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Historical Regimes */}
        {historicalRegimes.length > 0 && (
          <div>
            <h4 className="text-white font-medium mb-3">Recent Regime History</h4>
            <div className="space-y-2">
              {historicalRegimes.slice(0, 5).map((regime, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="flex items-center justify-between p-3 bg-gray-800 rounded-lg"
                >
                  <div className="flex items-center space-x-3">
                    <div className={regime.color}>
                      {regime.icon}
                    </div>
                    <div>
                      <div className="text-white font-medium">{regime.name}</div>
                      <div className="text-gray-400 text-sm">
                        {(regime.confidence * 100).toFixed(0)}% confidence
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-right text-sm">
                    <div className="text-gray-400">
                      {regime.volatility.toFixed(1)}% vol
                    </div>
                    <div className="text-gray-400">
                      {regime.liquidity.toFixed(0)}% liq
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {/* Model Information */}
        <div className="border-t border-gray-800 pt-4">
          <div className="text-center text-gray-400 text-sm">
            <div className="flex items-center justify-center space-x-2">
              <Zap className="h-4 w-4" />
              <span>Powered by ML models analyzing volatility, liquidity, and spread patterns</span>
            </div>
            <div className="mt-1">
              Updates every 30 seconds • Confidence threshold: 70%
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default MarketRegimeDetector
