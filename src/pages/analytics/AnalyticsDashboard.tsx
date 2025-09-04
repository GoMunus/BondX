import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { BarChart3, Brain, Zap, Target, TrendingUp, Activity } from 'lucide-react'

// Import our new components
import YieldCurve3D from '@/components/analytics/YieldCurve3D'
import ShockSimulator from '@/components/analytics/ShockSimulator'
import LiquidityHeatmap from '@/components/analytics/LiquidityHeatmap'
import LiquidityForecasting from '@/components/analytics/LiquidityForecasting'
import SpreadForecasting from '@/components/analytics/SpreadForecasting'
import PortfolioStressSimulation from '@/components/analytics/PortfolioStressSimulation'
import ShockScenarioStudio from '@/components/analytics/ShockScenarioStudio'
import LiquidityFlowNetworks from '@/components/analytics/LiquidityFlowNetworks'
import FractionalTradingDashboard from '@/components/analytics/FractionalTradingDashboard'
import AuctionPriceDiscovery from '@/components/analytics/AuctionPriceDiscovery'
import BondLiquidityScore from '@/components/analytics/BondLiquidityScore'
import AICopilot from '@/components/ai/AICopilot'
import MarketRegimeDetector from '@/components/ai/MarketRegimeDetector'
import StrategyRecommender from '@/components/ai/StrategyRecommender'

// Mock data generators
const generateYieldCurveData = () => {
  const maturities = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
  return maturities.map(maturity => ({
    maturity,
    yield: 6.5 + (maturity * 0.1) + (Math.random() - 0.5) * 0.5,
    liquidity: 0.3 + Math.random() * 0.7,
    volume: Math.random() * 2000000 + 500000,
    spread: Math.random() * 50 + 10
  }))
}

const generateLiquidityData = () => {
  const sectors = ['Government', 'Banking', 'Infrastructure', 'Power', 'Telecom', 'Oil & Gas']
  const ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A']
  
  return Array.from({ length: 50 }, (_, i) => ({
    isin: `INE${String(i).padStart(10, '0')}`,
    name: `Bond ${i + 1}`,
    maturity: Math.random() * 20 + 1,
    yield: 6 + Math.random() * 3,
    liquidityScore: Math.random(),
    volume: Math.random() * 5000000 + 100000,
    spread: Math.random() * 100 + 5,
    lastTrade: new Date(Date.now() - Math.random() * 86400000).toISOString(),
    sector: sectors[Math.floor(Math.random() * sectors.length)],
    rating: ratings[Math.floor(Math.random() * ratings.length)]
  }))
}

const AnalyticsDashboard: React.FC = () => {
  const [yieldData, setYieldData] = useState(generateYieldCurveData())
  const [liquidityData, setLiquidityData] = useState(generateLiquidityData())
  const [shockValue, setShockValue] = useState(0)
  const [currentRegime, setCurrentRegime] = useState('volatile')
  const [portfolioMetrics, setPortfolioMetrics] = useState({
    totalValue: 125000,
    duration: 4.2,
    convexity: 12.3,
    modifiedDuration: 4.0,
    pvbp: 1250,
    avgYield: 7.2,
    creditRisk: 0.35,
    liquidityScore: 0.68,
    sectorConcentration: {
      'Government': 0.45,
      'Banking': 0.25,
      'Infrastructure': 0.15,
      'Power': 0.10,
      'Others': 0.05
    }
  })

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      setYieldData(generateYieldCurveData())
      setLiquidityData(generateLiquidityData())
    }, 10000) // Update every 10 seconds

    return () => clearInterval(interval)
  }, [])

  const handleShockChange = (metrics: any) => {
    setPortfolioMetrics(prev => ({
      ...prev,
      totalValue: metrics.totalValue,
      duration: metrics.duration,
      convexity: metrics.convexity
    }))
  }

  const handleRegimeChange = (regime: any) => {
    setCurrentRegime(regime.id)
  }

  const handleBondSelect = (bond: any) => {
    console.log('Selected bond:', bond)
  }

  const handleRecommendationSelect = (recommendation: any) => {
    console.log('Selected recommendation:', recommendation)
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-white">BondX Analytics</h1>
              <p className="text-gray-400 mt-1">Advanced AI-powered bond analytics and insights</p>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 px-3 py-1 bg-green-600/20 rounded-full">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-green-400 text-sm font-medium">Live Data</span>
              </div>
              
              <div className="text-right">
                <div className="text-gray-400 text-sm">Last Updated</div>
                <div className="text-white text-sm font-mono">
                  {new Date().toLocaleTimeString()}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* Advanced Analytics Section */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-6">Advanced Predictive Analytics</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Liquidity Forecasting */}
            <LiquidityForecasting />
            
            {/* Spread Forecasting */}
            <SpreadForecasting />
          </div>
        </div>

        {/* Portfolio Stress Simulation */}
        <div className="mb-8">
          <PortfolioStressSimulation />
        </div>

        {/* Advanced Visual & Interactive Analytics */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-6">Advanced Visual & Interactive Analytics</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Shock Scenario Studio */}
            <ShockScenarioStudio />
            
            {/* Liquidity Flow & Correlation Networks */}
            <LiquidityFlowNetworks />
          </div>
        </div>

        {/* Fractional & Market Efficiency Enhancements */}
        <div className="mb-8">
          <h2 className="text-2xl font-bold text-white mb-6">Fractional & Market Efficiency</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Fractional Bond Trading Dashboard */}
            <FractionalTradingDashboard />
            
            {/* Auction-Driven Price Discovery */}
            <AuctionPriceDiscovery />
          </div>
        </div>

        {/* Bond Liquidity Score */}
        <div className="mb-8">
          <BondLiquidityScore />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Visual Analytics */}
          <div className="lg:col-span-2 space-y-8">
            {/* 3D Yield Curve */}
            <YieldCurve3D
              data={yieldData}
              shockValue={shockValue}
              onPointHover={(point) => console.log('Hovered point:', point)}
            />

            {/* Shock Simulator */}
            <ShockSimulator
              initialMetrics={portfolioMetrics}
              onMetricsChange={handleShockChange}
            />

            {/* Liquidity Heatmap */}
            <LiquidityHeatmap
              data={liquidityData}
              onBondSelect={handleBondSelect}
            />
          </div>

          {/* Right Column - AI Components */}
          <div className="space-y-8">
            {/* Market Regime Detector */}
            <MarketRegimeDetector
              onRegimeChange={handleRegimeChange}
            />

            {/* Strategy Recommender */}
            <StrategyRecommender
              portfolioMetrics={portfolioMetrics}
              marketRegime={currentRegime}
              onRecommendationSelect={handleRecommendationSelect}
            />

            {/* AI Copilot */}
            <AICopilot
              onAnalysisRequest={async (query) => {
                // Simulate API call
                await new Promise(resolve => setTimeout(resolve, 1000))
                return { analysis: 'Mock analysis result' }
              }}
            />
          </div>
        </div>

        {/* Bottom Section - Additional Analytics */}
        <div className="mt-12 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Portfolio Summary Cards */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-gray-900 rounded-lg p-6 border border-gray-800"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-blue-600 rounded-lg">
                <TrendingUp className="h-5 w-5 text-white" />
              </div>
              <span className="text-green-400 text-sm font-medium">+2.3%</span>
            </div>
            <div className="text-2xl font-bold text-white">₹{portfolioMetrics.totalValue.toLocaleString()}L</div>
            <div className="text-gray-400 text-sm">Portfolio Value</div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-gray-900 rounded-lg p-6 border border-gray-800"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-orange-600 rounded-lg">
                <Activity className="h-5 w-5 text-white" />
              </div>
              <span className="text-orange-400 text-sm font-medium">4.2Y</span>
            </div>
            <div className="text-2xl font-bold text-white">{portfolioMetrics.duration}</div>
            <div className="text-gray-400 text-sm">Duration</div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-gray-900 rounded-lg p-6 border border-gray-800"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-green-600 rounded-lg">
                <BarChart3 className="h-5 w-5 text-white" />
              </div>
              <span className="text-green-400 text-sm font-medium">68%</span>
            </div>
            <div className="text-2xl font-bold text-white">{(portfolioMetrics.liquidityScore * 100).toFixed(0)}%</div>
            <div className="text-gray-400 text-sm">Liquidity Score</div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="bg-gray-900 rounded-lg p-6 border border-gray-800"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="p-2 bg-purple-600 rounded-lg">
                <Brain className="h-5 w-5 text-white" />
              </div>
              <span className="text-purple-400 text-sm font-medium">AI</span>
            </div>
            <div className="text-2xl font-bold text-white">{portfolioMetrics.avgYield.toFixed(1)}%</div>
            <div className="text-gray-400 text-sm">Avg Yield</div>
          </motion.div>
        </div>

        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-8 bg-gray-900 rounded-lg p-6 border border-gray-800"
        >
          <h3 className="text-lg font-semibold text-white mb-4">Quick Actions</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <button className="p-4 bg-blue-600/20 hover:bg-blue-600/30 rounded-lg border border-blue-600/30 transition-colors">
              <div className="text-blue-400 font-medium">Export Report</div>
              <div className="text-gray-400 text-sm">Download analytics</div>
            </button>
            <button className="p-4 bg-green-600/20 hover:bg-green-600/30 rounded-lg border border-green-600/30 transition-colors">
              <div className="text-green-400 font-medium">Run Stress Test</div>
              <div className="text-gray-400 text-sm">Scenario analysis</div>
            </button>
            <button className="p-4 bg-purple-600/20 hover:bg-purple-600/30 rounded-lg border border-purple-600/30 transition-colors">
              <div className="text-purple-400 font-medium">AI Insights</div>
              <div className="text-gray-400 text-sm">Get recommendations</div>
            </button>
            <button className="p-4 bg-orange-600/20 hover:bg-orange-600/30 rounded-lg border border-orange-600/30 transition-colors">
              <div className="text-orange-400 font-medium">Risk Monitor</div>
              <div className="text-gray-400 text-sm">Real-time alerts</div>
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default AnalyticsDashboard
