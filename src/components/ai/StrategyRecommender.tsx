import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Target, TrendingUp, Shield, AlertTriangle, CheckCircle, ArrowRight, RefreshCw } from 'lucide-react'

interface PortfolioMetrics {
  totalValue: number
  duration: number
  convexity: number
  avgYield: number
  creditRisk: number
  liquidityScore: number
  sectorConcentration: { [sector: string]: number }
}

interface StrategyRecommendation {
  id: string
  title: string
  description: string
  priority: 'high' | 'medium' | 'low'
  impact: 'high' | 'medium' | 'low'
  confidence: number
  actions: StrategyAction[]
  expectedOutcome: string
  riskReduction?: number
  yieldImpact?: number
}

interface StrategyAction {
  id: string
  action: string
  description: string
  urgency: 'immediate' | 'short-term' | 'medium-term'
  difficulty: 'easy' | 'medium' | 'complex'
}

interface StrategyRecommenderProps {
  portfolioMetrics: PortfolioMetrics
  marketRegime: string
  onRecommendationSelect?: (recommendation: StrategyRecommendation) => void
  className?: string
}

const StrategyRecommender: React.FC<StrategyRecommenderProps> = ({
  portfolioMetrics,
  marketRegime,
  onRecommendationSelect,
  className = ''
}) => {
  const [recommendations, setRecommendations] = useState<StrategyRecommendation[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedRecommendation, setSelectedRecommendation] = useState<StrategyRecommendation | null>(null)
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date())

  // Generate AI-powered recommendations
  useEffect(() => {
    const generateRecommendations = async () => {
      setIsLoading(true)
      
      // Simulate AI processing time
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      const newRecommendations: StrategyRecommendation[] = []
      
      // Duration-based recommendations
      if (portfolioMetrics.duration > 4) {
        newRecommendations.push({
          id: 'reduce-duration',
          title: 'Reduce Portfolio Duration',
          description: 'Your portfolio duration of 4.2 years is high for the current volatile market. Consider shifting to shorter-duration bonds.',
          priority: 'high',
          impact: 'high',
          confidence: 0.85,
          riskReduction: 35,
          yieldImpact: -0.5,
          expectedOutcome: 'Reduced interest rate sensitivity and improved capital preservation',
          actions: [
            {
              id: 'sell-long-bonds',
              action: 'Sell bonds with maturity >5 years',
              description: 'Focus on bonds maturing within 2-3 years',
              urgency: 'short-term',
              difficulty: 'medium'
            },
            {
              id: 'buy-short-bonds',
              action: 'Increase allocation to 1-3 year bonds',
              description: 'Target government securities and AAA corporates',
              urgency: 'short-term',
              difficulty: 'easy'
            }
          ]
        })
      }
      
      // Credit risk recommendations
      if (portfolioMetrics.creditRisk > 0.6) {
        newRecommendations.push({
          id: 'improve-credit',
          title: 'Enhance Credit Quality',
          description: 'High credit risk exposure detected. Shift towards investment-grade securities.',
          priority: 'high',
          impact: 'medium',
          confidence: 0.78,
          riskReduction: 25,
          yieldImpact: -0.8,
          expectedOutcome: 'Improved credit stability and reduced default risk',
          actions: [
            {
              id: 'upgrade-ratings',
              action: 'Replace BBB bonds with AA+ bonds',
              description: 'Focus on government and quasi-government securities',
              urgency: 'medium-term',
              difficulty: 'medium'
            }
          ]
        })
      }
      
      // Liquidity recommendations
      if (portfolioMetrics.liquidityScore < 0.6) {
        newRecommendations.push({
          id: 'improve-liquidity',
          title: 'Enhance Portfolio Liquidity',
          description: 'Low liquidity score detected. Increase allocation to highly liquid bonds.',
          priority: 'medium',
          impact: 'medium',
          confidence: 0.72,
          expectedOutcome: 'Better trading flexibility and reduced market impact costs',
          actions: [
            {
              id: 'liquid-bonds',
              action: 'Increase liquid bond allocation',
              description: 'Target bonds with daily volume >₹50L',
              urgency: 'short-term',
              difficulty: 'easy'
            }
          ]
        })
      }
      
      // Market regime specific recommendations
      if (marketRegime === 'volatile' || marketRegime === 'crisis') {
        newRecommendations.push({
          id: 'defensive-positioning',
          title: 'Defensive Positioning',
          description: 'Current market regime requires defensive strategies. Focus on capital preservation.',
          priority: 'high',
          impact: 'high',
          confidence: 0.88,
          riskReduction: 40,
          yieldImpact: -1.2,
          expectedOutcome: 'Improved downside protection and reduced volatility',
          actions: [
            {
              id: 'increase-cash',
              action: 'Increase cash allocation to 15-20%',
              description: 'Maintain liquidity for opportunities',
              urgency: 'immediate',
              difficulty: 'easy'
            },
            {
              id: 'hedge-positions',
              action: 'Consider interest rate hedges',
              description: 'Use futures or swaps to hedge duration risk',
              urgency: 'short-term',
              difficulty: 'complex'
            }
          ]
        })
      }
      
      // Sector concentration recommendations
      const maxSectorWeight = Math.max(...Object.values(portfolioMetrics.sectorConcentration))
      if (maxSectorWeight > 0.4) {
        newRecommendations.push({
          id: 'diversify-sectors',
          title: 'Diversify Sector Exposure',
          description: 'High sector concentration detected. Spread risk across multiple sectors.',
          priority: 'medium',
          impact: 'medium',
          confidence: 0.75,
          riskReduction: 20,
          expectedOutcome: 'Reduced sector-specific risk and improved diversification',
          actions: [
            {
              id: 'sector-rebalance',
              action: 'Rebalance sector allocation',
              description: 'Limit any single sector to <30% of portfolio',
              urgency: 'medium-term',
              difficulty: 'medium'
            }
          ]
        })
      }
      
      // Sort by priority and confidence
      newRecommendations.sort((a, b) => {
        const priorityOrder = { high: 3, medium: 2, low: 1 }
        if (priorityOrder[a.priority] !== priorityOrder[b.priority]) {
          return priorityOrder[b.priority] - priorityOrder[a.priority]
        }
        return b.confidence - a.confidence
      })
      
      setRecommendations(newRecommendations)
      setLastUpdate(new Date())
      setIsLoading(false)
    }

    generateRecommendations()
  }, [portfolioMetrics, marketRegime])

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'text-red-400 bg-red-600/20'
      case 'medium': return 'text-yellow-400 bg-yellow-600/20'
      case 'low': return 'text-green-400 bg-green-600/20'
      default: return 'text-gray-400 bg-gray-600/20'
    }
  }

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'high': return 'text-blue-400'
      case 'medium': return 'text-yellow-400'
      case 'low': return 'text-green-400'
      default: return 'text-gray-400'
    }
  }

  const getUrgencyColor = (urgency: string) => {
    switch (urgency) {
      case 'immediate': return 'text-red-400'
      case 'short-term': return 'text-orange-400'
      case 'medium-term': return 'text-yellow-400'
      default: return 'text-gray-400'
    }
  }

  const handleRecommendationClick = (recommendation: StrategyRecommendation) => {
    setSelectedRecommendation(recommendation)
    onRecommendationSelect?.(recommendation)
  }

  const refreshRecommendations = () => {
    setIsLoading(true)
    // Trigger re-generation
    setTimeout(() => setIsLoading(false), 1000)
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
            <div className="p-2 bg-green-600 rounded-lg">
              <Target className="h-6 w-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-white">AI Strategy Recommender</h3>
              <p className="text-gray-400 text-sm">Personalized investment recommendations</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <div className="text-right text-sm">
              <div className="text-gray-400">Last Updated</div>
              <div className="text-white font-mono">
                {lastUpdate.toLocaleTimeString()}
              </div>
            </div>
            <button
              onClick={refreshRecommendations}
              disabled={isLoading}
              className="p-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed rounded-lg text-white transition-colors"
            >
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Loading State */}
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500 mx-auto mb-4"></div>
              <p className="text-gray-400">Analyzing portfolio and generating recommendations...</p>
            </div>
          </div>
        ) : (
          <>
            {/* Recommendations List */}
            <div className="space-y-4">
              {recommendations.map((recommendation, index) => (
                <motion.div
                  key={recommendation.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  onClick={() => handleRecommendationClick(recommendation)}
                  className="bg-gray-800 rounded-lg p-4 border border-gray-700 hover:border-gray-600 cursor-pointer transition-all hover:bg-gray-750"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-start space-x-3">
                      <div className="p-2 bg-green-600/20 rounded-lg">
                        <Target className="h-5 w-5 text-green-400" />
                      </div>
                      <div className="flex-1">
                        <h4 className="text-white font-medium mb-1">
                          {recommendation.title}
                        </h4>
                        <p className="text-gray-300 text-sm">
                          {recommendation.description}
                        </p>
                      </div>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getPriorityColor(recommendation.priority)}`}>
                        {recommendation.priority.toUpperCase()}
                      </span>
                      <ArrowRight className="h-4 w-4 text-gray-400" />
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div className="text-center">
                      <div className="text-white font-medium">
                        {(recommendation.confidence * 100).toFixed(0)}%
                      </div>
                      <div className="text-gray-400">Confidence</div>
                    </div>
                    <div className="text-center">
                      <div className={`font-medium ${getImpactColor(recommendation.impact)}`}>
                        {recommendation.impact.toUpperCase()}
                      </div>
                      <div className="text-gray-400">Impact</div>
                    </div>
                    <div className="text-center">
                      <div className="text-white font-medium">
                        {recommendation.actions.length}
                      </div>
                      <div className="text-gray-400">Actions</div>
                    </div>
                  </div>
                  
                  {recommendation.riskReduction && (
                    <div className="mt-3 pt-3 border-t border-gray-700">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-400">Expected Risk Reduction:</span>
                        <span className="text-green-400 font-medium">
                          -{recommendation.riskReduction}%
                        </span>
                      </div>
                      {recommendation.yieldImpact && (
                        <div className="flex items-center justify-between text-sm mt-1">
                          <span className="text-gray-400">Yield Impact:</span>
                          <span className={`font-medium ${
                            recommendation.yieldImpact < 0 ? 'text-red-400' : 'text-green-400'
                          }`}>
                            {recommendation.yieldImpact > 0 ? '+' : ''}{recommendation.yieldImpact}%
                          </span>
                        </div>
                      )}
                    </div>
                  )}
                </motion.div>
              ))}
            </div>

            {/* Selected Recommendation Details */}
            <AnimatePresence>
              {selectedRecommendation && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="bg-gray-800 rounded-lg p-6 border border-gray-700"
                >
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-white font-medium text-lg">
                      {selectedRecommendation.title}
                    </h4>
                    <button
                      onClick={() => setSelectedRecommendation(null)}
                      className="text-gray-400 hover:text-white"
                    >
                      ×
                    </button>
                  </div>
                  
                  <div className="mb-4">
                    <p className="text-gray-300 mb-3">
                      {selectedRecommendation.description}
                    </p>
                    <div className="bg-blue-600/20 rounded-lg p-3">
                      <div className="flex items-start space-x-2">
                        <CheckCircle className="h-5 w-5 text-blue-400 mt-0.5" />
                        <div>
                          <div className="text-blue-400 font-medium mb-1">Expected Outcome</div>
                          <div className="text-gray-300 text-sm">
                            {selectedRecommendation.expectedOutcome}
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <h5 className="text-white font-medium mb-3">Recommended Actions</h5>
                    <div className="space-y-3">
                      {selectedRecommendation.actions.map((action, index) => (
                        <div key={action.id} className="bg-gray-700 rounded-lg p-3">
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex items-center space-x-2">
                              <div className="w-6 h-6 bg-green-600 rounded-full flex items-center justify-center text-white text-xs font-medium">
                                {index + 1}
                              </div>
                              <span className="text-white font-medium">{action.action}</span>
                            </div>
                            <div className="flex items-center space-x-2">
                              <span className={`text-xs px-2 py-1 rounded ${getUrgencyColor(action.urgency)} bg-gray-600`}>
                                {action.urgency}
                              </span>
                              <span className="text-xs px-2 py-1 rounded text-gray-400 bg-gray-600">
                                {action.difficulty}
                              </span>
                            </div>
                          </div>
                          <p className="text-gray-300 text-sm ml-8">
                            {action.description}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* No Recommendations */}
            {recommendations.length === 0 && (
              <div className="text-center py-12">
                <div className="mx-auto h-16 w-16 text-gray-600 mb-4">
                  <Target className="h-full w-full" />
                </div>
                <h4 className="text-white font-medium mb-2">No Recommendations</h4>
                <p className="text-gray-400">
                  Your portfolio appears to be well-balanced for current market conditions.
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </motion.div>
  )
}

export default StrategyRecommender
