import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, Bot, User, Loader2, Sparkles, TrendingUp, AlertTriangle, Calculator, Brain, Target, BarChart3, Lightbulb, ChevronDown, ChevronRight } from 'lucide-react'

interface ExplanationFactor {
  factor: string
  impact: number
  confidence: number
  description: string
  causalChain: string[]
}

interface ChatMessage {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
  metadata?: {
    type: 'analysis' | 'recommendation' | 'explanation' | 'error' | 'explainable_ai'
    data?: any
    explanations?: ExplanationFactor[]
    confidence?: number
    causalAnalysis?: {
      primaryCause: string
      secondaryFactors: string[]
      confidence: number
    }
  }
}

interface AICopilotProps {
  onAnalysisRequest?: (query: string) => Promise<any>
  className?: string
}

const SUGGESTED_QUERIES = [
  "What happens to my portfolio if yields rise 25bps?",
  "Explain my current duration and convexity in simple terms",
  "Suggest a safer bond allocation in a volatile market",
  "Which bonds have the highest liquidity risk?",
  "How does the current market regime affect my strategy?",
  "What's the impact of RBI policy changes on my portfolio?",
  "Why did my portfolio duration increase by 0.5 years?",
  "Which bonds are likely to perform best next week?",
  "Explain the causal factors behind recent spread movements",
  "What's driving the liquidity changes in my portfolio?",
  "Predict which bonds will have the highest returns in 30 days",
  "Identify bonds with liquidity risk in the next 72 hours",
  "Rank bonds by predicted spread tightening potential",
  "Which bonds are most likely to become illiquid next week?",
  "Find bonds with the best risk-adjusted return potential",
  "Predict the impact of inflation shock on my portfolio",
  "Which bonds have the highest probability of rating upgrade?",
  "Identify bonds with predicted spread widening risk"
]

const AICopilot: React.FC<AICopilotProps> = ({
  onAnalysisRequest,
  className = ''
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'assistant',
      content: "Hello! I'm BondX AI Copilot. I can help you analyze your portfolio, explain risk metrics, and provide investment recommendations. What would you like to know?",
      timestamp: new Date(),
      metadata: { type: 'explanation' }
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isMinimized, setIsMinimized] = useState(false)
  const [expandedExplanations, setExpandedExplanations] = useState<Set<string>>(new Set())
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Handle sending message
  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      // Simulate AI response (in real app, this would call the backend)
      const response = await simulateAIResponse(inputValue.trim())
      
      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.content,
        timestamp: new Date(),
        metadata: response.metadata
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: "I apologize, but I encountered an error processing your request. Please try again.",
        timestamp: new Date(),
        metadata: { type: 'error' }
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  // Simulate AI response (replace with actual API call)
  const simulateAIResponse = async (query: string): Promise<{ content: string; metadata: any }> => {
    await new Promise(resolve => setTimeout(resolve, 1500 + Math.random() * 1000))

    const lowerQuery = query.toLowerCase()

    if (lowerQuery.includes('yield') && lowerQuery.includes('rise')) {
      return {
        content: "Based on your current portfolio with a duration of 4.2 years, a 25bps yield rise would result in approximately -1.05% price impact. However, your portfolio's convexity of 12.3 provides some protection, reducing the actual impact to about -0.98%. I recommend monitoring shorter-duration bonds for better protection against rate hikes.",
        metadata: { 
          type: 'explainable_ai',
          confidence: 0.87,
          explanations: [
            {
              factor: "Portfolio Duration",
              impact: 0.6,
              confidence: 0.92,
              description: "Your portfolio's 4.2-year duration is the primary driver of rate sensitivity",
              causalChain: ["Rate increase → Bond prices fall → Duration amplifies impact"]
            },
            {
              factor: "Convexity Protection",
              impact: -0.15,
              confidence: 0.85,
              description: "Positive convexity reduces the actual price impact by 0.07%",
              causalChain: ["Rate increase → Duration changes → Convexity provides cushion"]
            },
            {
              factor: "Bond Mix",
              impact: 0.25,
              confidence: 0.78,
              description: "Corporate bonds (60% of portfolio) are more sensitive than government bonds",
              causalChain: ["Rate increase → Credit spreads widen → Corporate bonds underperform"]
            }
          ],
          causalAnalysis: {
            primaryCause: "Interest rate sensitivity driven by portfolio duration",
            secondaryFactors: ["Credit spread widening", "Liquidity premium changes", "Market volatility"],
            confidence: 0.87
          }
        }
      }
    }

    if (lowerQuery.includes('duration') && lowerQuery.includes('convexity')) {
      return {
        content: "**Duration (4.2 years)**: This measures how sensitive your portfolio is to interest rate changes. Think of it as a 'speedometer' for rate sensitivity. A 1% rate increase typically causes a 4.2% price decline.\n\n**Convexity (12.3)**: This is the 'acceleration' - it shows how duration changes as rates move. Positive convexity (like yours) provides protection by reducing duration impact in volatile markets. It's like having airbags in your portfolio!",
        metadata: { type: 'explanation' }
      }
    }

    if (lowerQuery.includes('safer') || lowerQuery.includes('allocation')) {
      return {
        content: "Given the current volatile market regime, I recommend a multi-step strategic approach:\n\n**Phase 1: Immediate Risk Reduction (Next 7 days)**\n• Reduce duration to 2-3 years (currently 4.2)\n• Increase allocation to AAA-rated government securities by 15%\n• Exit positions in bonds with liquidity score <30%\n\n**Phase 2: Portfolio Optimization (Next 30 days)**\n• Focus on liquid bonds with daily trading volume >₹50L\n• Consider floating rate bonds for rate protection\n• Diversify across 3-4 sectors to reduce concentration risk\n\n**Phase 3: Long-term Positioning (Next 90 days)**\n• Gradually increase exposure to high-quality corporate bonds\n• Implement dynamic hedging strategies\n• Monitor and adjust based on market regime changes\n\nThis phased approach would reduce your interest rate risk by ~40% while maintaining yield and improving liquidity.",
        metadata: { 
          type: 'explainable_ai',
          confidence: 0.89,
          explanations: [
            {
              factor: "Market Regime Analysis",
              impact: 0.4,
              confidence: 0.92,
              description: "Current volatile regime requires defensive positioning",
              causalChain: ["Market volatility → Risk appetite reduction → Defensive strategy needed"]
            },
            {
              factor: "Duration Risk",
              impact: 0.35,
              confidence: 0.88,
              description: "4.2-year duration is vulnerable to rate hikes",
              causalChain: ["High duration → Rate sensitivity → Risk reduction priority"]
            },
            {
              factor: "Liquidity Constraints",
              impact: 0.25,
              confidence: 0.85,
              description: "Illiquid positions limit flexibility in volatile markets",
              causalChain: ["Low liquidity → Trading constraints → Liquidity improvement needed"]
            }
          ],
          causalAnalysis: {
            primaryCause: "Volatile market regime requiring defensive positioning",
            secondaryFactors: ["Duration risk", "Liquidity constraints", "Concentration risk"],
            confidence: 0.89
          }
        }
      }
    }

    if (lowerQuery.includes('liquidity')) {
      return {
        content: "Your portfolio has moderate liquidity risk:\n\n• **High Risk**: 3 bonds with liquidity score <30%\n• **Medium Risk**: 7 bonds with 30-70% liquidity\n• **Low Risk**: 12 bonds with >70% liquidity\n\n**Recommendation**: Consider reducing positions in illiquid bonds during volatile periods. The 3 high-risk bonds represent 15% of your portfolio - consider rebalancing.",
        metadata: { type: 'analysis' }
      }
    }

    if (lowerQuery.includes('market regime')) {
      return {
        content: "Current market regime: **VOLATILE** 🔄\n\n• **Volatility Index**: 18.5 (High)\n• **Liquidity Score**: 65% (Moderate)\n• **Credit Spreads**: Widening (+15bps)\n• **RBI Stance**: Hawkish\n\n**Strategy**: Defensive positioning recommended. Focus on quality over yield, maintain higher cash levels, and consider hedging strategies.",
        metadata: { type: 'analysis' }
      }
    }

    if (lowerQuery.includes('rbi') || lowerQuery.includes('policy')) {
      return {
        content: "RBI's hawkish stance impacts your portfolio through:\n\n• **Direct Impact**: +25-50bps rate hikes expected\n• **Duration Risk**: Your 4.2Y duration is vulnerable\n• **Credit Spreads**: Likely to widen by 10-20bps\n• **Liquidity**: May tighten, affecting trading costs\n\n**Action Items**:\n1. Reduce duration to 2-3 years\n2. Increase government bond allocation\n3. Monitor credit spreads closely",
        metadata: { type: 'analysis' }
      }
    }

    if (lowerQuery.includes('why') && lowerQuery.includes('duration')) {
      return {
        content: "Your portfolio duration increased by 0.5 years due to several interconnected factors. The primary driver is the recent yield curve steepening, which has disproportionately affected longer-duration bonds in your portfolio.",
        metadata: { 
          type: 'explainable_ai',
          confidence: 0.91,
          explanations: [
            {
              factor: "Yield Curve Steepening",
              impact: 0.4,
              confidence: 0.95,
              description: "5-7 year BBB bond yields rose 35bps while short-term rates remained stable",
              causalChain: ["Market expectations → Long-term yields rise → Duration increases"]
            },
            {
              factor: "Portfolio Rebalancing",
              impact: 0.2,
              confidence: 0.88,
              description: "Recent purchases of 5-year corporate bonds increased average maturity",
              causalChain: ["New bond purchases → Higher maturity → Portfolio duration rises"]
            },
            {
              factor: "Convexity Effects",
              impact: 0.1,
              confidence: 0.82,
              description: "Positive convexity in volatile markets extends effective duration",
              causalChain: ["Market volatility → Convexity activation → Duration extension"]
            }
          ],
          causalAnalysis: {
            primaryCause: "Yield curve steepening affecting 5-7 year BBB bond yields",
            secondaryFactors: ["Portfolio rebalancing", "Convexity effects", "Market volatility"],
            confidence: 0.91
          }
        }
      }
    }

    if (lowerQuery.includes('likely to perform') || lowerQuery.includes('next week')) {
      return {
        content: "Based on our predictive models, here are the bonds likely to perform best next week:\n\n1. **HDFC Bank 7.25% 2029** - Expected +2.3% return (87% confidence)\n2. **Power Finance Corp 6.84% 2028** - Expected +1.8% return (82% confidence)\n3. **Indian Oil Corp 6.62% 2029** - Expected +1.5% return (79% confidence)\n\nThese predictions are based on liquidity forecasts, spread analysis, and macro indicators.",
        metadata: { 
          type: 'explainable_ai',
          confidence: 0.83,
          explanations: [
            {
              factor: "Liquidity Forecast",
              impact: 0.35,
              confidence: 0.89,
              description: "Predicted increase in trading volume for these bonds",
              causalChain: ["Market activity → Higher liquidity → Better price discovery"]
            },
            {
              factor: "Spread Compression",
              impact: 0.3,
              confidence: 0.85,
              description: "Credit spreads expected to tighten by 5-8bps",
              causalChain: ["Risk sentiment → Spread compression → Price appreciation"]
            },
            {
              factor: "Macro Environment",
              impact: 0.25,
              confidence: 0.78,
              description: "Favorable macro conditions supporting corporate bonds",
              causalChain: ["Economic indicators → Risk appetite → Bond demand"]
            }
          ],
          causalAnalysis: {
            primaryCause: "Predicted liquidity improvements and spread compression",
            secondaryFactors: ["Macro environment", "Risk sentiment", "Trading activity"],
            confidence: 0.83
          }
        }
      }
    }

    if (lowerQuery.includes('causal factors') || lowerQuery.includes('spread movements')) {
      return {
        content: "Recent spread movements are driven by a complex interplay of market forces. The primary causal factor is the shift in risk sentiment following RBI's policy stance, which has created a ripple effect across credit markets.",
        metadata: { 
          type: 'explainable_ai',
          confidence: 0.86,
          explanations: [
            {
              factor: "RBI Policy Impact",
              impact: 0.45,
              confidence: 0.92,
              description: "Hawkish stance increased risk-free rates by 25bps",
              causalChain: ["RBI policy → Risk-free rates rise → Credit spreads adjust"]
            },
            {
              factor: "Liquidity Conditions",
              impact: 0.3,
              confidence: 0.88,
              description: "Tightening liquidity increased trading costs",
              causalChain: ["Liquidity reduction → Higher trading costs → Spread widening"]
            },
            {
              factor: "Credit Risk Perception",
              impact: 0.25,
              confidence: 0.81,
              description: "Market reassessment of corporate credit quality",
              causalChain: ["Economic uncertainty → Credit risk premium → Spread adjustment"]
            }
          ],
          causalAnalysis: {
            primaryCause: "RBI's hawkish policy stance affecting risk-free rates",
            secondaryFactors: ["Liquidity conditions", "Credit risk perception", "Market volatility"],
            confidence: 0.86
          }
        }
      }
    }

    // New predictive query handlers
    if (lowerQuery.includes('predict') && lowerQuery.includes('highest returns')) {
      return {
        content: "**Top 5 Bonds with Highest Predicted Returns (30-day horizon):**\n\n1. **HDFC Bank 7.25% 2029** - +3.2% (91% confidence)\n2. **Power Finance Corp 6.84% 2028** - +2.8% (88% confidence)\n3. **Indian Oil Corp 6.62% 2029** - +2.5% (85% confidence)\n4. **Small Industries Dev Bank 6.95% 2027** - +2.1% (82% confidence)\n5. **National Bank for Agriculture 6.78% 2028** - +1.9% (79% confidence)\n\n*Based on ensemble ML models combining liquidity forecasts, spread predictions, and macro indicators.*",
        metadata: { 
          type: 'explainable_ai',
          confidence: 0.87,
          explanations: [
            {
              factor: "Liquidity Improvement Forecast",
              impact: 0.4,
              confidence: 0.92,
              description: "Predicted 15-25% increase in trading volume",
              causalChain: ["Market activity → Higher liquidity → Better price discovery → Returns"]
            },
            {
              factor: "Spread Compression Prediction",
              impact: 0.35,
              confidence: 0.89,
              description: "Credit spreads expected to tighten by 8-12bps",
              causalChain: ["Risk sentiment → Spread compression → Price appreciation"]
            },
            {
              factor: "Macro Environment Support",
              impact: 0.25,
              confidence: 0.84,
              description: "Favorable economic indicators supporting corporate bonds",
              causalChain: ["Economic data → Risk appetite → Bond demand → Returns"]
            }
          ],
          causalAnalysis: {
            primaryCause: "Predicted liquidity improvements and spread compression",
            secondaryFactors: ["Macro environment", "Risk sentiment", "Trading activity"],
            confidence: 0.87
          }
        }
      }
    }

    if (lowerQuery.includes('liquidity risk') && lowerQuery.includes('72 hours')) {
      return {
        content: "**Bonds at Liquidity Risk (Next 72 Hours):**\n\n🚨 **HIGH RISK (BLS < 30):**\n• GMR Airports 6.45% 2030 - BLS: 22 (Critical)\n• JSW Steel 6.78% 2029 - BLS: 28 (Critical)\n\n⚠️ **MEDIUM RISK (BLS 30-50):**\n• Muthoot Finance 7.12% 2028 - BLS: 35 (High)\n• ONGC Petro Additions 6.89% 2027 - BLS: 42 (High)\n\n**Recommendation**: Consider reducing positions in high-risk bonds or implementing hedging strategies.",
        metadata: { 
          type: 'explainable_ai',
          confidence: 0.89,
          explanations: [
            {
              factor: "Trading Volume Decline",
              impact: 0.45,
              confidence: 0.94,
              description: "Predicted 40-60% reduction in trading activity",
              causalChain: ["Market conditions → Volume decline → Liquidity risk"]
            },
            {
              factor: "Bid-Ask Spread Widening",
              impact: 0.35,
              confidence: 0.91,
              description: "Expected 15-25bps spread widening",
              causalChain: ["Liquidity reduction → Spread widening → Trading difficulty"]
            },
            {
              factor: "Market Maker Withdrawal",
              impact: 0.2,
              confidence: 0.87,
              description: "Reduced market maker participation",
              causalChain: ["Risk aversion → Market maker exit → Liquidity reduction"]
            }
          ],
          causalAnalysis: {
            primaryCause: "Predicted trading volume decline and spread widening",
            secondaryFactors: ["Market maker withdrawal", "Risk aversion", "Market conditions"],
            confidence: 0.89
          }
        }
      }
    }

    if (lowerQuery.includes('rank') && lowerQuery.includes('spread tightening')) {
      return {
        content: "**Bonds Ranked by Spread Tightening Potential:**\n\n1. **HDFC Bank 7.25% 2029** - -12bps (89% confidence)\n2. **Power Finance Corp 6.84% 2028** - -10bps (86% confidence)\n3. **Indian Oil Corp 6.62% 2029** - -8bps (83% confidence)\n4. **Small Industries Dev Bank 6.95% 2027** - -7bps (80% confidence)\n5. **National Bank for Agriculture 6.78% 2028** - -6bps (77% confidence)\n\n*Based on credit quality improvements, liquidity forecasts, and market sentiment analysis.*",
        metadata: { 
          type: 'explainable_ai',
          confidence: 0.85,
          explanations: [
            {
              factor: "Credit Quality Improvement",
              impact: 0.4,
              confidence: 0.91,
              description: "Predicted rating upgrades or outlook improvements",
              causalChain: ["Credit improvement → Spread compression → Price appreciation"]
            },
            {
              factor: "Liquidity Enhancement",
              impact: 0.35,
              confidence: 0.88,
              description: "Expected increase in trading activity",
              causalChain: ["Higher liquidity → Better price discovery → Spread tightening"]
            },
            {
              factor: "Market Sentiment Shift",
              impact: 0.25,
              confidence: 0.82,
              description: "Positive sentiment towards corporate bonds",
              causalChain: ["Risk appetite → Demand increase → Spread compression"]
            }
          ],
          causalAnalysis: {
            primaryCause: "Predicted credit quality improvements and liquidity enhancement",
            secondaryFactors: ["Market sentiment", "Risk appetite", "Trading activity"],
            confidence: 0.85
          }
        }
      }
    }

    if (lowerQuery.includes('liquidity changes') || lowerQuery.includes('driving')) {
      return {
        content: "Your portfolio's liquidity changes are primarily driven by market microstructure shifts and individual bond characteristics. The overall liquidity score has decreased from 78% to 65% over the past week.",
        metadata: { 
          type: 'explainable_ai',
          confidence: 0.84,
          explanations: [
            {
              factor: "Market Volatility",
              impact: 0.4,
              confidence: 0.91,
              description: "Increased volatility reduced market maker participation",
              causalChain: ["Volatility spike → Market makers withdraw → Liquidity drops"]
            },
            {
              factor: "Bond-Specific Factors",
              impact: 0.35,
              confidence: 0.87,
              description: "3 bonds approaching maturity have reduced trading activity",
              causalChain: ["Maturity proximity → Reduced interest → Lower liquidity"]
            },
            {
              factor: "Macro Environment",
              impact: 0.25,
              confidence: 0.79,
              description: "Uncertainty about rate path affected overall market depth",
              causalChain: ["Policy uncertainty → Risk aversion → Liquidity reduction"]
            }
          ],
          causalAnalysis: {
            primaryCause: "Market volatility reducing market maker participation",
            secondaryFactors: ["Bond-specific factors", "Macro environment", "Trading patterns"],
            confidence: 0.84
          }
        }
      }
    }

    // Default response
    return {
      content: "I understand you're asking about: \"" + query + "\". I can help you with portfolio analysis, risk metrics, market insights, and investment recommendations. Could you be more specific about what you'd like to know?",
      metadata: { type: 'explanation' }
    }
  }

  const handleSuggestedQuery = (query: string) => {
    setInputValue(query)
  }

  const clearChat = () => {
    setMessages([messages[0]]) // Keep the welcome message
  }

  const toggleExplanation = (messageId: string) => {
    setExpandedExplanations(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageId)) {
        newSet.delete(messageId)
      } else {
        newSet.add(messageId)
      }
      return newSet
    })
  }

  const renderExplanation = (message: ChatMessage) => {
    if (message.metadata?.type !== 'explainable_ai' || !message.metadata.explanations) {
      return null
    }

    const isExpanded = expandedExplanations.has(message.id)
    const { explanations, confidence, causalAnalysis } = message.metadata

    return (
      <motion.div
        initial={{ opacity: 0, height: 0 }}
        animate={{ opacity: 1, height: isExpanded ? 'auto' : 0 }}
        exit={{ opacity: 0, height: 0 }}
        className="mt-4 border-t border-gray-700 pt-4"
      >
        <button
          onClick={() => toggleExplanation(message.id)}
          className="flex items-center space-x-2 text-sm text-purple-400 hover:text-purple-300 transition-colors mb-3"
        >
          {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          <Brain className="w-4 h-4" />
          <span>Explainable AI Analysis</span>
          <span className="text-xs bg-purple-500/20 px-2 py-1 rounded">
            {Math.round(confidence * 100)}% confidence
          </span>
        </button>

        {isExpanded && (
          <div className="space-y-4">
            {/* Causal Analysis */}
            {causalAnalysis && (
              <div className="bg-gray-800/50 rounded-lg p-4">
                <h5 className="text-sm font-medium text-white mb-2 flex items-center">
                  <Target className="w-4 h-4 mr-2 text-blue-400" />
                  Causal Analysis
                </h5>
                <div className="space-y-2">
                  <div>
                    <span className="text-xs text-gray-400">Primary Cause:</span>
                    <p className="text-sm text-white">{causalAnalysis.primaryCause}</p>
                  </div>
                  <div>
                    <span className="text-xs text-gray-400">Secondary Factors:</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {causalAnalysis.secondaryFactors.map((factor, index) => (
                        <span key={index} className="text-xs bg-gray-700 px-2 py-1 rounded">
                          {factor}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Contributing Factors */}
            <div className="space-y-3">
              <h5 className="text-sm font-medium text-white flex items-center">
                <BarChart3 className="w-4 h-4 mr-2 text-green-400" />
                Contributing Factors
              </h5>
              {explanations.map((explanation, index) => (
                <div key={index} className="bg-gray-800/30 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <h6 className="text-sm font-medium text-white">{explanation.factor}</h6>
                    <div className="flex items-center space-x-2">
                      <span className={`text-xs px-2 py-1 rounded ${
                        explanation.impact > 0 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'
                      }`}>
                        {explanation.impact > 0 ? '+' : ''}{(explanation.impact * 100).toFixed(1)}%
                      </span>
                      <span className="text-xs text-gray-400">
                        {Math.round(explanation.confidence * 100)}%
                      </span>
                    </div>
                  </div>
                  <p className="text-xs text-gray-300 mb-2">{explanation.description}</p>
                  <div className="space-y-1">
                    <span className="text-xs text-gray-400">Causal Chain:</span>
                    <div className="flex items-center space-x-1 text-xs text-gray-300">
                      {explanation.causalChain.map((step, stepIndex) => (
                        <React.Fragment key={stepIndex}>
                          <span>{step}</span>
                          {stepIndex < explanation.causalChain.length - 1 && (
                            <span className="text-gray-500">→</span>
                          )}
                        </React.Fragment>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={`bg-gray-900 rounded-lg overflow-hidden ${className} ${
        isMinimized ? 'h-16' : 'h-[600px]'
      } transition-all duration-300`}
    >
      {/* Header */}
      <div className="p-4 border-b border-gray-800 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-purple-600 rounded-lg">
            <Bot className="h-5 w-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">BondX AI Copilot</h3>
            <p className="text-gray-400 text-sm">Your intelligent bond advisor</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={clearChat}
            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-md text-sm text-white transition-colors"
          >
            Clear
          </button>
          <button
            onClick={() => setIsMinimized(!isMinimized)}
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded-md text-white transition-colors"
          >
            {isMinimized ? '↑' : '↓'}
          </button>
        </div>
      </div>

      {!isMinimized && (
        <>
          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 h-[400px]">
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex items-start space-x-3 max-w-[80%] ${
                  message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''
                }`}>
                  <div className={`p-2 rounded-lg ${
                    message.type === 'user' 
                      ? 'bg-blue-600' 
                      : message.metadata?.type === 'error'
                      ? 'bg-red-600'
                      : 'bg-gray-700'
                  }`}>
                    {message.type === 'user' ? (
                      <User className="h-4 w-4 text-white" />
                    ) : (
                      <Bot className="h-4 w-4 text-white" />
                    )}
                  </div>
                  
                  <div className={`rounded-lg p-3 ${
                    message.type === 'user' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-800 text-white'
                  }`}>
                    <div className="whitespace-pre-wrap text-sm">
                      {message.content}
                    </div>
                    <div className={`text-xs mt-2 ${
                      message.type === 'user' ? 'text-blue-200' : 'text-gray-400'
                    }`}>
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                    {message.type === 'assistant' && renderExplanation(message)}
                  </div>
                </div>
              </motion.div>
            ))}
            
            {isLoading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex justify-start"
              >
                <div className="flex items-start space-x-3">
                  <div className="p-2 bg-gray-700 rounded-lg">
                    <Bot className="h-4 w-4 text-white" />
                  </div>
                  <div className="bg-gray-800 rounded-lg p-3">
                    <div className="flex items-center space-x-2">
                      <Loader2 className="h-4 w-4 animate-spin text-white" />
                      <span className="text-white text-sm">AI is thinking...</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Suggested Queries */}
          {messages.length === 1 && (
            <div className="p-4 border-t border-gray-800">
              <h4 className="text-white font-medium mb-3 flex items-center">
                <Sparkles className="h-4 w-4 mr-2" />
                Try asking:
              </h4>
              <div className="grid grid-cols-1 gap-2">
                {SUGGESTED_QUERIES.slice(0, 3).map((query, index) => (
                  <button
                    key={index}
                    onClick={() => handleSuggestedQuery(query)}
                    className="text-left p-2 bg-gray-800 hover:bg-gray-700 rounded-md text-sm text-gray-300 hover:text-white transition-colors"
                  >
                    {query}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input */}
          <div className="p-4 border-t border-gray-800">
            <div className="flex items-center space-x-3">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder="Ask me about your portfolio, risk metrics, or market insights..."
                className="flex-1 px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                disabled={isLoading}
              />
              <button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
                className="p-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg text-white transition-colors"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </div>
        </>
      )}
    </motion.div>
  )
}

export default AICopilot
