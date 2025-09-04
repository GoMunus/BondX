import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, Pause, RotateCcw, Sparkles, Zap, Target, TrendingUp } from 'lucide-react'

interface DemoStep {
  id: string
  title: string
  description: string
  component: string
  duration: number
  highlight: string
}

const DemoShowcase: React.FC = () => {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentStep, setCurrentStep] = useState(0)
  const [isAutoPlaying, setIsAutoPlaying] = useState(false)

  const demoSteps: DemoStep[] = [
    {
      id: 'yield-curve',
      title: '3D Yield Curve Visualization',
      description: 'Interactive 3D yield curve with real-time data, hover effects, and shock simulation capabilities.',
      component: 'YieldCurve3D',
      duration: 8000,
      highlight: 'Bloomberg-grade 3D visualization with Three.js'
    },
    {
      id: 'shock-simulator',
      title: 'Interest Rate Shock Simulator',
      description: 'Simulate RBI repo rate hikes and instantly see portfolio impact on duration, convexity, and P&L.',
      component: 'ShockSimulator',
      duration: 6000,
      highlight: 'Real-time portfolio stress testing'
    },
    {
      id: 'liquidity-heatmap',
      title: 'Liquidity Heatmap',
      description: 'Color-coded heatmap showing bond liquidity scores with live updates and filtering capabilities.',
      component: 'LiquidityHeatmap',
      duration: 7000,
      highlight: 'Real-time liquidity pulse monitoring'
    },
    {
      id: 'ai-copilot',
      title: 'AI Copilot Assistant',
      description: 'Natural language queries about portfolio analysis, risk metrics, and investment recommendations.',
      component: 'AICopilot',
      duration: 8000,
      highlight: 'ChatGPT-style bond analytics assistant'
    },
    {
      id: 'regime-detector',
      title: 'Market Regime Detector',
      description: 'ML-powered classification of market conditions: Calm, Volatile, Crisis, or Recovery.',
      component: 'MarketRegimeDetector',
      duration: 6000,
      highlight: 'AI-driven market condition analysis'
    },
    {
      id: 'strategy-recommender',
      title: 'AI Strategy Recommender',
      description: 'Personalized investment recommendations based on portfolio metrics and market regime.',
      component: 'StrategyRecommender',
      duration: 7000,
      highlight: 'Intelligent portfolio optimization'
    }
  ]

  // Auto-play functionality
  useEffect(() => {
    if (!isAutoPlaying) return

    const timer = setTimeout(() => {
      setCurrentStep((prev) => (prev + 1) % demoSteps.length)
    }, demoSteps[currentStep].duration)

    return () => clearTimeout(timer)
  }, [currentStep, isAutoPlaying, demoSteps])

  const handlePlay = () => {
    setIsPlaying(true)
    setIsAutoPlaying(true)
  }

  const handlePause = () => {
    setIsPlaying(false)
    setIsAutoPlaying(false)
  }

  const handleReset = () => {
    setIsPlaying(false)
    setIsAutoPlaying(false)
    setCurrentStep(0)
  }

  const handleStepClick = (stepIndex: number) => {
    setCurrentStep(stepIndex)
    setIsAutoPlaying(false)
    setIsPlaying(false)
  }

  return (
    <div className="bg-gray-950 min-h-screen text-white">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-900/50 to-blue-900/50 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="mb-6"
            >
              <div className="flex items-center justify-center space-x-3 mb-4">
                <div className="p-3 bg-gradient-to-br from-purple-600 to-blue-600 rounded-xl">
                  <Sparkles className="h-8 w-8 text-white" />
                </div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
                  BondX AI Analytics
                </h1>
              </div>
              <p className="text-xl text-gray-300 max-w-3xl mx-auto">
                Enterprise-grade AI-powered bond trading and analytics platform with advanced visualizations and intelligent insights
              </p>
            </motion.div>

            {/* Demo Controls */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="flex items-center justify-center space-x-4 mb-8"
            >
              <button
                onClick={isPlaying ? handlePause : handlePlay}
                className="flex items-center space-x-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 rounded-lg font-medium transition-all"
              >
                {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
                <span>{isPlaying ? 'Pause Demo' : 'Start Demo'}</span>
              </button>
              
              <button
                onClick={handleReset}
                className="flex items-center space-x-2 px-6 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg font-medium transition-all"
              >
                <RotateCcw className="h-5 w-5" />
                <span>Reset</span>
              </button>
            </motion.div>

            {/* Progress Bar */}
            <div className="w-full max-w-2xl mx-auto">
              <div className="flex justify-between text-sm text-gray-400 mb-2">
                <span>Demo Progress</span>
                <span>{currentStep + 1} / {demoSteps.length}</span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-2">
                <motion.div
                  className="bg-gradient-to-r from-purple-500 to-blue-500 h-2 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${((currentStep + 1) / demoSteps.length) * 100}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Demo Steps Navigation */}
      <div className="bg-gray-900 border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-center space-x-2 overflow-x-auto">
            {demoSteps.map((step, index) => (
              <motion.button
                key={step.id}
                onClick={() => handleStepClick(index)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap ${
                  currentStep === index
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className={`w-2 h-2 rounded-full ${
                  currentStep === index ? 'bg-white' : 'bg-gray-500'
                }`} />
                <span className="text-sm font-medium">{step.title}</span>
              </motion.button>
            ))}
          </div>
        </div>
      </div>

      {/* Current Step Display */}
      <div className="max-w-7xl mx-auto px-6 py-12">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -50 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <div className="flex items-center justify-center space-x-3 mb-4">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Zap className="h-6 w-6 text-white" />
              </div>
              <h2 className="text-3xl font-bold text-white">
                {demoSteps[currentStep].title}
              </h2>
            </div>
            
            <p className="text-xl text-gray-300 mb-6 max-w-3xl mx-auto">
              {demoSteps[currentStep].description}
            </p>
            
            <div className="inline-flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-green-600/20 to-blue-600/20 rounded-full border border-green-600/30">
              <Target className="h-4 w-4 text-green-400" />
              <span className="text-green-400 font-medium">
                {demoSteps[currentStep].highlight}
              </span>
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {demoSteps.map((step, index) => (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className={`p-6 rounded-xl border-2 transition-all cursor-pointer ${
                currentStep === index
                  ? 'border-blue-500 bg-blue-600/10'
                  : 'border-gray-800 bg-gray-900 hover:border-gray-700'
              }`}
              onClick={() => handleStepClick(index)}
            >
              <div className="flex items-center space-x-3 mb-4">
                <div className={`p-2 rounded-lg ${
                  currentStep === index ? 'bg-blue-600' : 'bg-gray-700'
                }`}>
                  <TrendingUp className="h-5 w-5 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-white">
                  {step.title}
                </h3>
              </div>
              
              <p className="text-gray-300 text-sm mb-4">
                {step.description}
              </p>
              
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-400">
                  Duration: {step.duration / 1000}s
                </span>
                {currentStep === index && (
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                    <span className="text-xs text-blue-400">Active</span>
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Call to Action */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="text-center mt-16"
        >
          <div className="bg-gradient-to-r from-purple-600/20 to-blue-600/20 rounded-2xl p-8 border border-purple-600/30">
            <h3 className="text-2xl font-bold text-white mb-4">
              Ready to Experience BondX?
            </h3>
            <p className="text-gray-300 mb-6 max-w-2xl mx-auto">
              This is just a preview of BondX's capabilities. The full platform includes real-time data feeds, 
              advanced risk management, and institutional-grade trading tools.
            </p>
            <div className="flex items-center justify-center space-x-4">
              <button className="px-8 py-3 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 rounded-lg font-medium transition-all">
                View Live Demo
              </button>
              <button className="px-8 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg font-medium transition-all">
                Learn More
              </button>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default DemoShowcase
