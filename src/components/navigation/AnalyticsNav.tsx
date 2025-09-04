import React from 'react'
import { motion } from 'framer-motion'
import { Link, useLocation } from 'react-router-dom'
import { 
  BarChart3, 
  Brain, 
  Zap, 
  Target, 
  TrendingUp, 
  Activity,
  Sparkles,
  Droplets,
  AlertTriangle
} from 'lucide-react'

const AnalyticsNav: React.FC = () => {
  const location = useLocation()

  const navItems = [
    {
      path: '/dashboard',
      label: 'Dashboard',
      icon: BarChart3,
      description: 'Overview & monitoring'
    },
    {
      path: '/analytics',
      label: 'AI Analytics',
      icon: Brain,
      description: 'Advanced analytics & AI insights',
      isNew: true,
      features: [
        '3D Yield Curve',
        'Shock Simulator', 
        'Liquidity Heatmap',
        'AI Copilot',
        'Market Regime Detector',
        'Strategy Recommender'
      ]
    },
    {
      path: '/trading',
      label: 'Trading',
      icon: TrendingUp,
      description: 'Order management & execution'
    },
    {
      path: '/risk',
      label: 'Risk Management',
      icon: AlertTriangle,
      description: 'Risk monitoring & controls'
    },
    {
      path: '/demo',
      label: 'Live Demo',
      icon: Sparkles,
      description: 'Interactive feature showcase',
      isNew: true
    }
  ]

  return (
    <nav className="bg-gray-900 border-b border-gray-800">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-4">
            <Link to="/" className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-sm">BX</span>
              </div>
              <span className="text-xl font-bold text-white">BondX</span>
            </Link>
          </div>

          {/* Navigation Items */}
          <div className="flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.path
              
              return (
                <motion.div
                  key={item.path}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <Link
                    to={item.path}
                    className={`relative flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                      isActive
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-gray-800'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span className="font-medium">{item.label}</span>
                    
                    {item.isNew && (
                      <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        className="absolute -top-1 -right-1"
                      >
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                      </motion.div>
                    )}
                  </Link>
                </motion.div>
              )
            })}
          </div>

          {/* User Actions */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 px-3 py-1 bg-green-600/20 rounded-full">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-green-400 text-sm font-medium">Live</span>
            </div>
            
            <div className="text-right">
              <div className="text-white text-sm font-medium">Portfolio Value</div>
              <div className="text-gray-400 text-xs">₹125.5L (+2.3%)</div>
            </div>
          </div>
        </div>
      </div>

      {/* Analytics Features Banner */}
      {location.pathname === '/analytics' && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-gradient-to-r from-purple-600/20 to-blue-600/20 border-t border-purple-600/30"
        >
          <div className="max-w-7xl mx-auto px-6 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <Sparkles className="h-4 w-4 text-purple-400" />
                  <span className="text-purple-400 font-medium text-sm">AI-Powered Analytics</span>
                </div>
                
                <div className="flex items-center space-x-6 text-xs text-gray-300">
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-blue-400 rounded-full" />
                    <span>3D Yield Curve</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-red-400 rounded-full" />
                    <span>Shock Simulator</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-green-400 rounded-full" />
                    <span>Liquidity Heatmap</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-purple-400 rounded-full" />
                    <span>AI Copilot</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-yellow-400 rounded-full" />
                    <span>Regime Detector</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-orange-400 rounded-full" />
                    <span>Strategy AI</span>
                  </div>
                </div>
              </div>
              
              <div className="text-xs text-gray-400">
                Real-time updates • ML-powered insights • Institutional-grade analytics
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </nav>
  )
}

export default AnalyticsNav
