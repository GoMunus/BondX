import React from 'react'

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
      {/* Header */}
      <header className="bg-gray-800 shadow-xl">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center">
              <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">B</span>
              </div>
              <h1 className="ml-3 text-2xl font-bold text-white">BondX</h1>
            </div>
            <nav className="hidden md:flex space-x-6">
              <a href="#" className="text-gray-300 hover:text-white px-3 py-2 rounded transition-colors">Dashboard</a>
              <a href="#" className="text-gray-300 hover:text-white px-3 py-2 rounded transition-colors">Trading</a>
              <a href="#" className="text-gray-300 hover:text-white px-3 py-2 rounded transition-colors">Analytics</a>
              <a href="#" className="text-gray-300 hover:text-white px-3 py-2 rounded transition-colors">Risk</a>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="py-16">
        <div className="max-w-7xl mx-auto px-6">
          {/* Hero Section */}
          <div className="text-center mb-20">
            <h1 className="text-6xl font-bold text-white mb-8">
              Enterprise Bond Trading Platform
            </h1>
            <p className="text-xl text-gray-300 mb-10 max-w-4xl mx-auto leading-relaxed">
              Institutional-grade bond trading with AI-powered analytics, real-time risk management, 
              and seamless execution for financial institutions worldwide.
            </p>
            <div className="flex justify-center space-x-6">
              <button className="bg-blue-500 hover:bg-blue-600 text-white px-8 py-4 rounded-lg font-semibold text-lg transition-all transform hover:scale-105">
                Start Trading Now
              </button>
              <button className="border-2 border-gray-400 hover:border-white text-gray-300 hover:text-white px-8 py-4 rounded-lg font-semibold text-lg transition-all">
                View Demo
              </button>
            </div>
          </div>

          {/* Platform Stats */}
          <div className="grid md:grid-cols-4 gap-8 mb-20">
            <div className="text-center">
              <div className="text-4xl font-bold text-blue-400 mb-2">₹2.5T+</div>
              <div className="text-gray-400">Assets Under Management</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-green-400 mb-2">99.99%</div>
              <div className="text-gray-400">System Uptime</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-purple-400 mb-2">500+</div>
              <div className="text-gray-400">Financial Institutions</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-yellow-400 mb-2">&lt;1ms</div>
              <div className="text-gray-400">Execution Latency</div>
            </div>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-10 mb-20">
            <div className="bg-gray-800 p-8 rounded-xl hover:bg-gray-750 transition-colors">
              <div className="w-16 h-16 bg-gradient-to-r from-green-400 to-green-600 rounded-lg flex items-center justify-center mb-6">
                <span className="text-white font-bold text-2xl">📊</span>
              </div>
              <h3 className="text-2xl font-semibold text-white mb-4">AI-Powered Analytics</h3>
              <p className="text-gray-400 leading-relaxed">
                Advanced machine learning algorithms provide real-time market insights, 
                predictive modeling, and intelligent trade recommendations for optimal portfolio performance.
              </p>
            </div>

            <div className="bg-gray-800 p-8 rounded-xl hover:bg-gray-750 transition-colors">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-400 to-blue-600 rounded-lg flex items-center justify-center mb-6">
                <span className="text-white font-bold text-2xl">⚡</span>
              </div>
              <h3 className="text-2xl font-semibold text-white mb-4">Lightning Execution</h3>
              <p className="text-gray-400 leading-relaxed">
                Sub-millisecond order execution with direct market access, institutional-grade 
                infrastructure, and real-time settlement capabilities.
              </p>
            </div>

            <div className="bg-gray-800 p-8 rounded-xl hover:bg-gray-750 transition-colors">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-400 to-purple-600 rounded-lg flex items-center justify-center mb-6">
                <span className="text-white font-bold text-2xl">🛡️</span>
              </div>
              <h3 className="text-2xl font-semibold text-white mb-4">Risk Management</h3>
              <p className="text-gray-400 leading-relaxed">
                Comprehensive portfolio risk analysis, Value-at-Risk calculations, stress testing, 
                and real-time compliance monitoring with regulatory reporting.
              </p>
            </div>
          </div>

          {/* Live Dashboard Preview */}
          <div className="bg-gray-800 rounded-2xl p-10 shadow-2xl">
            <h2 className="text-4xl font-bold text-white mb-10 text-center">
              Live Trading Dashboard
            </h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
              {/* Market Status */}
              <div className="bg-gray-700 p-6 rounded-xl">
                <h4 className="text-lg font-semibold text-white mb-6 border-b border-gray-600 pb-2">Market Status</h4>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">NSE</span>
                    <span className="flex items-center text-green-400">
                      <span className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></span>
                      Open
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">BSE</span>
                    <span className="flex items-center text-green-400">
                      <span className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></span>
                      Open
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">RBI Bonds</span>
                    <span className="flex items-center text-blue-400">
                      <span className="w-2 h-2 bg-blue-400 rounded-full mr-2 animate-pulse"></span>
                      Active
                    </span>
                  </div>
                </div>
              </div>

              {/* Portfolio Summary */}
              <div className="bg-gray-700 p-6 rounded-xl">
                <h4 className="text-lg font-semibold text-white mb-6 border-b border-gray-600 pb-2">Portfolio Value</h4>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Total AUM</span>
                    <span className="text-white font-bold text-lg">₹2.47T</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Today's P&L</span>
                    <span className="text-green-400 font-semibold">+₹142.3M</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Active Positions</span>
                    <span className="text-white font-semibold">1,247</span>
                  </div>
                </div>
              </div>

              {/* Risk Metrics */}
              <div className="bg-gray-700 p-6 rounded-xl">
                <h4 className="text-lg font-semibold text-white mb-6 border-b border-gray-600 pb-2">Risk Metrics</h4>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">VaR (95%)</span>
                    <span className="text-yellow-400 font-semibold">₹67.2M</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Duration</span>
                    <span className="text-white font-semibold">4.8 years</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Avg Rating</span>
                    <span className="text-green-400 font-semibold">AA+</span>
                  </div>
                </div>
              </div>

              {/* Recent Activity */}
              <div className="bg-gray-700 p-6 rounded-xl">
                <h4 className="text-lg font-semibold text-white mb-6 border-b border-gray-600 pb-2">Live Trades</h4>
                <div className="space-y-4">
                  <div className="text-sm">
                    <div className="text-white font-medium">IRFC 7.30% 2029</div>
                    <div className="text-green-400">Bought ₹85M</div>
                    <div className="text-gray-500">2 min ago</div>
                  </div>
                  <div className="text-sm">
                    <div className="text-white font-medium">NTPC 6.25% 2027</div>
                    <div className="text-red-400">Sold ₹45M</div>
                    <div className="text-gray-500">5 min ago</div>
                  </div>
                  <div className="text-sm">
                    <div className="text-white font-medium">SBI 8.75% 2032</div>
                    <div className="text-green-400">Bought ₹120M</div>
                    <div className="text-gray-500">8 min ago</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* System Status */}
          <div className="mt-20 text-center">
            <div className="inline-flex items-center space-x-6 bg-gray-800 px-8 py-4 rounded-xl">
              <div className="flex items-center space-x-3">
                <div className="w-4 h-4 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-white font-semibold text-lg">All Systems Operational</span>
              </div>
              <div className="text-gray-400">|</div>
              <div className="text-gray-400">
                Last Updated: {new Date().toLocaleTimeString()}
              </div>
            </div>
            <p className="text-gray-500 mt-6 text-lg">
              BondX Enterprise Platform v3.0 | Licensed for Institutional Use
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 mt-20 border-t border-gray-700">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <div className="text-center">
            <div className="flex justify-center items-center mb-4">
              <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center mr-3">
                <span className="text-white font-bold">B</span>
              </div>
              <span className="text-xl font-bold text-white">BondX</span>
            </div>
            <p className="text-gray-400 mb-4">
              © 2024 BondX Technologies. Enterprise Bond Trading Platform. All rights reserved.
            </p>
            <p className="text-gray-500 text-sm">
              Regulated by SEBI | Member of NSE, BSE | ISO 27001 Certified
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
