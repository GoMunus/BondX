import React from 'react'

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900">
      {/* Header */}
      <header className="bg-slate-800 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">B</span>
              </div>
              <h1 className="ml-3 text-2xl font-bold text-white">BondX</h1>
            </div>
            <nav className="hidden md:flex space-x-8">
              <a href="#" className="text-gray-300 hover:text-white transition-colors">Dashboard</a>
              <a href="#" className="text-gray-300 hover:text-white transition-colors">Trading</a>
              <a href="#" className="text-gray-300 hover:text-white transition-colors">Analytics</a>
              <a href="#" className="text-gray-300 hover:text-white transition-colors">Risk</a>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Hero Section */}
          <div className="text-center mb-16">
            <h1 className="text-5xl font-bold text-white mb-6">
              Next-Generation Bond Trading Platform
            </h1>
            <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
              Institutional-grade bond trading with AI-powered analytics, 
              real-time risk management, and seamless execution.
            </p>
            <div className="flex justify-center space-x-4">
              <button className="bg-blue-500 hover:bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold transition-colors">
                Start Trading
              </button>
              <button className="border border-gray-400 hover:border-white text-gray-300 hover:text-white px-8 py-3 rounded-lg font-semibold transition-colors">
                Learn More
              </button>
            </div>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-8 mb-16">
            <div className="bg-slate-800 p-8 rounded-xl">
              <div className="w-12 h-12 bg-green-500 rounded-lg flex items-center justify-center mb-4">
                <span className="text-white font-bold">📊</span>
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">Real-time Analytics</h3>
              <p className="text-gray-400">
                Advanced market analysis with AI-powered insights and predictive modeling
                for optimal trading decisions.
              </p>
            </div>

            <div className="bg-slate-800 p-8 rounded-xl">
              <div className="w-12 h-12 bg-blue-500 rounded-lg flex items-center justify-center mb-4">
                <span className="text-white font-bold">⚡</span>
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">Lightning Fast Execution</h3>
              <p className="text-gray-400">
                Sub-millisecond order execution with direct market access and 
                institutional-grade infrastructure.
              </p>
            </div>

            <div className="bg-slate-800 p-8 rounded-xl">
              <div className="w-12 h-12 bg-purple-500 rounded-lg flex items-center justify-center mb-4">
                <span className="text-white font-bold">🛡️</span>
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">Advanced Risk Management</h3>
              <p className="text-gray-400">
                Comprehensive portfolio risk analysis, VaR calculations, and 
                real-time compliance monitoring.
              </p>
            </div>
          </div>

          {/* Dashboard Preview */}
          <div className="bg-slate-800 rounded-xl p-8">
            <h2 className="text-3xl font-bold text-white mb-8 text-center">
              Platform Overview
            </h2>
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Market Status */}
              <div className="bg-slate-700 p-6 rounded-lg">
                <h4 className="text-lg font-semibold text-white mb-4">Market Status</h4>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">NSE</span>
                    <span className="text-green-400">🟢 Open</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">BSE</span>
                    <span className="text-green-400">🟢 Open</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">RBI</span>
                    <span className="text-blue-400">🔵 Active</span>
                  </div>
                </div>
              </div>

              {/* Portfolio Summary */}
              <div className="bg-slate-700 p-6 rounded-lg">
                <h4 className="text-lg font-semibold text-white mb-4">Portfolio</h4>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Total Value</span>
                    <span className="text-white font-semibold">₹2.5B</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">P&L Today</span>
                    <span className="text-green-400">+₹12.3M</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Active Bonds</span>
                    <span className="text-white">247</span>
                  </div>
                </div>
              </div>

              {/* Risk Metrics */}
              <div className="bg-slate-700 p-6 rounded-lg">
                <h4 className="text-lg font-semibold text-white mb-4">Risk Metrics</h4>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">VaR (95%)</span>
                    <span className="text-yellow-400">₹45.2M</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Duration</span>
                    <span className="text-white">4.2 years</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Credit Rating</span>
                    <span className="text-green-400">AA+</span>
                  </div>
                </div>
              </div>

              {/* Recent Activity */}
              <div className="bg-slate-700 p-6 rounded-lg">
                <h4 className="text-lg font-semibold text-white mb-4">Recent Trades</h4>
                <div className="space-y-3">
                  <div className="text-sm">
                    <div className="text-white">IRFC 7.30% 2029</div>
                    <div className="text-green-400">Bought ₹50M</div>
                  </div>
                  <div className="text-sm">
                    <div className="text-white">NTPC 6.25% 2027</div>
                    <div className="text-red-400">Sold ₹25M</div>
                  </div>
                  <div className="text-sm">
                    <div className="text-white">SBI 8.75% 2032</div>
                    <div className="text-green-400">Bought ₹75M</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* System Status */}
          <div className="mt-16 text-center">
            <div className="inline-flex items-center space-x-4 bg-slate-800 px-6 py-3 rounded-lg">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                <span className="text-white font-semibold">System Status: All Systems Operational</span>
              </div>
            </div>
            <p className="text-gray-400 mt-4">
              BondX Platform v2.0 | Last updated: {new Date().toLocaleString()}
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-slate-800 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center">
            <p className="text-gray-400">
              © 2024 BondX. Enterprise Bond Trading Platform. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
