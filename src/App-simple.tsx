import React from 'react'

const App: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4">🚀 BondX Platform</h1>
        <p className="text-xl text-gray-300 mb-8">
          Institutional Bond Trading Platform
        </p>
        <div className="p-6 bg-gray-800 rounded-lg">
          <h2 className="text-2xl font-semibold mb-4">System Status</h2>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span>Frontend:</span>
              <span className="text-green-400">✅ Running</span>
            </div>
            <div className="flex justify-between">
              <span>React:</span>
              <span className="text-green-400">✅ Loaded</span>
            </div>
            <div className="flex justify-between">
              <span>TypeScript:</span>
              <span className="text-green-400">✅ Compiled</span>
            </div>
            <div className="flex justify-between">
              <span>Tailwind CSS:</span>
              <span className="text-green-400">✅ Active</span>
            </div>
          </div>
        </div>
        <div className="mt-8">
          <button className="bg-blue-500 hover:bg-blue-600 px-6 py-3 rounded-lg font-semibold transition-colors">
            Access Trading Platform
          </button>
        </div>
      </div>
    </div>
  )
}

export default App
