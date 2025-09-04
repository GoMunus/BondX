import React, { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Droplets, Activity, TrendingUp, AlertCircle } from 'lucide-react'
import * as d3 from 'd3'

interface BondLiquidity {
  isin: string
  name: string
  maturity: number
  yield: number
  liquidityScore: number // 0-1
  volume: number
  spread: number
  lastTrade: string
  sector: string
  rating: string
}

interface LiquidityHeatmapProps {
  data: BondLiquidity[]
  onBondSelect?: (bond: BondLiquidity) => void
  className?: string
}

interface HeatmapCell {
  x: number
  y: number
  bond: BondLiquidity
  color: string
  intensity: number
}

const LiquidityHeatmap: React.FC<LiquidityHeatmapProps> = ({
  data,
  onBondSelect,
  className = ''
}) => {
  const [selectedBond, setSelectedBond] = useState<BondLiquidity | null>(null)
  const [hoveredBond, setHoveredBond] = useState<BondLiquidity | null>(null)
  const [sortBy, setSortBy] = useState<'liquidity' | 'volume' | 'spread'>('liquidity')
  const [filterSector, setFilterSector] = useState<string>('all')
  const [isLive, setIsLive] = useState(true)

  // Color scale for liquidity
  const colorScale = useMemo(() => {
    return d3.scaleSequential(d3.interpolateRdYlGn)
      .domain([0, 1]) // 0 = illiquid (red), 1 = liquid (green)
  }, [])

  // Process data for heatmap
  const heatmapData = useMemo(() => {
    if (!data.length) return []

    // Filter by sector
    const filteredData = filterSector === 'all' 
      ? data 
      : data.filter(bond => bond.sector === filterSector)

    // Sort data
    const sortedData = [...filteredData].sort((a, b) => {
      switch (sortBy) {
        case 'liquidity':
          return b.liquidityScore - a.liquidityScore
        case 'volume':
          return b.volume - a.volume
        case 'spread':
          return a.spread - b.spread
        default:
          return 0
      }
    })

    // Create grid layout (10 columns)
    const cols = 10
    const rows = Math.ceil(sortedData.length / cols)

    return sortedData.map((bond, index) => {
      const x = index % cols
      const y = Math.floor(index / cols)
      
      return {
        x,
        y,
        bond,
        color: colorScale(bond.liquidityScore),
        intensity: bond.liquidityScore
      }
    })
  }, [data, sortBy, filterSector, colorScale])

  // Get unique sectors
  const sectors = useMemo(() => {
    const uniqueSectors = [...new Set(data.map(bond => bond.sector))]
    return ['all', ...uniqueSectors]
  }, [data])

  // Simulate live updates
  useEffect(() => {
    if (!isLive) return

    const interval = setInterval(() => {
      // Simulate real-time updates by slightly modifying liquidity scores
      // In a real app, this would come from WebSocket updates
    }, 2000)

    return () => clearInterval(interval)
  }, [isLive])

  // Calculate aggregate statistics
  const stats = useMemo(() => {
    if (!data.length) return null

    const avgLiquidity = data.reduce((sum, bond) => sum + bond.liquidityScore, 0) / data.length
    const totalVolume = data.reduce((sum, bond) => sum + bond.volume, 0)
    const avgSpread = data.reduce((sum, bond) => sum + bond.spread, 0) / data.length
    const liquidBonds = data.filter(bond => bond.liquidityScore > 0.7).length
    const illiquidBonds = data.filter(bond => bond.liquidityScore < 0.3).length

    return {
      avgLiquidity,
      totalVolume,
      avgSpread,
      liquidBonds,
      illiquidBonds,
      totalBonds: data.length
    }
  }, [data])

  const handleCellClick = (bond: BondLiquidity) => {
    setSelectedBond(bond)
    onBondSelect?.(bond)
  }

  const handleCellHover = (bond: BondLiquidity | null) => {
    setHoveredBond(bond)
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
            <div className="p-2 bg-blue-600 rounded-lg">
              <Droplets className="h-6 w-6 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-white">Liquidity Heatmap</h3>
              <p className="text-gray-400 text-sm">Real-time bond liquidity visualization</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${
              isLive ? 'bg-green-600/20' : 'bg-gray-600/20'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                isLive ? 'bg-green-400 animate-pulse' : 'bg-gray-400'
              }`} />
              <span className={`text-sm ${
                isLive ? 'text-green-400' : 'text-gray-400'
              }`}>
                {isLive ? 'Live' : 'Paused'}
              </span>
            </div>
            
            <button
              onClick={() => setIsLive(!isLive)}
              className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded-md text-sm text-white transition-colors"
            >
              {isLive ? 'Pause' : 'Resume'}
            </button>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="p-6 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div>
              <label className="text-gray-300 text-sm mb-1 block">Sort by</label>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as any)}
                className="px-3 py-1 bg-gray-800 border border-gray-700 rounded-md text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="liquidity">Liquidity Score</option>
                <option value="volume">Trading Volume</option>
                <option value="spread">Bid-Ask Spread</option>
              </select>
            </div>
            
            <div>
              <label className="text-gray-300 text-sm mb-1 block">Sector</label>
              <select
                value={filterSector}
                onChange={(e) => setFilterSector(e.target.value)}
                className="px-3 py-1 bg-gray-800 border border-gray-700 rounded-md text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                {sectors.map(sector => (
                  <option key={sector} value={sector}>
                    {sector === 'all' ? 'All Sectors' : sector}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Statistics */}
          {stats && (
            <div className="flex items-center space-x-6 text-sm">
              <div className="text-center">
                <div className="text-white font-medium">{stats.liquidBonds}</div>
                <div className="text-green-400 text-xs">Liquid</div>
              </div>
              <div className="text-center">
                <div className="text-white font-medium">{stats.illiquidBonds}</div>
                <div className="text-red-400 text-xs">Illiquid</div>
              </div>
              <div className="text-center">
                <div className="text-white font-medium">{(stats.avgLiquidity * 100).toFixed(0)}%</div>
                <div className="text-gray-400 text-xs">Avg Liquidity</div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Heatmap Grid */}
      <div className="p-6">
        <div className="relative">
          <div className="grid grid-cols-10 gap-1 max-w-4xl mx-auto">
            {heatmapData.map((cell, index) => (
              <motion.div
                key={cell.bond.isin}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.01 }}
                whileHover={{ scale: 1.1, zIndex: 10 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => handleCellClick(cell.bond)}
                onMouseEnter={() => handleCellHover(cell.bond)}
                onMouseLeave={() => handleCellHover(null)}
                className="w-8 h-8 rounded cursor-pointer border border-gray-700 hover:border-white transition-all relative"
                style={{ backgroundColor: cell.color }}
                title={`${cell.bond.name} - Liquidity: ${(cell.intensity * 100).toFixed(0)}%`}
              >
                {/* Intensity indicator */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className={`w-1 h-1 rounded-full ${
                    cell.intensity > 0.7 ? 'bg-white' : 
                    cell.intensity > 0.3 ? 'bg-gray-300' : 'bg-gray-600'
                  }`} />
                </div>
              </motion.div>
            ))}
          </div>

          {/* Legend */}
          <div className="mt-6 flex items-center justify-center space-x-4">
            <span className="text-gray-400 text-sm">Illiquid</span>
            <div className="flex space-x-1">
              {[0, 0.25, 0.5, 0.75, 1].map((value) => (
                <div
                  key={value}
                  className="w-4 h-4 rounded border border-gray-700"
                  style={{ backgroundColor: colorScale(value) }}
                />
              ))}
            </div>
            <span className="text-gray-400 text-sm">Liquid</span>
          </div>
        </div>
      </div>

      {/* Bond Details Panel */}
      <AnimatePresence>
        {(selectedBond || hoveredBond) && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="border-t border-gray-800 p-6"
          >
            <div className="bg-gray-800 rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-white font-medium">
                  {(selectedBond || hoveredBond)?.name}
                </h4>
                <div className={`px-2 py-1 rounded text-xs font-medium ${
                  ((selectedBond || hoveredBond)?.liquidityScore || 0) > 0.7 
                    ? 'bg-green-600 text-white' 
                    : ((selectedBond || hoveredBond)?.liquidityScore || 0) > 0.3
                    ? 'bg-yellow-600 text-white'
                    : 'bg-red-600 text-white'
                }`}>
                  {((selectedBond || hoveredBond)?.liquidityScore || 0) > 0.7 ? 'Liquid' :
                   ((selectedBond || hoveredBond)?.liquidityScore || 0) > 0.3 ? 'Moderate' : 'Illiquid'}
                </div>
              </div>
              
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <div className="text-gray-400">ISIN</div>
                  <div className="text-white font-mono">{(selectedBond || hoveredBond)?.isin}</div>
                </div>
                <div>
                  <div className="text-gray-400">Maturity</div>
                  <div className="text-white">{(selectedBond || hoveredBond)?.maturity}Y</div>
                </div>
                <div>
                  <div className="text-gray-400">Yield</div>
                  <div className="text-white">{(selectedBond || hoveredBond)?.yield.toFixed(2)}%</div>
                </div>
                <div>
                  <div className="text-gray-400">Liquidity Score</div>
                  <div className="text-white">{((selectedBond || hoveredBond)?.liquidityScore || 0 * 100).toFixed(0)}%</div>
                </div>
                <div>
                  <div className="text-gray-400">Volume</div>
                  <div className="text-white">₹{((selectedBond || hoveredBond)?.volume || 0 / 100000).toFixed(0)}L</div>
                </div>
                <div>
                  <div className="text-gray-400">Spread</div>
                  <div className="text-white">{(selectedBond || hoveredBond)?.spread}bps</div>
                </div>
                <div>
                  <div className="text-gray-400">Sector</div>
                  <div className="text-white">{(selectedBond || hoveredBond)?.sector}</div>
                </div>
                <div>
                  <div className="text-gray-400">Rating</div>
                  <div className="text-white">{(selectedBond || hoveredBond)?.rating}</div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

export default LiquidityHeatmap
