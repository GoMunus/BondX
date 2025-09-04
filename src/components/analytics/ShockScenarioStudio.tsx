import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, AlertTriangle, Zap, Target, BarChart3, RotateCcw, Play, Pause } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

interface ShockEvent {
  id: string;
  type: 'repo_rate' | 'inflation' | 'rating' | 'liquidity' | 'credit_spread';
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  magnitude: number;
  duration: number; // in days
  impact: {
    duration: number;
    convexity: number;
    pnl: number;
    liquidity: number;
  };
}

interface PortfolioMetrics {
  totalValue: number;
  duration: number;
  convexity: number;
  avgYield: number;
  liquidityScore: number;
  var95: number;
  var99: number;
}

interface ShockScenarioStudioProps {
  className?: string;
  onMetricsUpdate?: (metrics: PortfolioMetrics) => void;
}

const ShockScenarioStudio: React.FC<ShockScenarioStudioProps> = ({ 
  className = '',
  onMetricsUpdate
}) => {
  const [shockEvents, setShockEvents] = useState<ShockEvent[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [portfolioMetrics, setPortfolioMetrics] = useState<PortfolioMetrics>({
    totalValue: 125000000,
    duration: 4.2,
    convexity: 12.3,
    avgYield: 7.2,
    liquidityScore: 0.68,
    var95: 2500000,
    var99: 3200000
  });
  const [historicalMetrics, setHistoricalMetrics] = useState<PortfolioMetrics[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [dragPreview, setDragPreview] = useState<ShockEvent | null>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);

  // Available shock events
  const availableShocks: ShockEvent[] = [
    {
      id: 'repo_hike_25',
      type: 'repo_rate',
      name: 'Repo Rate +25bps',
      description: 'RBI increases repo rate by 25 basis points',
      icon: <TrendingUp className="w-4 h-4" />,
      color: 'text-yellow-400',
      magnitude: 25,
      duration: 30,
      impact: { duration: 0.1, convexity: 0.2, pnl: -0.025, liquidity: -0.05 }
    },
    {
      id: 'repo_hike_50',
      type: 'repo_rate',
      name: 'Repo Rate +50bps',
      description: 'RBI increases repo rate by 50 basis points',
      icon: <TrendingUp className="w-4 h-4" />,
      color: 'text-orange-400',
      magnitude: 50,
      duration: 30,
      impact: { duration: 0.2, convexity: 0.4, pnl: -0.05, liquidity: -0.1 }
    },
    {
      id: 'inflation_shock',
      type: 'inflation',
      name: 'Inflation Shock',
      description: 'CPI rises 2% above target',
      icon: <AlertTriangle className="w-4 h-4" />,
      color: 'text-red-400',
      magnitude: 2,
      duration: 60,
      impact: { duration: 0.15, convexity: 0.3, pnl: -0.03, liquidity: -0.08 }
    },
    {
      id: 'rating_downgrade',
      type: 'rating',
      name: 'Rating Downgrade',
      description: 'Widespread corporate rating downgrades',
      icon: <TrendingDown className="w-4 h-4" />,
      color: 'text-red-400',
      magnitude: 1,
      duration: 45,
      impact: { duration: 0.05, convexity: 0.1, pnl: -0.02, liquidity: -0.15 }
    },
    {
      id: 'liquidity_crisis',
      type: 'liquidity',
      name: 'Liquidity Crisis',
      description: 'Market liquidity drops by 50%',
      icon: <Zap className="w-4 h-4" />,
      color: 'text-red-400',
      magnitude: 50,
      duration: 20,
      impact: { duration: 0.02, convexity: 0.05, pnl: -0.01, liquidity: -0.5 }
    },
    {
      id: 'credit_spread_widen',
      type: 'credit_spread',
      name: 'Credit Spread Widening',
      description: 'Corporate spreads widen by 100bps',
      icon: <BarChart3 className="w-4 h-4" />,
      color: 'text-orange-400',
      magnitude: 100,
      duration: 40,
      impact: { duration: 0.08, convexity: 0.15, pnl: -0.04, liquidity: -0.12 }
    }
  ];

  // Simulate portfolio metrics over time
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setCurrentTime(prev => {
        const newTime = prev + 1;
        
        // Calculate cumulative impact of active shock events
        const activeShocks = shockEvents.filter(shock => 
          newTime >= 0 && newTime <= shock.duration
        );

        let totalImpact = {
          duration: 0,
          convexity: 0,
          pnl: 0,
          liquidity: 0
        };

        activeShocks.forEach(shock => {
          // Decay impact over time
          const timeDecay = Math.max(0, 1 - (newTime / shock.duration));
          totalImpact.duration += shock.impact.duration * timeDecay;
          totalImpact.convexity += shock.impact.convexity * timeDecay;
          totalImpact.pnl += shock.impact.pnl * timeDecay;
          totalImpact.liquidity += shock.impact.liquidity * timeDecay;
        });

        // Update portfolio metrics
        const newMetrics: PortfolioMetrics = {
          totalValue: portfolioMetrics.totalValue * (1 + totalImpact.pnl),
          duration: Math.max(0, portfolioMetrics.duration + totalImpact.duration),
          convexity: Math.max(0, portfolioMetrics.convexity + totalImpact.convexity),
          avgYield: portfolioMetrics.avgYield + (totalImpact.duration * 0.1),
          liquidityScore: Math.max(0, Math.min(1, portfolioMetrics.liquidityScore + totalImpact.liquidity)),
          var95: portfolioMetrics.var95 * (1 + totalImpact.pnl * 0.8),
          var99: portfolioMetrics.var99 * (1 + totalImpact.pnl * 0.9)
        };

        setPortfolioMetrics(newMetrics);
        setHistoricalMetrics(prev => [...prev.slice(-29), newMetrics]);
        
        if (onMetricsUpdate) {
          onMetricsUpdate(newMetrics);
        }

        return newTime;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [isPlaying, shockEvents, portfolioMetrics, onMetricsUpdate]);

  const handleDragStart = (event: React.DragEvent, shock: ShockEvent) => {
    setIsDragging(true);
    setDragPreview(shock);
    event.dataTransfer.setData('application/json', JSON.stringify(shock));
  };

  const handleDragEnd = () => {
    setIsDragging(false);
    setDragPreview(null);
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const shockData = JSON.parse(event.dataTransfer.getData('application/json'));
    
    // Add shock event at current time
    const newShock: ShockEvent = {
      ...shockData,
      id: `${shockData.id}_${Date.now()}`
    };
    
    setShockEvents(prev => [...prev, newShock]);
    setIsDragging(false);
    setDragPreview(null);
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const removeShockEvent = (shockId: string) => {
    setShockEvents(prev => prev.filter(shock => shock.id !== shockId));
  };

  const resetSimulation = () => {
    setShockEvents([]);
    setCurrentTime(0);
    setIsPlaying(false);
    setPortfolioMetrics({
      totalValue: 125000000,
      duration: 4.2,
      convexity: 12.3,
      avgYield: 7.2,
      liquidityScore: 0.68,
      var95: 2500000,
      var99: 3200000
    });
    setHistoricalMetrics([]);
  };

  const togglePlayback = () => {
    setIsPlaying(!isPlaying);
  };

  const formatCurrency = (value: number) => {
    return `₹${(value / 1000000).toFixed(1)}M`;
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div className={`bg-gray-900/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-orange-500/20 rounded-lg">
            <Target className="w-6 h-6 text-orange-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-white">Shock Scenario Studio</h3>
            <p className="text-sm text-gray-400">Drag-and-drop macro shocks with instant portfolio updates</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={resetSimulation}
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-white transition-colors"
            title="Reset Simulation"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <button
            onClick={togglePlayback}
            className={`p-2 rounded-lg text-white transition-colors ${
              isPlaying ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'
            }`}
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Shock Library */}
        <div className="lg:col-span-1">
          <h4 className="text-lg font-semibold text-white mb-4">Shock Library</h4>
          <div className="space-y-2">
            {availableShocks.map((shock) => (
              <motion.div
                key={shock.id}
                draggable
                onDragStart={(e) => handleDragStart(e as React.DragEvent, shock)}
                onDragEnd={handleDragEnd}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="p-3 bg-gray-800/50 rounded-lg border border-gray-700 cursor-grab hover:border-gray-600 transition-colors"
              >
                <div className="flex items-center space-x-3">
                  <div className={shock.color}>
                    {shock.icon}
                  </div>
                  <div className="flex-1">
                    <h5 className="text-sm font-medium text-white">{shock.name}</h5>
                    <p className="text-xs text-gray-400">{shock.description}</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Simulation Area */}
        <div className="lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-semibold text-white">Simulation Timeline</h4>
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-400">Time:</span>
              <span className="text-sm font-mono text-white">{currentTime}s</span>
            </div>
          </div>

          {/* Drop Zone */}
          <div
            ref={dropZoneRef}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            className={`min-h-32 p-4 rounded-lg border-2 border-dashed transition-colors ${
              isDragging
                ? 'border-orange-400 bg-orange-400/10'
                : 'border-gray-600 bg-gray-800/30'
            }`}
          >
            {shockEvents.length === 0 ? (
              <div className="flex items-center justify-center h-24">
                <div className="text-center">
                  <DragDrop className="w-8 h-8 text-gray-500 mx-auto mb-2" />
                  <p className="text-gray-400 text-sm">Drag shock events here to simulate</p>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                {shockEvents.map((shock, index) => (
                  <motion.div
                    key={shock.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg"
                  >
                    <div className="flex items-center space-x-3">
                      <div className={shock.color}>
                        {shock.icon}
                      </div>
                      <div>
                        <h5 className="text-sm font-medium text-white">{shock.name}</h5>
                        <p className="text-xs text-gray-400">Duration: {shock.duration}s</p>
                      </div>
                    </div>
                    <button
                      onClick={() => removeShockEvent(shock.id)}
                      className="p-1 text-gray-400 hover:text-red-400 transition-colors"
                    >
                      ×
                    </button>
                  </motion.div>
                ))}
              </div>
            )}
          </div>

          {/* Portfolio Metrics */}
          <div className="mt-6">
            <h4 className="text-lg font-semibold text-white mb-4">Portfolio Impact</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <TrendingUp className="w-4 h-4 text-blue-400" />
                  <span className="text-sm text-gray-400">Total Value</span>
                </div>
                <p className="text-xl font-bold text-white">{formatCurrency(portfolioMetrics.totalValue)}</p>
                <p className={`text-xs ${
                  portfolioMetrics.totalValue >= 125000000 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {portfolioMetrics.totalValue >= 125000000 ? '+' : ''}
                  {formatPercentage((portfolioMetrics.totalValue - 125000000) / 125000000)}
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <Target className="w-4 h-4 text-orange-400" />
                  <span className="text-sm text-gray-400">Duration</span>
                </div>
                <p className="text-xl font-bold text-white">{portfolioMetrics.duration.toFixed(2)}Y</p>
                <p className={`text-xs ${
                  portfolioMetrics.duration <= 4.2 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {portfolioMetrics.duration <= 4.2 ? '' : '+'}
                  {(portfolioMetrics.duration - 4.2).toFixed(2)}Y
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <BarChart3 className="w-4 h-4 text-green-400" />
                  <span className="text-sm text-gray-400">Liquidity</span>
                </div>
                <p className="text-xl font-bold text-white">{formatPercentage(portfolioMetrics.liquidityScore)}</p>
                <p className={`text-xs ${
                  portfolioMetrics.liquidityScore >= 0.68 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {portfolioMetrics.liquidityScore >= 0.68 ? '+' : ''}
                  {formatPercentage(portfolioMetrics.liquidityScore - 0.68)}
                </p>
              </div>

              <div className="bg-gray-800/50 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <AlertTriangle className="w-4 h-4 text-red-400" />
                  <span className="text-sm text-gray-400">VaR 95%</span>
                </div>
                <p className="text-xl font-bold text-white">{formatCurrency(portfolioMetrics.var95)}</p>
                <p className={`text-xs ${
                  portfolioMetrics.var95 <= 2500000 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {portfolioMetrics.var95 <= 2500000 ? '' : '+'}
                  {formatPercentage((portfolioMetrics.var95 - 2500000) / 2500000)}
                </p>
              </div>
            </div>
          </div>

          {/* Historical Chart */}
          {historicalMetrics.length > 0 && (
            <div className="mt-6">
              <h4 className="text-lg font-semibold text-white mb-4">Portfolio Value Over Time</h4>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={historicalMetrics.map((metrics, index) => ({
                    time: index,
                    value: metrics.totalValue / 1000000,
                    duration: metrics.duration,
                    liquidity: metrics.liquidityScore * 100
                  }))}>
                    <defs>
                      <linearGradient id="valueGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="time" 
                      stroke="#9ca3af"
                      fontSize={12}
                    />
                    <YAxis 
                      stroke="#9ca3af"
                      fontSize={12}
                      label={{ value: 'Value (₹M)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: '#9ca3af' } }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1f2937', 
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        color: '#f9fafb'
                      }}
                      formatter={(value: number) => [`₹${value.toFixed(1)}M`, 'Portfolio Value']}
                    />
                    <Area
                      type="monotone"
                      dataKey="value"
                      stroke="#3b82f6"
                      fillOpacity={1}
                      fill="url(#valueGradient)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ShockScenarioStudio;
