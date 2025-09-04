import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, AlertTriangle, Clock, Activity, BarChart3 } from 'lucide-react';
import * as d3 from 'd3';

interface LiquidityPrediction {
  isin: string;
  name: string;
  currentLiquidity: number;
  predictedLiquidity24h: number;
  predictedLiquidity72h: number;
  confidence: number;
  riskLevel: 'low' | 'medium' | 'high';
  factors: {
    volumeTrend: number;
    spreadChange: number;
    ratingStability: number;
    macroImpact: number;
  };
  recommendation: string;
}

interface LiquidityForecastingProps {
  className?: string;
}

const LiquidityForecasting: React.FC<LiquidityForecastingProps> = ({ className = '' }) => {
  const [predictions, setPredictions] = useState<LiquidityPrediction[]>([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'24h' | '72h'>('24h');
  const [isLoading, setIsLoading] = useState(true);
  const [selectedBond, setSelectedBond] = useState<string | null>(null);

  // Generate mock liquidity predictions based on real bond data
  const generateMockPredictions = (): LiquidityPrediction[] => {
    const bondNames = [
      'HINDUSTAN PETROLEUM CORPORATION LIMITED',
      'SMALL INDUSTRIES DEVELOPMENT BANK OF INDIA',
      'ONGC PETRO ADDITIONS LIMITED',
      'GMR AIRPORTS LIMITED',
      'INDIAN OIL CORPORATION LIMITED',
      'NATIONAL BANK FOR AGRICULTURE AND RURAL DEVELOPMENT',
      'MUTHOOT FINANCE LIMITED',
      'POWER FINANCE CORPORATION LIMITED',
      'JSW STEEL LIMITED',
      'HDFC BANK LIMITED'
    ];

    return bondNames.map((name, index) => {
      const currentLiquidity = Math.random() * 100;
      const trend = (Math.random() - 0.5) * 0.3; // -15% to +15% change
      const predictedLiquidity24h = Math.max(0, Math.min(100, currentLiquidity + trend * currentLiquidity));
      const predictedLiquidity72h = Math.max(0, Math.min(100, predictedLiquidity24h + (Math.random() - 0.5) * 0.2 * predictedLiquidity24h));
      
      const confidence = 0.7 + Math.random() * 0.3;
      const riskLevel = predictedLiquidity72h < 30 ? 'high' : predictedLiquidity72h < 60 ? 'medium' : 'low';
      
      return {
        isin: `INE${String(index).padStart(9, '0')}A`,
        name: name,
        currentLiquidity,
        predictedLiquidity24h,
        predictedLiquidity72h,
        confidence,
        riskLevel,
        factors: {
          volumeTrend: (Math.random() - 0.5) * 2,
          spreadChange: (Math.random() - 0.5) * 0.5,
          ratingStability: Math.random(),
          macroImpact: (Math.random() - 0.5) * 0.3
        },
        recommendation: riskLevel === 'high' 
          ? 'Consider reducing position or hedging with liquid alternatives'
          : riskLevel === 'medium'
          ? 'Monitor closely, consider partial position adjustment'
          : 'Maintain current position, favorable liquidity outlook'
      };
    });
  };

  useEffect(() => {
    const loadPredictions = async () => {
      setIsLoading(true);
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      setPredictions(generateMockPredictions());
      setIsLoading(false);
    };

    loadPredictions();
    
    // Refresh predictions every 30 seconds
    const interval = setInterval(loadPredictions, 30000);
    return () => clearInterval(interval);
  }, []);

  const getLiquidityColor = (liquidity: number) => {
    if (liquidity >= 80) return 'text-green-400 bg-green-400/20';
    if (liquidity >= 60) return 'text-yellow-400 bg-yellow-400/20';
    if (liquidity >= 40) return 'text-orange-400 bg-orange-400/20';
    return 'text-red-400 bg-red-400/20';
  };

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'high': return <AlertTriangle className="w-4 h-4 text-red-400" />;
      case 'medium': return <Clock className="w-4 h-4 text-yellow-400" />;
      default: return <Activity className="w-4 h-4 text-green-400" />;
    }
  };

  const sortedPredictions = useMemo(() => {
    return [...predictions].sort((a, b) => {
      const aLiquidity = selectedTimeframe === '24h' ? a.predictedLiquidity24h : a.predictedLiquidity72h;
      const bLiquidity = selectedTimeframe === '24h' ? b.predictedLiquidity24h : b.predictedLiquidity72h;
      return bLiquidity - aLiquidity;
    });
  }, [predictions, selectedTimeframe]);

  const highRiskBonds = predictions.filter(p => p.riskLevel === 'high').length;
  const improvingBonds = predictions.filter(p => {
    const predicted = selectedTimeframe === '24h' ? p.predictedLiquidity24h : p.predictedLiquidity72h;
    return predicted > p.currentLiquidity;
  }).length;

  if (isLoading) {
    return (
      <div className={`bg-gray-900/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 ${className}`}>
        <div className="flex items-center justify-center h-64">
          <div className="flex flex-col items-center space-y-4">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-gray-400">Loading liquidity predictions...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-900/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-500/20 rounded-lg">
            <BarChart3 className="w-6 h-6 text-blue-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-white">Autonomous Liquidity Forecasting</h3>
            <p className="text-sm text-gray-400">AI-powered 24-72 hour liquidity predictions</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setSelectedTimeframe('24h')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              selectedTimeframe === '24h'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            24H
          </button>
          <button
            onClick={() => setSelectedTimeframe('72h')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              selectedTimeframe === '72h'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            72H
          </button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-4 h-4 text-green-400" />
            <span className="text-sm text-gray-400">Improving</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">{improvingBonds}</p>
          <p className="text-xs text-gray-500">bonds</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-red-400" />
            <span className="text-sm text-gray-400">High Risk</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">{highRiskBonds}</p>
          <p className="text-xs text-gray-500">bonds</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-blue-400" />
            <span className="text-sm text-gray-400">Avg Confidence</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">
            {Math.round(predictions.reduce((acc, p) => acc + p.confidence, 0) / predictions.length * 100)}%
          </p>
          <p className="text-xs text-gray-500">prediction accuracy</p>
        </div>
      </div>

      {/* Liquidity Heatmap */}
      <div className="mb-6">
        <h4 className="text-lg font-semibold text-white mb-4">Liquidity Heatmap</h4>
        <div className="grid grid-cols-2 gap-2">
          {sortedPredictions.slice(0, 8).map((prediction, index) => {
            const predictedLiquidity = selectedTimeframe === '24h' 
              ? prediction.predictedLiquidity24h 
              : prediction.predictedLiquidity72h;
            const change = predictedLiquidity - prediction.currentLiquidity;
            
            return (
              <motion.div
                key={prediction.isin}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                className={`p-3 rounded-lg border cursor-pointer transition-all hover:scale-105 ${
                  selectedBond === prediction.isin 
                    ? 'border-blue-500 bg-blue-500/10' 
                    : 'border-gray-700 bg-gray-800/30'
                }`}
                onClick={() => setSelectedBond(selectedBond === prediction.isin ? null : prediction.isin)}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    {getRiskIcon(prediction.riskLevel)}
                    <span className="text-sm font-medium text-white truncate">
                      {prediction.name.split(' ').slice(0, 2).join(' ')}
                    </span>
                  </div>
                  <div className="flex items-center space-x-1">
                    {change > 0 ? (
                      <TrendingUp className="w-3 h-3 text-green-400" />
                    ) : (
                      <TrendingDown className="w-3 h-3 text-red-400" />
                    )}
                    <span className={`text-xs font-medium ${
                      change > 0 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {change > 0 ? '+' : ''}{change.toFixed(1)}%
                    </span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className="w-16 h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-500 ${
                          predictedLiquidity >= 80 ? 'bg-green-400' :
                          predictedLiquidity >= 60 ? 'bg-yellow-400' :
                          predictedLiquidity >= 40 ? 'bg-orange-400' : 'bg-red-400'
                        }`}
                        style={{ width: `${predictedLiquidity}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-400">
                      {predictedLiquidity.toFixed(0)}%
                    </span>
                  </div>
                  <span className="text-xs text-gray-500">
                    {Math.round(prediction.confidence * 100)}%
                  </span>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Detailed View */}
      <AnimatePresence>
        {selectedBond && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-t border-gray-700 pt-6"
          >
            {(() => {
              const prediction = predictions.find(p => p.isin === selectedBond);
              if (!prediction) return null;
              
              const predictedLiquidity = selectedTimeframe === '24h' 
                ? prediction.predictedLiquidity24h 
                : prediction.predictedLiquidity72h;
              
              return (
                <div>
                  <h4 className="text-lg font-semibold text-white mb-4">
                    {prediction.name}
                  </h4>
                  
                  <div className="grid grid-cols-2 gap-6">
                    {/* Liquidity Metrics */}
                    <div>
                      <h5 className="text-sm font-medium text-gray-400 mb-3">Liquidity Metrics</h5>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-300">Current</span>
                          <span className="text-sm font-medium text-white">
                            {prediction.currentLiquidity.toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-300">Predicted ({selectedTimeframe})</span>
                          <span className={`text-sm font-medium ${
                            predictedLiquidity >= 80 ? 'text-green-400' :
                            predictedLiquidity >= 60 ? 'text-yellow-400' :
                            predictedLiquidity >= 40 ? 'text-orange-400' : 'text-red-400'
                          }`}>
                            {predictedLiquidity.toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-300">Confidence</span>
                          <span className="text-sm font-medium text-blue-400">
                            {Math.round(prediction.confidence * 100)}%
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    {/* Contributing Factors */}
                    <div>
                      <h5 className="text-sm font-medium text-gray-400 mb-3">Contributing Factors</h5>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-300">Volume Trend</span>
                          <span className={`text-sm font-medium ${
                            prediction.factors.volumeTrend > 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {prediction.factors.volumeTrend > 0 ? '+' : ''}{(prediction.factors.volumeTrend * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-300">Spread Change</span>
                          <span className={`text-sm font-medium ${
                            prediction.factors.spreadChange < 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {prediction.factors.spreadChange > 0 ? '+' : ''}{(prediction.factors.spreadChange * 100).toFixed(1)}bps
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-300">Rating Stability</span>
                          <span className="text-sm font-medium text-blue-400">
                            {Math.round(prediction.factors.ratingStability * 100)}%
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-300">Macro Impact</span>
                          <span className={`text-sm font-medium ${
                            prediction.factors.macroImpact > 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {prediction.factors.macroImpact > 0 ? '+' : ''}{(prediction.factors.macroImpact * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Recommendation */}
                  <div className="mt-6 p-4 bg-gray-800/50 rounded-lg">
                    <h5 className="text-sm font-medium text-gray-400 mb-2">AI Recommendation</h5>
                    <p className="text-sm text-gray-300">{prediction.recommendation}</p>
                  </div>
                </div>
              );
            })()}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default LiquidityForecasting;
