import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, Target, Brain, Zap, AlertCircle, CheckCircle, XCircle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';

interface SpreadPrediction {
  isin: string;
  name: string;
  currentSpread: number;
  predictedSpread1h: number;
  predictedSpread4h: number;
  predictedSpread24h: number;
  confidence: number;
  modelConsensus: {
    gradientBoosting: number;
    lstm: number;
    temporalCNN: number;
  };
  valuation: 'undervalued' | 'fair' | 'overvalued';
  riskFactors: string[];
  recommendation: string;
}

interface SpreadForecastingProps {
  className?: string;
}

const SpreadForecasting: React.FC<SpreadForecastingProps> = ({ className = '' }) => {
  const [predictions, setPredictions] = useState<SpreadPrediction[]>([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState<'1h' | '4h' | '24h'>('4h');
  const [isLoading, setIsLoading] = useState(true);
  const [selectedBond, setSelectedBond] = useState<string | null>(null);
  const [historicalData, setHistoricalData] = useState<any[]>([]);

  // Generate mock spread predictions
  const generateMockPredictions = (): SpreadPrediction[] => {
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
      const currentSpread = 50 + Math.random() * 200; // 50-250 bps
      const volatility = 0.1 + Math.random() * 0.3; // 10-40% volatility
      
      const predictedSpread1h = currentSpread + (Math.random() - 0.5) * currentSpread * volatility * 0.1;
      const predictedSpread4h = predictedSpread1h + (Math.random() - 0.5) * currentSpread * volatility * 0.2;
      const predictedSpread24h = predictedSpread4h + (Math.random() - 0.5) * currentSpread * volatility * 0.3;
      
      const confidence = 0.6 + Math.random() * 0.4;
      
      const modelConsensus = {
        gradientBoosting: predictedSpread24h + (Math.random() - 0.5) * 10,
        lstm: predictedSpread24h + (Math.random() - 0.5) * 15,
        temporalCNN: predictedSpread24h + (Math.random() - 0.5) * 12
      };
      
      const avgPrediction = (modelConsensus.gradientBoosting + modelConsensus.lstm + modelConsensus.temporalCNN) / 3;
      const valuation = avgPrediction < currentSpread * 0.95 ? 'undervalued' : 
                       avgPrediction > currentSpread * 1.05 ? 'overvalued' : 'fair';
      
      const riskFactors = [];
      if (volatility > 0.3) riskFactors.push('High volatility');
      if (Math.abs(predictedSpread24h - currentSpread) > currentSpread * 0.2) riskFactors.push('Large spread movement expected');
      if (confidence < 0.7) riskFactors.push('Low model confidence');
      
      return {
        isin: `INE${String(index).padStart(9, '0')}A`,
        name: name,
        currentSpread,
        predictedSpread1h,
        predictedSpread4h,
        predictedSpread24h,
        confidence,
        modelConsensus,
        valuation,
        riskFactors,
        recommendation: valuation === 'undervalued' 
          ? 'Consider buying - spread likely to tighten'
          : valuation === 'overvalued'
          ? 'Consider selling - spread likely to widen'
          : 'Hold position - fair valuation expected'
      };
    });
  };

  // Generate historical data for charts
  const generateHistoricalData = () => {
    const data = [];
    const now = new Date();
    
    for (let i = 23; i >= 0; i--) {
      const time = new Date(now.getTime() - i * 60 * 60 * 1000); // Last 24 hours
      data.push({
        time: time.toISOString().substr(11, 5),
        actual: 100 + Math.sin(i * 0.5) * 20 + Math.random() * 10,
        predicted: 100 + Math.sin(i * 0.5) * 20 + Math.random() * 8,
        confidence: 0.7 + Math.random() * 0.3
      });
    }
    
    return data;
  };

  useEffect(() => {
    const loadPredictions = async () => {
      setIsLoading(true);
      await new Promise(resolve => setTimeout(resolve, 1200));
      setPredictions(generateMockPredictions());
      setHistoricalData(generateHistoricalData());
      setIsLoading(false);
    };

    loadPredictions();
    
    const interval = setInterval(loadPredictions, 30000);
    return () => clearInterval(interval);
  }, []);

  const getValuationIcon = (valuation: string) => {
    switch (valuation) {
      case 'undervalued': return <TrendingUp className="w-4 h-4 text-green-400" />;
      case 'overvalued': return <TrendingDown className="w-4 h-4 text-red-400" />;
      default: return <Target className="w-4 h-4 text-blue-400" />;
    }
  };

  const getValuationColor = (valuation: string) => {
    switch (valuation) {
      case 'undervalued': return 'text-green-400 bg-green-400/20';
      case 'overvalued': return 'text-red-400 bg-red-400/20';
      default: return 'text-blue-400 bg-blue-400/20';
    }
  };

  const sortedPredictions = useMemo(() => {
    return [...predictions].sort((a, b) => {
      const aSpread = selectedTimeframe === '1h' ? a.predictedSpread1h : 
                     selectedTimeframe === '4h' ? a.predictedSpread4h : a.predictedSpread24h;
      const bSpread = selectedTimeframe === '1h' ? b.predictedSpread1h : 
                     selectedTimeframe === '4h' ? b.predictedSpread4h : b.predictedSpread24h;
      return Math.abs(aSpread - a.currentSpread) - Math.abs(bSpread - b.currentSpread);
    });
  }, [predictions, selectedTimeframe]);

  const undervaluedBonds = predictions.filter(p => p.valuation === 'undervalued').length;
  const overvaluedBonds = predictions.filter(p => p.valuation === 'overvalued').length;
  const avgConfidence = predictions.reduce((acc, p) => acc + p.confidence, 0) / predictions.length;

  if (isLoading) {
    return (
      <div className={`bg-gray-900/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 ${className}`}>
        <div className="flex items-center justify-center h-64">
          <div className="flex flex-col items-center space-y-4">
            <div className="w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-gray-400">Loading spread predictions...</p>
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
          <div className="p-2 bg-purple-500/20 rounded-lg">
            <Brain className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-white">Dynamic Spread Forecasting</h3>
            <p className="text-sm text-gray-400">Ensemble ML models with confidence intervals</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setSelectedTimeframe('1h')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              selectedTimeframe === '1h'
                ? 'bg-purple-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            1H
          </button>
          <button
            onClick={() => setSelectedTimeframe('4h')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              selectedTimeframe === '4h'
                ? 'bg-purple-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            4H
          </button>
          <button
            onClick={() => setSelectedTimeframe('24h')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              selectedTimeframe === '24h'
                ? 'bg-purple-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            24H
          </button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <CheckCircle className="w-4 h-4 text-green-400" />
            <span className="text-sm text-gray-400">Undervalued</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">{undervaluedBonds}</p>
          <p className="text-xs text-gray-500">opportunities</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <XCircle className="w-4 h-4 text-red-400" />
            <span className="text-sm text-gray-400">Overvalued</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">{overvaluedBonds}</p>
          <p className="text-xs text-gray-500">risks</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Brain className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-gray-400">Avg Confidence</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">{Math.round(avgConfidence * 100)}%</p>
          <p className="text-xs text-gray-500">model accuracy</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Zap className="w-4 h-4 text-yellow-400" />
            <span className="text-sm text-gray-400">Active Models</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">3</p>
          <p className="text-xs text-gray-500">ensemble</p>
        </div>
      </div>

      {/* Model Performance Chart */}
      <div className="mb-6">
        <h4 className="text-lg font-semibold text-white mb-4">Model Performance (24H)</h4>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={historicalData}>
              <defs>
                <linearGradient id="actualGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="predictedGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0}/>
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
                label={{ value: 'Spread (bps)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: '#9ca3af' } }}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1f2937', 
                  border: '1px solid #374151',
                  borderRadius: '8px',
                  color: '#f9fafb'
                }}
              />
              <Area
                type="monotone"
                dataKey="actual"
                stroke="#3b82f6"
                fillOpacity={1}
                fill="url(#actualGradient)"
                strokeWidth={2}
                name="Actual"
              />
              <Area
                type="monotone"
                dataKey="predicted"
                stroke="#8b5cf6"
                fillOpacity={1}
                fill="url(#predictedGradient)"
                strokeWidth={2}
                name="Predicted"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Spread Predictions Grid */}
      <div className="mb-6">
        <h4 className="text-lg font-semibold text-white mb-4">Spread Predictions</h4>
        <div className="grid grid-cols-2 gap-3">
          {sortedPredictions.slice(0, 8).map((prediction, index) => {
            const predictedSpread = selectedTimeframe === '1h' ? prediction.predictedSpread1h :
                                  selectedTimeframe === '4h' ? prediction.predictedSpread4h : prediction.predictedSpread24h;
            const change = predictedSpread - prediction.currentSpread;
            const changePercent = (change / prediction.currentSpread) * 100;
            
            return (
              <motion.div
                key={prediction.isin}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                className={`p-4 rounded-lg border cursor-pointer transition-all hover:scale-105 ${
                  selectedBond === prediction.isin 
                    ? 'border-purple-500 bg-purple-500/10' 
                    : 'border-gray-700 bg-gray-800/30'
                }`}
                onClick={() => setSelectedBond(selectedBond === prediction.isin ? null : prediction.isin)}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    {getValuationIcon(prediction.valuation)}
                    <span className="text-sm font-medium text-white truncate">
                      {prediction.name.split(' ').slice(0, 2).join(' ')}
                    </span>
                  </div>
                  <div className={`px-2 py-1 rounded text-xs font-medium ${getValuationColor(prediction.valuation)}`}>
                    {prediction.valuation}
                  </div>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Current</span>
                    <span className="text-sm font-medium text-white">
                      {prediction.currentSpread.toFixed(0)} bps
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Predicted</span>
                    <span className={`text-sm font-medium ${
                      change > 0 ? 'text-red-400' : 'text-green-400'
                    }`}>
                      {predictedSpread.toFixed(0)} bps
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Change</span>
                    <span className={`text-sm font-medium ${
                      change > 0 ? 'text-red-400' : 'text-green-400'
                    }`}>
                      {change > 0 ? '+' : ''}{changePercent.toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Confidence</span>
                    <span className="text-sm font-medium text-purple-400">
                      {Math.round(prediction.confidence * 100)}%
                    </span>
                  </div>
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
              
              return (
                <div>
                  <h4 className="text-lg font-semibold text-white mb-4">
                    {prediction.name}
                  </h4>
                  
                  <div className="grid grid-cols-2 gap-6">
                    {/* Model Consensus */}
                    <div>
                      <h5 className="text-sm font-medium text-gray-400 mb-3">Model Consensus</h5>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-300">Gradient Boosting</span>
                          <span className="text-sm font-medium text-white">
                            {prediction.modelConsensus.gradientBoosting.toFixed(0)} bps
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-300">LSTM</span>
                          <span className="text-sm font-medium text-white">
                            {prediction.modelConsensus.lstm.toFixed(0)} bps
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-gray-300">Temporal CNN</span>
                          <span className="text-sm font-medium text-white">
                            {prediction.modelConsensus.temporalCNN.toFixed(0)} bps
                          </span>
                        </div>
                        <div className="flex justify-between items-center border-t border-gray-700 pt-2">
                          <span className="text-sm font-medium text-gray-300">Ensemble Average</span>
                          <span className="text-sm font-bold text-purple-400">
                            {((prediction.modelConsensus.gradientBoosting + 
                               prediction.modelConsensus.lstm + 
                               prediction.modelConsensus.temporalCNN) / 3).toFixed(0)} bps
                          </span>
                        </div>
                      </div>
                    </div>
                    
                    {/* Risk Factors */}
                    <div>
                      <h5 className="text-sm font-medium text-gray-400 mb-3">Risk Factors</h5>
                      <div className="space-y-2">
                        {prediction.riskFactors.length > 0 ? (
                          prediction.riskFactors.map((factor, index) => (
                            <div key={index} className="flex items-center space-x-2">
                              <AlertCircle className="w-4 h-4 text-yellow-400" />
                              <span className="text-sm text-gray-300">{factor}</span>
                            </div>
                          ))
                        ) : (
                          <div className="flex items-center space-x-2">
                            <CheckCircle className="w-4 h-4 text-green-400" />
                            <span className="text-sm text-gray-300">No significant risk factors</span>
                          </div>
                        )}
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

export default SpreadForecasting;
