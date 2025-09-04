import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, TrendingDown, Activity, Target, Zap, AlertTriangle, CheckCircle, Clock, BarChart3 } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, ComposedChart, Area, AreaChart } from 'recharts';

interface BondLiquidityScore {
  bondId: string;
  bondName: string;
  currentBLS: number;
  previousBLS: number;
  trend: 'improving' | 'stable' | 'declining';
  confidence: number;
  factors: LiquidityFactor[];
  predictedSpreadChange: number;
  spreadChangeConfidence: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  lastUpdated: Date;
  historicalBLS: BLSDataPoint[];
}

interface LiquidityFactor {
  name: string;
  weight: number;
  currentValue: number;
  impact: 'positive' | 'negative' | 'neutral';
  description: string;
}

interface BLSDataPoint {
  timestamp: Date;
  bls: number;
  volume: number;
  spread: number;
  volatility: number;
}

interface BondLiquidityScoreProps {
  className?: string;
}

const BondLiquidityScore: React.FC<BondLiquidityScoreProps> = ({ className = '' }) => {
  const [bonds, setBonds] = useState<BondLiquidityScore[]>([]);
  const [selectedBond, setSelectedBond] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d' | '30d'>('24h');
  const [sortBy, setSortBy] = useState<'bls' | 'trend' | 'risk' | 'spread'>('bls');

  // Generate mock BLS data
  const generateMockBLS = (): BondLiquidityScore[] => {
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

    const trends: ('improving' | 'stable' | 'declining')[] = ['improving', 'stable', 'declining'];
    const riskLevels: ('low' | 'medium' | 'high' | 'critical')[] = ['low', 'medium', 'high', 'critical'];

    return bondNames.map((name, index) => {
      const currentBLS = 20 + Math.random() * 80; // 20-100
      const previousBLS = currentBLS + (Math.random() - 0.5) * 20;
      const trend = currentBLS > previousBLS ? 'improving' : 
                   Math.abs(currentBLS - previousBLS) < 5 ? 'stable' : 'declining';
      const confidence = 0.7 + Math.random() * 0.3; // 70-100%
      const predictedSpreadChange = (Math.random() - 0.5) * 50; // -25 to +25 bps
      const spreadChangeConfidence = 0.6 + Math.random() * 0.4; // 60-100%
      
      let riskLevel: 'low' | 'medium' | 'high' | 'critical';
      if (currentBLS >= 80) riskLevel = 'low';
      else if (currentBLS >= 60) riskLevel = 'medium';
      else if (currentBLS >= 40) riskLevel = 'high';
      else riskLevel = 'critical';

      // Generate liquidity factors
      const factors: LiquidityFactor[] = [
        {
          name: 'Trading Volume',
          weight: 0.3,
          currentValue: 0.3 + Math.random() * 0.7,
          impact: Math.random() > 0.3 ? 'positive' : 'negative',
          description: 'Recent trading activity and volume patterns'
        },
        {
          name: 'Bid-Ask Spread',
          weight: 0.25,
          currentValue: 0.1 + Math.random() * 0.9,
          impact: Math.random() > 0.5 ? 'positive' : 'negative',
          description: 'Tightness of bid-ask spreads'
        },
        {
          name: 'Market Depth',
          weight: 0.2,
          currentValue: 0.2 + Math.random() * 0.8,
          impact: Math.random() > 0.4 ? 'positive' : 'negative',
          description: 'Order book depth and market maker presence'
        },
        {
          name: 'Price Volatility',
          weight: 0.15,
          currentValue: 0.1 + Math.random() * 0.9,
          impact: Math.random() > 0.6 ? 'negative' : 'positive',
          description: 'Price stability and volatility measures'
        },
        {
          name: 'Credit Rating',
          weight: 0.1,
          currentValue: 0.4 + Math.random() * 0.6,
          impact: Math.random() > 0.2 ? 'positive' : 'negative',
          description: 'Credit quality and rating stability'
        }
      ];

      // Generate historical BLS data
      const historicalBLS: BLSDataPoint[] = [];
      const now = new Date();
      for (let i = 30; i >= 0; i--) {
        const time = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
        const bls = currentBLS + (Math.random() - 0.5) * 20;
        const volume = 1000000 + Math.random() * 10000000;
        const spread = 0.1 + Math.random() * 2;
        const volatility = 0.05 + Math.random() * 0.3;
        
        historicalBLS.push({
          timestamp: time,
          bls: Math.max(0, Math.min(100, bls)),
          volume,
          spread,
          volatility
        });
      }

      return {
        bondId: `bond_${index}`,
        bondName: name,
        currentBLS,
        previousBLS,
        trend,
        confidence,
        factors,
        predictedSpreadChange,
        spreadChangeConfidence,
        riskLevel,
        lastUpdated: new Date(),
        historicalBLS
      };
    });
  };

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      await new Promise(resolve => setTimeout(resolve, 1000));
      setBonds(generateMockBLS());
      setIsLoading(false);
    };

    loadData();
    
    const interval = setInterval(loadData, 20000); // Update every 20 seconds
    return () => clearInterval(interval);
  }, []);

  const sortedBonds = [...bonds].sort((a, b) => {
    switch (sortBy) {
      case 'bls':
        return b.currentBLS - a.currentBLS;
      case 'trend':
        const trendOrder = { 'improving': 3, 'stable': 2, 'declining': 1 };
        return trendOrder[b.trend] - trendOrder[a.trend];
      case 'risk':
        const riskOrder = { 'low': 4, 'medium': 3, 'high': 2, 'critical': 1 };
        return riskOrder[b.riskLevel] - riskOrder[a.riskLevel];
      case 'spread':
        return Math.abs(b.predictedSpreadChange) - Math.abs(a.predictedSpreadChange);
      default:
        return 0;
    }
  });

  const avgBLS = bonds.reduce((sum, bond) => sum + bond.currentBLS, 0) / bonds.length;
  const improvingBonds = bonds.filter(b => b.trend === 'improving').length;
  const highRiskBonds = bonds.filter(b => b.riskLevel === 'high' || b.riskLevel === 'critical').length;
  const avgSpreadChange = bonds.reduce((sum, bond) => sum + bond.predictedSpreadChange, 0) / bonds.length;

  const selectedBondData = bonds.find(b => b.bondId === selectedBond);

  // Generate BLS distribution data
  const generateDistributionData = () => {
    const ranges = [
      { range: '80-100', count: bonds.filter(b => b.currentBLS >= 80).length, color: '#22c55e' },
      { range: '60-79', count: bonds.filter(b => b.currentBLS >= 60 && b.currentBLS < 80).length, color: '#3b82f6' },
      { range: '40-59', count: bonds.filter(b => b.currentBLS >= 40 && b.currentBLS < 60).length, color: '#f59e0b' },
      { range: '20-39', count: bonds.filter(b => b.currentBLS >= 20 && b.currentBLS < 40).length, color: '#ef4444' },
      { range: '0-19', count: bonds.filter(b => b.currentBLS < 20).length, color: '#dc2626' }
    ];
    return ranges;
  };

  const distributionData = generateDistributionData();

  if (isLoading) {
    return (
      <div className={`bg-gray-900/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 ${className}`}>
        <div className="flex items-center justify-center h-64">
          <div className="flex flex-col items-center space-y-4">
            <div className="w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-gray-400">Loading Bond Liquidity Scores...</p>
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
            <BarChart3 className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-white">Bond Liquidity Score (BLS)</h3>
            <p className="text-sm text-gray-400">Real-time liquidity quantification with spread predictions</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="bg-gray-700 text-white px-3 py-1 rounded-lg text-sm border border-gray-600"
          >
            <option value="bls">Sort by BLS</option>
            <option value="trend">Sort by Trend</option>
            <option value="risk">Sort by Risk</option>
            <option value="spread">Sort by Spread Change</option>
          </select>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Target className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-gray-400">Average BLS</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">{avgBLS.toFixed(1)}</p>
          <p className="text-xs text-gray-500">out of 100</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-4 h-4 text-green-400" />
            <span className="text-sm text-gray-400">Improving</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">{improvingBonds}</p>
          <p className="text-xs text-gray-500">bonds trending up</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 text-red-400" />
            <span className="text-sm text-gray-400">High Risk</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">{highRiskBonds}</p>
          <p className="text-xs text-gray-500">bonds need attention</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-yellow-400" />
            <span className="text-sm text-gray-400">Avg Spread Change</span>
          </div>
          <p className={`text-2xl font-bold mt-1 ${avgSpreadChange >= 0 ? 'text-red-400' : 'text-green-400'}`}>
            {avgSpreadChange >= 0 ? '+' : ''}{avgSpreadChange.toFixed(1)}bps
          </p>
          <p className="text-xs text-gray-500">predicted change</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* BLS Distribution */}
        <div className="bg-gray-800/30 rounded-lg p-4">
          <h4 className="text-lg font-semibold text-white mb-4">BLS Score Distribution</h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distributionData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="range" 
                  stroke="#9ca3af"
                  fontSize={12}
                />
                <YAxis 
                  stroke="#9ca3af"
                  fontSize={12}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1f2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#f9fafb'
                  }}
                  formatter={(value: number) => [`${value} bonds`, 'Count']}
                />
                <Bar dataKey="count" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Risk Level Overview */}
        <div className="bg-gray-800/30 rounded-lg p-4">
          <h4 className="text-lg font-semibold text-white mb-4">Risk Level Distribution</h4>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-sm text-gray-300">Low Risk</span>
              </div>
              <span className="text-sm font-medium text-white">
                {bonds.filter(b => b.riskLevel === 'low').length}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="text-sm text-gray-300">Medium Risk</span>
              </div>
              <span className="text-sm font-medium text-white">
                {bonds.filter(b => b.riskLevel === 'medium').length}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <span className="text-sm text-gray-300">High Risk</span>
              </div>
              <span className="text-sm font-medium text-white">
                {bonds.filter(b => b.riskLevel === 'high').length}
              </span>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                <span className="text-sm text-gray-300">Critical Risk</span>
              </div>
              <span className="text-sm font-medium text-white">
                {bonds.filter(b => b.riskLevel === 'critical').length}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Bond List */}
      <div className="mt-6">
        <h4 className="text-lg font-semibold text-white mb-4">Bond Liquidity Scores</h4>
        <div className="bg-gray-800/30 rounded-lg overflow-hidden">
          <div className="max-h-96 overflow-y-auto">
            <table className="w-full">
              <thead className="bg-gray-700/50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Bond</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">BLS Score</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Trend</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Risk Level</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Spread Change</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Confidence</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {sortedBonds.map((bond) => (
                  <motion.tr
                    key={bond.bondId}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="hover:bg-gray-700/30 transition-colors"
                  >
                    <td className="px-4 py-3 text-sm text-white">
                      {bond.bondName.split(' ').slice(0, 2).join(' ')}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <div className="flex items-center space-x-2">
                        <span className="text-white font-medium">{bond.currentBLS.toFixed(1)}</span>
                        <div className="w-16 bg-gray-700 rounded-full h-2">
                          <div 
                            className={`h-2 rounded-full transition-all duration-500 ${
                              bond.currentBLS >= 80 ? 'bg-green-400' :
                              bond.currentBLS >= 60 ? 'bg-blue-400' :
                              bond.currentBLS >= 40 ? 'bg-yellow-400' : 'bg-red-400'
                            }`}
                            style={{ width: `${bond.currentBLS}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                        bond.trend === 'improving' ? 'bg-green-500/20 text-green-400' :
                        bond.trend === 'stable' ? 'bg-blue-500/20 text-blue-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {bond.trend === 'improving' ? <TrendingUp className="w-3 h-3 mr-1" /> :
                         bond.trend === 'declining' ? <TrendingDown className="w-3 h-3 mr-1" /> :
                         <Activity className="w-3 h-3 mr-1" />}
                        {bond.trend}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                        bond.riskLevel === 'low' ? 'bg-green-500/20 text-green-400' :
                        bond.riskLevel === 'medium' ? 'bg-blue-500/20 text-blue-400' :
                        bond.riskLevel === 'high' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {bond.riskLevel}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span className={`font-medium ${
                        bond.predictedSpreadChange >= 0 ? 'text-red-400' : 'text-green-400'
                      }`}>
                        {bond.predictedSpreadChange >= 0 ? '+' : ''}{bond.predictedSpreadChange.toFixed(1)}bps
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span className="text-white">
                        {(bond.spreadChangeConfidence * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <button
                        onClick={() => setSelectedBond(bond.bondId)}
                        className="text-blue-400 hover:text-blue-300 text-xs font-medium"
                      >
                        View Details
                      </button>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Selected Bond Details */}
      {selectedBondData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 bg-gray-800/30 rounded-lg p-4"
        >
          <h4 className="text-lg font-semibold text-white mb-4">
            BLS Analysis: {selectedBondData.bondName}
          </h4>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Historical BLS Chart */}
            <div>
              <h5 className="text-sm font-medium text-gray-300 mb-3">BLS Historical Trend</h5>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={selectedBondData.historicalBLS}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="timestamp" 
                      stroke="#9ca3af"
                      fontSize={10}
                      tickFormatter={(time) => new Date(time).toLocaleDateString()}
                    />
                    <YAxis 
                      stroke="#9ca3af"
                      fontSize={10}
                      domain={[0, 100]}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1f2937', 
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        color: '#f9fafb'
                      }}
                      formatter={(value: number) => [`${value.toFixed(1)}`, 'BLS Score']}
                      labelFormatter={(time) => new Date(time).toLocaleDateString()}
                    />
                    <Area 
                      type="monotone" 
                      dataKey="bls" 
                      stroke="#8b5cf6" 
                      fill="#8b5cf6"
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Liquidity Factors */}
            <div>
              <h5 className="text-sm font-medium text-gray-300 mb-3">Liquidity Factors</h5>
              <div className="space-y-3">
                {selectedBondData.factors.map((factor, index) => (
                  <div key={index} className="bg-gray-700/30 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm text-white">{factor.name}</span>
                      <span className={`text-xs font-medium ${
                        factor.impact === 'positive' ? 'text-green-400' :
                        factor.impact === 'negative' ? 'text-red-400' : 'text-gray-400'
                      }`}>
                        {factor.impact}
                      </span>
                    </div>
                    <div className="text-xs text-gray-400 mb-2">{factor.description}</div>
                    <div className="flex items-center space-x-2">
                      <div className="w-full bg-gray-600 rounded-full h-2">
                        <div 
                          className={`h-2 rounded-full transition-all duration-500 ${
                            factor.impact === 'positive' ? 'bg-green-400' :
                            factor.impact === 'negative' ? 'bg-red-400' : 'bg-gray-400'
                          }`}
                          style={{ width: `${factor.currentValue * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-white">{(factor.currentValue * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default BondLiquidityScore;
