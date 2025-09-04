import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Users, TrendingUp, TrendingDown, Activity, Target, BarChart3, Zap, DollarSign } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

interface FractionalTrade {
  id: string;
  bondId: string;
  bondName: string;
  investorId: string;
  investorType: 'retail' | 'institutional';
  amount: number;
  price: number;
  timestamp: Date;
  status: 'pending' | 'executed' | 'cancelled';
}

interface FractionalOwnership {
  bondId: string;
  bondName: string;
  totalValue: number;
  totalShares: number;
  fractionalShares: number;
  retailOwnership: number;
  institutionalOwnership: number;
  liquidityScore: number;
}

interface FractionalTradingDashboardProps {
  className?: string;
}

const FractionalTradingDashboard: React.FC<FractionalTradingDashboardProps> = ({ className = '' }) => {
  const [trades, setTrades] = useState<FractionalTrade[]>([]);
  const [ownership, setOwnership] = useState<FractionalOwnership[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedBond, setSelectedBond] = useState<string | null>(null);
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d'>('24h');

  // Generate mock fractional trading data
  const generateMockTrades = (): FractionalTrade[] => {
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

    const trades: FractionalTrade[] = [];
    const now = new Date();
    
    for (let i = 0; i < 50; i++) {
      const bondIndex = Math.floor(Math.random() * bondNames.length);
      const investorType = Math.random() > 0.7 ? 'institutional' : 'retail';
      const amount = investorType === 'retail' 
        ? Math.random() * 100000 + 10000  // ₹10K - ₹1L for retail
        : Math.random() * 10000000 + 1000000; // ₹1M - ₹10M for institutional
      
      trades.push({
        id: `trade_${i}`,
        bondId: `bond_${bondIndex}`,
        bondName: bondNames[bondIndex],
        investorId: `${investorType}_${Math.floor(Math.random() * 1000)}`,
        investorType,
        amount,
        price: 95 + Math.random() * 10, // ₹95-105
        timestamp: new Date(now.getTime() - Math.random() * 24 * 60 * 60 * 1000),
        status: Math.random() > 0.1 ? 'executed' : 'pending'
      });
    }
    
    return trades.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  };

  const generateMockOwnership = (): FractionalOwnership[] => {
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
      const totalValue = 100000000 + Math.random() * 500000000; // ₹100M - ₹600M
      const totalShares = Math.floor(totalValue / (95 + Math.random() * 10));
      const fractionalShares = Math.floor(totalShares * (0.1 + Math.random() * 0.3)); // 10-40% fractional
      const retailOwnership = Math.random() * 0.4 + 0.1; // 10-50% retail
      const institutionalOwnership = 1 - retailOwnership;
      
      return {
        bondId: `bond_${index}`,
        bondName: name,
        totalValue,
        totalShares,
        fractionalShares,
        retailOwnership,
        institutionalOwnership,
        liquidityScore: 0.3 + Math.random() * 0.7 // 30-100%
      };
    });
  };

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      await new Promise(resolve => setTimeout(resolve, 1000));
      setTrades(generateMockTrades());
      setOwnership(generateMockOwnership());
      setIsLoading(false);
    };

    loadData();
    
    const interval = setInterval(loadData, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const getTimeRangeData = () => {
    const now = new Date();
    const cutoff = timeRange === '1h' ? 60 * 60 * 1000 :
                   timeRange === '24h' ? 24 * 60 * 60 * 1000 :
                   7 * 24 * 60 * 60 * 1000;
    
    return trades.filter(trade => 
      now.getTime() - trade.timestamp.getTime() <= cutoff
    );
  };

  const filteredTrades = getTimeRangeData();
  const retailTrades = filteredTrades.filter(t => t.investorType === 'retail');
  const institutionalTrades = filteredTrades.filter(t => t.investorType === 'institutional');
  
  const totalVolume = filteredTrades.reduce((sum, trade) => sum + trade.amount, 0);
  const retailVolume = retailTrades.reduce((sum, trade) => sum + trade.amount, 0);
  const institutionalVolume = institutionalTrades.reduce((sum, trade) => sum + trade.amount, 0);

  // Generate hourly volume data for chart
  const generateVolumeData = () => {
    const data = [];
    const now = new Date();
    const hours = timeRange === '1h' ? 12 : timeRange === '24h' ? 24 : 7 * 24;
    
    for (let i = hours - 1; i >= 0; i--) {
      const time = new Date(now.getTime() - i * (timeRange === '7d' ? 24 * 60 * 60 * 1000 : 60 * 60 * 1000));
      const hourTrades = filteredTrades.filter(trade => {
        const tradeTime = trade.timestamp.getTime();
        const hourStart = time.getTime();
        const hourEnd = hourStart + (timeRange === '7d' ? 24 * 60 * 60 * 1000 : 60 * 60 * 1000);
        return tradeTime >= hourStart && tradeTime < hourEnd;
      });
      
      data.push({
        time: timeRange === '7d' ? time.toLocaleDateString() : time.toLocaleTimeString().slice(0, 5),
        retail: hourTrades.filter(t => t.investorType === 'retail').reduce((sum, t) => sum + t.amount, 0),
        institutional: hourTrades.filter(t => t.investorType === 'institutional').reduce((sum, t) => sum + t.amount, 0),
        total: hourTrades.reduce((sum, t) => sum + t.amount, 0)
      });
    }
    
    return data;
  };

  const volumeData = generateVolumeData();

  // Generate ownership distribution data
  const ownershipData = ownership.map(bond => ({
    name: bond.bondName.split(' ').slice(0, 2).join(' '),
    retail: bond.retailOwnership * 100,
    institutional: bond.institutionalOwnership * 100,
    fractional: (bond.fractionalShares / bond.totalShares) * 100
  }));

  const COLORS = ['#22c55e', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6'];

  if (isLoading) {
    return (
      <div className={`bg-gray-900/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 ${className}`}>
        <div className="flex items-center justify-center h-64">
          <div className="flex flex-col items-center space-y-4">
            <div className="w-8 h-8 border-2 border-green-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-gray-400">Loading fractional trading data...</p>
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
          <div className="p-2 bg-green-500/20 rounded-lg">
            <Users className="w-6 h-6 text-green-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-white">Fractional Bond Trading Dashboard</h3>
            <p className="text-sm text-gray-400">Real-time fractional ownership and trading activity</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setTimeRange('1h')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              timeRange === '1h'
                ? 'bg-green-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            1H
          </button>
          <button
            onClick={() => setTimeRange('24h')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              timeRange === '24h'
                ? 'bg-green-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            24H
          </button>
          <button
            onClick={() => setTimeRange('7d')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              timeRange === '7d'
                ? 'bg-green-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            7D
          </button>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <DollarSign className="w-4 h-4 text-green-400" />
            <span className="text-sm text-gray-400">Total Volume</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">₹{(totalVolume / 1000000).toFixed(1)}M</p>
          <p className="text-xs text-gray-500">fractional trades</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Users className="w-4 h-4 text-blue-400" />
            <span className="text-sm text-gray-400">Retail Volume</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">₹{(retailVolume / 1000000).toFixed(1)}M</p>
          <p className="text-xs text-gray-500">{((retailVolume / totalVolume) * 100).toFixed(0)}% of total</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-gray-400">Active Trades</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">{filteredTrades.length}</p>
          <p className="text-xs text-gray-500">in {timeRange}</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Target className="w-4 h-4 text-yellow-400" />
            <span className="text-sm text-gray-400">Avg Trade Size</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">₹{(totalVolume / filteredTrades.length / 1000).toFixed(0)}K</p>
          <p className="text-xs text-gray-500">per trade</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trading Volume Chart */}
        <div className="bg-gray-800/30 rounded-lg p-4">
          <h4 className="text-lg font-semibold text-white mb-4">Trading Volume Over Time</h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={volumeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="time" 
                  stroke="#9ca3af"
                  fontSize={12}
                />
                <YAxis 
                  stroke="#9ca3af"
                  fontSize={12}
                  label={{ value: 'Volume (₹M)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: '#9ca3af' } }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1f2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#f9fafb'
                  }}
                  formatter={(value: number) => [`₹${(value / 1000000).toFixed(1)}M`, 'Volume']}
                />
                <Bar dataKey="retail" stackId="a" fill="#22c55e" name="Retail" />
                <Bar dataKey="institutional" stackId="a" fill="#3b82f6" name="Institutional" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Ownership Distribution */}
        <div className="bg-gray-800/30 rounded-lg p-4">
          <h4 className="text-lg font-semibold text-white mb-4">Fractional Ownership Distribution</h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={ownershipData.slice(0, 5)}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="fractional"
                  nameKey="name"
                >
                  {ownershipData.slice(0, 5).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1f2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#f9fafb'
                  }}
                  formatter={(value: number) => [`${value.toFixed(1)}%`, 'Fractional Ownership']}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Recent Trades */}
      <div className="mt-6">
        <h4 className="text-lg font-semibold text-white mb-4">Recent Fractional Trades</h4>
        <div className="bg-gray-800/30 rounded-lg overflow-hidden">
          <div className="max-h-64 overflow-y-auto">
            <table className="w-full">
              <thead className="bg-gray-700/50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Bond</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Investor</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Amount</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Price</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Time</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-700">
                {filteredTrades.slice(0, 10).map((trade) => (
                  <motion.tr
                    key={trade.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="hover:bg-gray-700/30 transition-colors"
                  >
                    <td className="px-4 py-3 text-sm text-white">
                      {trade.bondName.split(' ').slice(0, 2).join(' ')}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                        trade.investorType === 'retail' 
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-blue-500/20 text-blue-400'
                      }`}>
                        {trade.investorType === 'retail' ? 'Retail' : 'Institutional'}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-white">
                      ₹{(trade.amount / 1000).toFixed(0)}K
                    </td>
                    <td className="px-4 py-3 text-sm text-white">
                      ₹{trade.price.toFixed(2)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-400">
                      {trade.timestamp.toLocaleTimeString()}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                        trade.status === 'executed' 
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-yellow-500/20 text-yellow-400'
                      }`}>
                        {trade.status}
                      </span>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Bond Ownership Details */}
      <div className="mt-6">
        <h4 className="text-lg font-semibold text-white mb-4">Fractional Ownership by Bond</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {ownership.slice(0, 6).map((bond) => (
            <motion.div
              key={bond.bondId}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gray-800/50 rounded-lg p-4"
            >
              <div className="space-y-3">
                <div>
                  <h5 className="text-sm font-medium text-white truncate">{bond.bondName}</h5>
                  <p className="text-xs text-gray-400">Total Value: ₹{(bond.totalValue / 1000000).toFixed(1)}M</p>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Fractional Shares</span>
                    <span className="text-xs font-medium text-white">
                      {((bond.fractionalShares / bond.totalShares) * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Retail Ownership</span>
                    <span className="text-xs font-medium text-green-400">
                      {(bond.retailOwnership * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Liquidity Score</span>
                    <span className={`text-xs font-medium ${
                      bond.liquidityScore >= 0.8 ? 'text-green-400' :
                      bond.liquidityScore >= 0.6 ? 'text-yellow-400' :
                      bond.liquidityScore >= 0.4 ? 'text-orange-400' : 'text-red-400'
                    }`}>
                      {(bond.liquidityScore * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-green-400 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${(bond.fractionalShares / bond.totalShares) * 100}%` }}
                  />
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FractionalTradingDashboard;
