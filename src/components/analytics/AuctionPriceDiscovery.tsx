import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Gavel, TrendingUp, TrendingDown, Clock, Users, DollarSign, Activity, Target, Zap } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter } from 'recharts';

interface Auction {
  id: string;
  bondId: string;
  bondName: string;
  auctionType: 'mini' | 'batch' | 'continuous';
  status: 'pending' | 'active' | 'completed' | 'cancelled';
  startTime: Date;
  endTime: Date;
  initialPrice: number;
  currentPrice: number;
  targetVolume: number;
  currentVolume: number;
  buyOrders: Order[];
  sellOrders: Order[];
  priceHistory: PricePoint[];
  participants: number;
}

interface Order {
  id: string;
  type: 'buy' | 'sell';
  amount: number;
  price: number;
  timestamp: Date;
  participantId: string;
  participantType: 'retail' | 'institutional' | 'market_maker';
}

interface PricePoint {
  timestamp: Date;
  price: number;
  volume: number;
  participants: number;
}

interface AuctionPriceDiscoveryProps {
  className?: string;
}

const AuctionPriceDiscovery: React.FC<AuctionPriceDiscoveryProps> = ({ className = '' }) => {
  const [auctions, setAuctions] = useState<Auction[]>([]);
  const [selectedAuction, setSelectedAuction] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [timeRange, setTimeRange] = useState<'1h' | '24h' | '7d'>('24h');

  // Generate mock auction data
  const generateMockAuctions = (): Auction[] => {
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

    const auctionTypes: ('mini' | 'batch' | 'continuous')[] = ['mini', 'batch', 'continuous'];
    const statuses: ('pending' | 'active' | 'completed' | 'cancelled')[] = ['pending', 'active', 'completed'];

    return bondNames.slice(0, 8).map((name, index) => {
      const auctionType = auctionTypes[Math.floor(Math.random() * auctionTypes.length)];
      const status = statuses[Math.floor(Math.random() * statuses.length)];
      const initialPrice = 95 + Math.random() * 10;
      const currentPrice = initialPrice + (Math.random() - 0.5) * 2;
      const targetVolume = 1000000 + Math.random() * 10000000;
      const currentVolume = targetVolume * (0.1 + Math.random() * 0.8);
      
      // Generate price history
      const priceHistory: PricePoint[] = [];
      const now = new Date();
      for (let i = 0; i < 20; i++) {
        const time = new Date(now.getTime() - (20 - i) * 5 * 60 * 1000); // 5-minute intervals
        const price = initialPrice + (Math.random() - 0.5) * 3;
        const volume = Math.random() * 1000000;
        const participants = Math.floor(Math.random() * 50) + 5;
        
        priceHistory.push({
          timestamp: time,
          price,
          volume,
          participants
        });
      }

      // Generate orders
      const buyOrders: Order[] = [];
      const sellOrders: Order[] = [];
      
      for (let i = 0; i < 5 + Math.random() * 10; i++) {
        const orderType = Math.random() > 0.5 ? 'buy' : 'sell';
        const participantType = Math.random() > 0.7 ? 'institutional' : 
                               Math.random() > 0.5 ? 'market_maker' : 'retail';
        
        const order: Order = {
          id: `order_${i}`,
          type: orderType,
          amount: 100000 + Math.random() * 5000000,
          price: currentPrice + (Math.random() - 0.5) * 1,
          timestamp: new Date(now.getTime() - Math.random() * 60 * 60 * 1000),
          participantId: `${participantType}_${Math.floor(Math.random() * 1000)}`,
          participantType
        };
        
        if (orderType === 'buy') {
          buyOrders.push(order);
        } else {
          sellOrders.push(order);
        }
      }

      return {
        id: `auction_${index}`,
        bondId: `bond_${index}`,
        bondName: name,
        auctionType,
        status,
        startTime: new Date(now.getTime() - Math.random() * 2 * 60 * 60 * 1000),
        endTime: new Date(now.getTime() + Math.random() * 2 * 60 * 60 * 1000),
        initialPrice,
        currentPrice,
        targetVolume,
        currentVolume,
        buyOrders,
        sellOrders,
        priceHistory,
        participants: Math.floor(Math.random() * 50) + 10
      };
    });
  };

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      await new Promise(resolve => setTimeout(resolve, 1000));
      setAuctions(generateMockAuctions());
      setIsLoading(false);
    };

    loadData();
    
    const interval = setInterval(loadData, 15000); // Update every 15 seconds
    return () => clearInterval(interval);
  }, []);

  const activeAuctions = auctions.filter(a => a.status === 'active');
  const completedAuctions = auctions.filter(a => a.status === 'completed');
  const pendingAuctions = auctions.filter(a => a.status === 'pending');

  const totalVolume = auctions.reduce((sum, auction) => sum + auction.currentVolume, 0);
  const totalParticipants = auctions.reduce((sum, auction) => sum + auction.participants, 0);
  const avgPriceChange = auctions.reduce((sum, auction) => 
    sum + ((auction.currentPrice - auction.initialPrice) / auction.initialPrice), 0) / auctions.length;

  const selectedAuctionData = auctions.find(a => a.id === selectedAuction);

  // Generate market efficiency data
  const generateEfficiencyData = () => {
    return auctions.map(auction => ({
      name: auction.bondName.split(' ').slice(0, 2).join(' '),
      priceChange: ((auction.currentPrice - auction.initialPrice) / auction.initialPrice) * 100,
      volume: auction.currentVolume / 1000000,
      participants: auction.participants,
      efficiency: Math.min(100, (auction.currentVolume / auction.targetVolume) * 100)
    }));
  };

  const efficiencyData = generateEfficiencyData();

  if (isLoading) {
    return (
      <div className={`bg-gray-900/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 ${className}`}>
        <div className="flex items-center justify-center h-64">
          <div className="flex flex-col items-center space-y-4">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-gray-400">Loading auction data...</p>
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
            <Gavel className="w-6 h-6 text-blue-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-white">Auction-Driven Price Discovery</h3>
            <p className="text-sm text-gray-400">Real-time mini-auctions and batch order matching</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setTimeRange('1h')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              timeRange === '1h'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            1H
          </button>
          <button
            onClick={() => setTimeRange('24h')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              timeRange === '24h'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            24H
          </button>
          <button
            onClick={() => setTimeRange('7d')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              timeRange === '7d'
                ? 'bg-blue-500 text-white'
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
            <Activity className="w-4 h-4 text-blue-400" />
            <span className="text-sm text-gray-400">Active Auctions</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">{activeAuctions.length}</p>
          <p className="text-xs text-gray-500">currently running</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <DollarSign className="w-4 h-4 text-green-400" />
            <span className="text-sm text-gray-400">Total Volume</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">₹{(totalVolume / 1000000).toFixed(1)}M</p>
          <p className="text-xs text-gray-500">in auctions</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Users className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-gray-400">Participants</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">{totalParticipants}</p>
          <p className="text-xs text-gray-500">across all auctions</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Target className="w-4 h-4 text-yellow-400" />
            <span className="text-sm text-gray-400">Avg Price Change</span>
          </div>
          <p className={`text-2xl font-bold mt-1 ${avgPriceChange >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {avgPriceChange >= 0 ? '+' : ''}{(avgPriceChange * 100).toFixed(2)}%
          </p>
          <p className="text-xs text-gray-500">price discovery</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Market Efficiency Chart */}
        <div className="bg-gray-800/30 rounded-lg p-4">
          <h4 className="text-lg font-semibold text-white mb-4">Market Efficiency by Bond</h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart data={efficiencyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="participants" 
                  stroke="#9ca3af"
                  fontSize={12}
                  label={{ value: 'Participants', position: 'insideBottom', offset: -5, style: { textAnchor: 'middle', fill: '#9ca3af' } }}
                />
                <YAxis 
                  dataKey="efficiency" 
                  stroke="#9ca3af"
                  fontSize={12}
                  label={{ value: 'Efficiency (%)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: '#9ca3af' } }}
                />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1f2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#f9fafb'
                  }}
                  formatter={(value: number, name: string) => [
                    name === 'efficiency' ? `${value.toFixed(1)}%` : 
                    name === 'volume' ? `₹${value.toFixed(1)}M` :
                    name === 'priceChange' ? `${value.toFixed(2)}%` : value,
                    name.charAt(0).toUpperCase() + name.slice(1)
                  ]}
                />
                <Scatter dataKey="efficiency" fill="#3b82f6" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Auction Status Distribution */}
        <div className="bg-gray-800/30 rounded-lg p-4">
          <h4 className="text-lg font-semibold text-white mb-4">Auction Status Overview</h4>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                <span className="text-sm text-gray-300">Active</span>
              </div>
              <span className="text-sm font-medium text-white">{activeAuctions.length}</span>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                <span className="text-sm text-gray-300">Completed</span>
              </div>
              <span className="text-sm font-medium text-white">{completedAuctions.length}</span>
            </div>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                <span className="text-sm text-gray-300">Pending</span>
              </div>
              <span className="text-sm font-medium text-white">{pendingAuctions.length}</span>
            </div>
            
            <div className="mt-4">
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div className="flex h-2 rounded-full">
                  <div 
                    className="bg-green-500 h-2 rounded-l-full"
                    style={{ width: `${(activeAuctions.length / auctions.length) * 100}%` }}
                  />
                  <div 
                    className="bg-blue-500 h-2"
                    style={{ width: `${(completedAuctions.length / auctions.length) * 100}%` }}
                  />
                  <div 
                    className="bg-yellow-500 h-2 rounded-r-full"
                    style={{ width: `${(pendingAuctions.length / auctions.length) * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Auction List */}
      <div className="mt-6">
        <h4 className="text-lg font-semibold text-white mb-4">Live Auctions</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {auctions.map((auction) => (
            <motion.div
              key={auction.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className={`bg-gray-800/50 rounded-lg p-4 cursor-pointer transition-all duration-200 ${
                selectedAuction === auction.id ? 'ring-2 ring-blue-500' : 'hover:bg-gray-700/50'
              }`}
              onClick={() => setSelectedAuction(auction.id)}
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h5 className="text-sm font-medium text-white truncate">
                    {auction.bondName.split(' ').slice(0, 2).join(' ')}
                  </h5>
                  <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                    auction.status === 'active' ? 'bg-green-500/20 text-green-400' :
                    auction.status === 'completed' ? 'bg-blue-500/20 text-blue-400' :
                    'bg-yellow-500/20 text-yellow-400'
                  }`}>
                    {auction.status}
                  </span>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Type</span>
                    <span className="text-xs font-medium text-white capitalize">{auction.auctionType}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Current Price</span>
                    <span className="text-xs font-medium text-white">₹{auction.currentPrice.toFixed(2)}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Volume Progress</span>
                    <span className="text-xs font-medium text-white">
                      {((auction.currentVolume / auction.targetVolume) * 100).toFixed(0)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Participants</span>
                    <span className="text-xs font-medium text-white">{auction.participants}</span>
                  </div>
                </div>
                
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-blue-400 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${(auction.currentVolume / auction.targetVolume) * 100}%` }}
                  />
                </div>
                
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">
                    {auction.status === 'active' ? 'Ends in' : 'Ended'}
                  </span>
                  <span className="text-white">
                    {auction.status === 'active' ? 
                      `${Math.floor((auction.endTime.getTime() - Date.now()) / (1000 * 60))}m` :
                      auction.endTime.toLocaleTimeString()
                    }
                  </span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Selected Auction Details */}
      {selectedAuctionData && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 bg-gray-800/30 rounded-lg p-4"
        >
          <h4 className="text-lg font-semibold text-white mb-4">
            Auction Details: {selectedAuctionData.bondName}
          </h4>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Price History Chart */}
            <div>
              <h5 className="text-sm font-medium text-gray-300 mb-3">Price Discovery Timeline</h5>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={selectedAuctionData.priceHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="timestamp" 
                      stroke="#9ca3af"
                      fontSize={10}
                      tickFormatter={(time) => new Date(time).toLocaleTimeString().slice(0, 5)}
                    />
                    <YAxis 
                      stroke="#9ca3af"
                      fontSize={10}
                      domain={['dataMin - 0.5', 'dataMax + 0.5']}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: '#1f2937', 
                        border: '1px solid #374151',
                        borderRadius: '8px',
                        color: '#f9fafb'
                      }}
                      formatter={(value: number) => [`₹${value.toFixed(2)}`, 'Price']}
                      labelFormatter={(time) => new Date(time).toLocaleString()}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="price" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      dot={{ fill: '#3b82f6', strokeWidth: 2, r: 3 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Order Book Summary */}
            <div>
              <h5 className="text-sm font-medium text-gray-300 mb-3">Order Book Summary</h5>
              <div className="space-y-4">
                <div className="bg-gray-700/30 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-green-400">Buy Orders</span>
                    <span className="text-sm font-medium text-white">{selectedAuctionData.buyOrders.length}</span>
                  </div>
                  <div className="text-xs text-gray-400">
                    Total: ₹{(selectedAuctionData.buyOrders.reduce((sum, order) => sum + order.amount, 0) / 1000000).toFixed(1)}M
                  </div>
                </div>
                
                <div className="bg-gray-700/30 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-red-400">Sell Orders</span>
                    <span className="text-sm font-medium text-white">{selectedAuctionData.sellOrders.length}</span>
                  </div>
                  <div className="text-xs text-gray-400">
                    Total: ₹{(selectedAuctionData.sellOrders.reduce((sum, order) => sum + order.amount, 0) / 1000000).toFixed(1)}M
                  </div>
                </div>
                
                <div className="bg-gray-700/30 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-blue-400">Price Change</span>
                    <span className={`text-sm font-medium ${
                      selectedAuctionData.currentPrice >= selectedAuctionData.initialPrice ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {selectedAuctionData.currentPrice >= selectedAuctionData.initialPrice ? '+' : ''}
                      {((selectedAuctionData.currentPrice - selectedAuctionData.initialPrice) / selectedAuctionData.initialPrice * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="text-xs text-gray-400">
                    ₹{selectedAuctionData.initialPrice.toFixed(2)} → ₹{selectedAuctionData.currentPrice.toFixed(2)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default AuctionPriceDiscovery;
