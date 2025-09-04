import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, TrendingDown, TrendingUp, Zap, Target, BarChart3, PieChart } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart as RechartsPieChart, Cell, LineChart, Line } from 'recharts';

interface StressScenario {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  parameters: {
    repoRateChange: number; // bps
    inflationShock: number; // percentage
    ratingDowngrade: number; // notches
    liquidityShock: number; // percentage
  };
}

interface StressResult {
  scenario: StressScenario;
  portfolioImpact: {
    totalPnL: number;
    durationChange: number;
    convexityChange: number;
    liquidityChange: number;
    var95: number;
    var99: number;
  };
  bondImpacts: Array<{
    isin: string;
    name: string;
    currentValue: number;
    stressedValue: number;
    pnl: number;
    durationImpact: number;
    convexityImpact: number;
  }>;
  distribution: Array<{
    range: string;
    count: number;
    percentage: number;
  }>;
}

interface PortfolioStressSimulationProps {
  className?: string;
}

const PortfolioStressSimulation: React.FC<PortfolioStressSimulationProps> = ({ className = '' }) => {
  const [scenarios, setScenarios] = useState<StressScenario[]>([]);
  const [selectedScenarios, setSelectedScenarios] = useState<string[]>([]);
  const [stressResults, setStressResults] = useState<StressResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [customScenario, setCustomScenario] = useState({
    repoRateChange: 0,
    inflationShock: 0,
    ratingDowngrade: 0,
    liquidityShock: 0
  });

  // Define stress scenarios
  const stressScenarios: StressScenario[] = [
    {
      id: 'mild_repo_hike',
      name: 'Mild Repo Hike',
      description: 'RBI increases repo rate by 25 bps',
      icon: <TrendingUp className="w-5 h-5" />,
      color: 'text-yellow-400',
      parameters: {
        repoRateChange: 25,
        inflationShock: 0,
        ratingDowngrade: 0,
        liquidityShock: 0
      }
    },
    {
      id: 'moderate_repo_hike',
      name: 'Moderate Repo Hike',
      description: 'RBI increases repo rate by 50 bps',
      icon: <TrendingUp className="w-5 h-5" />,
      color: 'text-orange-400',
      parameters: {
        repoRateChange: 50,
        inflationShock: 0,
        ratingDowngrade: 0,
        liquidityShock: 0
      }
    },
    {
      id: 'severe_repo_hike',
      name: 'Severe Repo Hike',
      description: 'RBI increases repo rate by 100 bps',
      icon: <TrendingUp className="w-5 h-5" />,
      color: 'text-red-400',
      parameters: {
        repoRateChange: 100,
        inflationShock: 0,
        ratingDowngrade: 0,
        liquidityShock: 0
      }
    },
    {
      id: 'inflation_shock',
      name: 'Inflation Shock',
      description: 'CPI rises by 2% above target',
      icon: <AlertTriangle className="w-5 h-5" />,
      color: 'text-red-400',
      parameters: {
        repoRateChange: 0,
        inflationShock: 2,
        ratingDowngrade: 0,
        liquidityShock: 0
      }
    },
    {
      id: 'rating_downgrade',
      name: 'Rating Downgrade',
      description: 'Widespread corporate rating downgrades',
      icon: <TrendingDown className="w-5 h-5" />,
      color: 'text-red-400',
      parameters: {
        repoRateChange: 0,
        inflationShock: 0,
        ratingDowngrade: 1,
        liquidityShock: 0
      }
    },
    {
      id: 'liquidity_crisis',
      name: 'Liquidity Crisis',
      description: 'Market liquidity drops by 50%',
      icon: <Zap className="w-5 h-5" />,
      color: 'text-red-400',
      parameters: {
        repoRateChange: 0,
        inflationShock: 0,
        ratingDowngrade: 0,
        liquidityShock: 50
      }
    }
  ];

  useEffect(() => {
    setScenarios(stressScenarios);
  }, []);

  // Generate mock stress test results
  const generateStressResults = (selectedScenarios: string[]): StressResult[] => {
    return selectedScenarios.map(scenarioId => {
      const scenario = scenarios.find(s => s.id === scenarioId);
      if (!scenario) return null;

      const totalPnL = -(scenario.parameters.repoRateChange * 0.1 + 
                        scenario.parameters.inflationShock * 0.05 + 
                        scenario.parameters.ratingDowngrade * 0.02 + 
                        scenario.parameters.liquidityShock * 0.01) * 1000000;

      const durationChange = scenario.parameters.repoRateChange * 0.01 + 
                            scenario.parameters.inflationShock * 0.005;
      
      const convexityChange = scenario.parameters.repoRateChange * 0.02 + 
                             scenario.parameters.inflationShock * 0.01;

      const liquidityChange = -scenario.parameters.liquidityShock * 0.1;

      // Generate bond impacts
      const bondImpacts = [
        {
          isin: 'INE094A08176',
          name: 'HINDUSTAN PETROLEUM',
          currentValue: 5000000,
          stressedValue: 5000000 + totalPnL * 0.3,
          pnl: totalPnL * 0.3,
          durationImpact: durationChange * 0.8,
          convexityImpact: convexityChange * 0.8
        },
        {
          isin: 'INE556F08KZ3',
          name: 'SIDBI',
          currentValue: 3200000,
          stressedValue: 3200000 + totalPnL * 0.2,
          pnl: totalPnL * 0.2,
          durationImpact: durationChange * 0.6,
          convexityImpact: convexityChange * 0.6
        },
        {
          isin: 'INE163N08289',
          name: 'ONGC PETRO',
          currentValue: 3200000,
          stressedValue: 3200000 + totalPnL * 0.25,
          pnl: totalPnL * 0.25,
          durationImpact: durationChange * 0.7,
          convexityImpact: convexityChange * 0.7
        },
        {
          isin: 'INE776C08075',
          name: 'GMR AIRPORTS',
          currentValue: 3150000,
          stressedValue: 3150000 + totalPnL * 0.25,
          pnl: totalPnL * 0.25,
          durationImpact: durationChange * 0.9,
          convexityImpact: convexityChange * 0.9
        }
      ];

      // Generate P&L distribution
      const distribution = [
        { range: '> -5%', count: Math.floor(Math.random() * 20) + 5, percentage: 0 },
        { range: '-5% to -3%', count: Math.floor(Math.random() * 30) + 10, percentage: 0 },
        { range: '-3% to -1%', count: Math.floor(Math.random() * 40) + 20, percentage: 0 },
        { range: '-1% to 0%', count: Math.floor(Math.random() * 50) + 30, percentage: 0 },
        { range: '0% to 1%', count: Math.floor(Math.random() * 30) + 15, percentage: 0 },
        { range: '1% to 3%', count: Math.floor(Math.random() * 20) + 5, percentage: 0 },
        { range: '> 3%', count: Math.floor(Math.random() * 10) + 2, percentage: 0 }
      ];

      const totalCount = distribution.reduce((sum, d) => sum + d.count, 0);
      distribution.forEach(d => {
        d.percentage = (d.count / totalCount) * 100;
      });

      return {
        scenario,
        portfolioImpact: {
          totalPnL,
          durationChange,
          convexityChange,
          liquidityChange,
          var95: totalPnL * 0.8,
          var99: totalPnL * 0.9
        },
        bondImpacts,
        distribution
      };
    }).filter(Boolean) as StressResult[];
  };

  const runStressTest = async () => {
    if (selectedScenarios.length === 0) return;
    
    setIsLoading(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    setStressResults(generateStressResults(selectedScenarios));
    setIsLoading(false);
  };

  const addCustomScenario = () => {
    const customScenarioData: StressScenario = {
      id: 'custom',
      name: 'Custom Scenario',
      description: `Custom stress test with repo: ${customScenario.repoRateChange}bps, inflation: ${customScenario.inflationShock}%, rating: ${customScenario.ratingDowngrade}, liquidity: ${customScenario.liquidityShock}%`,
      icon: <Target className="w-5 h-5" />,
      color: 'text-blue-400',
      parameters: customScenario
    };
    
    setScenarios([...scenarios, customScenarioData]);
    setSelectedScenarios([...selectedScenarios, 'custom']);
  };

  const toggleScenario = (scenarioId: string) => {
    setSelectedScenarios(prev => 
      prev.includes(scenarioId) 
        ? prev.filter(id => id !== scenarioId)
        : [...prev, scenarioId]
    );
  };

  const COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899'];

  return (
    <div className={`bg-gray-900/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-red-500/20 rounded-lg">
            <BarChart3 className="w-6 h-6 text-red-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-white">Portfolio Stress Simulation</h3>
            <p className="text-sm text-gray-400">Multi-factor stress testing with P&L distributions</p>
          </div>
        </div>
      </div>

      {/* Scenario Selection */}
      <div className="mb-6">
        <h4 className="text-lg font-semibold text-white mb-4">Select Stress Scenarios</h4>
        <div className="grid grid-cols-2 gap-3 mb-4">
          {scenarios.map((scenario) => (
            <motion.div
              key={scenario.id}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`p-4 rounded-lg border cursor-pointer transition-all ${
                selectedScenarios.includes(scenario.id)
                  ? 'border-blue-500 bg-blue-500/10'
                  : 'border-gray-700 bg-gray-800/30 hover:border-gray-600'
              }`}
              onClick={() => toggleScenario(scenario.id)}
            >
              <div className="flex items-center space-x-3">
                <div className={scenario.color}>
                  {scenario.icon}
                </div>
                <div className="flex-1">
                  <h5 className="text-sm font-medium text-white">{scenario.name}</h5>
                  <p className="text-xs text-gray-400">{scenario.description}</p>
                </div>
                {selectedScenarios.includes(scenario.id) && (
                  <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                )}
              </div>
            </motion.div>
          ))}
        </div>

        {/* Custom Scenario */}
        <div className="bg-gray-800/50 rounded-lg p-4 mb-4">
          <h5 className="text-sm font-medium text-white mb-3">Custom Scenario</h5>
          <div className="grid grid-cols-4 gap-4">
            <div>
              <label className="text-xs text-gray-400">Repo Rate Change (bps)</label>
              <input
                type="number"
                value={customScenario.repoRateChange}
                onChange={(e) => setCustomScenario(prev => ({ ...prev, repoRateChange: Number(e.target.value) }))}
                className="w-full mt-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400">Inflation Shock (%)</label>
              <input
                type="number"
                step="0.1"
                value={customScenario.inflationShock}
                onChange={(e) => setCustomScenario(prev => ({ ...prev, inflationShock: Number(e.target.value) }))}
                className="w-full mt-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400">Rating Downgrade</label>
              <input
                type="number"
                value={customScenario.ratingDowngrade}
                onChange={(e) => setCustomScenario(prev => ({ ...prev, ratingDowngrade: Number(e.target.value) }))}
                className="w-full mt-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400">Liquidity Shock (%)</label>
              <input
                type="number"
                value={customScenario.liquidityShock}
                onChange={(e) => setCustomScenario(prev => ({ ...prev, liquidityShock: Number(e.target.value) }))}
                className="w-full mt-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm"
              />
            </div>
          </div>
          <button
            onClick={addCustomScenario}
            className="mt-3 px-4 py-2 bg-blue-500 text-white rounded-lg text-sm font-medium hover:bg-blue-600 transition-colors"
          >
            Add Custom Scenario
          </button>
        </div>

        <button
          onClick={runStressTest}
          disabled={selectedScenarios.length === 0 || isLoading}
          className="w-full px-4 py-3 bg-red-500 text-white rounded-lg font-medium hover:bg-red-600 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? 'Running Stress Test...' : `Run Stress Test (${selectedScenarios.length} scenarios)`}
        </button>
      </div>

      {/* Stress Test Results */}
      <AnimatePresence>
        {stressResults.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="space-y-6"
          >
            {stressResults.map((result, index) => (
              <div key={result.scenario.id} className="bg-gray-800/50 rounded-lg p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <div className={result.scenario.color}>
                    {result.scenario.icon}
                  </div>
                  <div>
                    <h4 className="text-lg font-semibold text-white">{result.scenario.name}</h4>
                    <p className="text-sm text-gray-400">{result.scenario.description}</p>
                  </div>
                </div>

                {/* Portfolio Impact Summary */}
                <div className="grid grid-cols-3 gap-4 mb-6">
                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <TrendingDown className="w-4 h-4 text-red-400" />
                      <span className="text-sm text-gray-400">Total P&L Impact</span>
                    </div>
                    <p className={`text-2xl font-bold ${
                      result.portfolioImpact.totalPnL < 0 ? 'text-red-400' : 'text-green-400'
                    }`}>
                      ₹{(result.portfolioImpact.totalPnL / 1000000).toFixed(2)}M
                    </p>
                  </div>
                  
                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <Target className="w-4 h-4 text-blue-400" />
                      <span className="text-sm text-gray-400">Duration Change</span>
                    </div>
                    <p className={`text-2xl font-bold ${
                      result.portfolioImpact.durationChange > 0 ? 'text-red-400' : 'text-green-400'
                    }`}>
                      {result.portfolioImpact.durationChange > 0 ? '+' : ''}{result.portfolioImpact.durationChange.toFixed(2)} years
                    </p>
                  </div>
                  
                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <div className="flex items-center space-x-2 mb-2">
                      <AlertTriangle className="w-4 h-4 text-yellow-400" />
                      <span className="text-sm text-gray-400">VaR 95%</span>
                    </div>
                    <p className="text-2xl font-bold text-red-400">
                      ₹{(result.portfolioImpact.var95 / 1000000).toFixed(2)}M
                    </p>
                  </div>
                </div>

                {/* Bond Impact Details */}
                <div className="mb-6">
                  <h5 className="text-sm font-medium text-gray-400 mb-3">Individual Bond Impacts</h5>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={result.bondImpacts}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis 
                          dataKey="name" 
                          stroke="#9ca3af"
                          fontSize={12}
                          angle={-45}
                          textAnchor="end"
                          height={80}
                        />
                        <YAxis 
                          stroke="#9ca3af"
                          fontSize={12}
                          label={{ value: 'P&L (₹M)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: '#9ca3af' } }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: '#1f2937', 
                            border: '1px solid #374151',
                            borderRadius: '8px',
                            color: '#f9fafb'
                          }}
                          formatter={(value: number) => [`₹${(value / 1000000).toFixed(2)}M`, 'P&L Impact']}
                        />
                        <Bar 
                          dataKey="pnl" 
                          fill="#ef4444"
                          radius={[4, 4, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* P&L Distribution */}
                <div>
                  <h5 className="text-sm font-medium text-gray-400 mb-3">P&L Distribution</h5>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={result.distribution}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis 
                          dataKey="range" 
                          stroke="#9ca3af"
                          fontSize={12}
                        />
                        <YAxis 
                          stroke="#9ca3af"
                          fontSize={12}
                          label={{ value: 'Count', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fill: '#9ca3af' } }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: '#1f2937', 
                            border: '1px solid #374151',
                            borderRadius: '8px',
                            color: '#f9fafb'
                          }}
                          formatter={(value: number, name: string, props: any) => [
                            `${value} (${props.payload.percentage.toFixed(1)}%)`,
                            'Count'
                          ]}
                        />
                        <Bar 
                          dataKey="count" 
                          fill="#3b82f6"
                          radius={[4, 4, 0, 0]}
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default PortfolioStressSimulation;
