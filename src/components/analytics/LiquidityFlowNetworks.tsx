import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Network, Activity, TrendingUp, TrendingDown, Zap, Target, BarChart3 } from 'lucide-react';
import * as d3 from 'd3';

interface BondNode {
  id: string;
  name: string;
  sector: string;
  rating: string;
  liquidity: number;
  volume: number;
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
}

interface BondLink {
  source: string;
  target: string;
  correlation: number;
  volume: number;
  type: 'correlation' | 'trading_flow';
}

interface LiquidityFlowNetworksProps {
  className?: string;
}

const LiquidityFlowNetworks: React.FC<LiquidityFlowNetworksProps> = ({ className = '' }) => {
  const [nodes, setNodes] = useState<BondNode[]>([]);
  const [links, setLinks] = useState<BondLink[]>([]);
  const [selectedNode, setSelectedNode] = useState<BondNode | null>(null);
  const [networkType, setNetworkType] = useState<'correlation' | 'trading_flow'>('correlation');
  const [isLoading, setIsLoading] = useState(true);
  const svgRef = useRef<SVGSVGElement>(null);

  // Generate mock network data
  const generateNetworkData = () => {
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
      'HDFC BANK LIMITED',
      'BAJAJ FINANCE LIMITED',
      'POWER GRID CORPORATION OF INDIA LIMITED',
      'RURAL ELECTRIFICATION CORPORATION LIMITED',
      'INDIAN RAILWAY FINANCE CORPORATION LIMITED',
      'LARSEN & TOUBRO LIMITED'
    ];

    const sectors = ['Oil & Gas', 'Banking', 'Infrastructure', 'Power', 'Finance', 'Steel'];
    const ratings = ['AAA', 'AA+', 'AA', 'AA-', 'A+'];

    const generatedNodes: BondNode[] = bondNames.map((name, index) => ({
      id: `bond_${index}`,
      name: name.split(' ').slice(0, 2).join(' '),
      sector: sectors[index % sectors.length],
      rating: ratings[index % ratings.length],
      liquidity: Math.random() * 100,
      volume: Math.random() * 10000000 + 1000000
    }));

    const generatedLinks: BondLink[] = [];
    
    // Generate correlation links
    for (let i = 0; i < generatedNodes.length; i++) {
      for (let j = i + 1; j < generatedNodes.length; j++) {
        const correlation = Math.random() * 0.8 + 0.2; // 0.2 to 1.0
        if (correlation > 0.6) { // Only show strong correlations
          generatedLinks.push({
            source: generatedNodes[i].id,
            target: generatedNodes[j].id,
            correlation,
            volume: Math.random() * 5000000 + 1000000,
            type: 'correlation'
          });
        }
      }
    }

    // Generate trading flow links
    for (let i = 0; i < generatedNodes.length; i++) {
      const numFlows = Math.floor(Math.random() * 3) + 1; // 1-3 flows per bond
      for (let j = 0; j < numFlows; j++) {
        const targetIndex = Math.floor(Math.random() * generatedNodes.length);
        if (targetIndex !== i) {
          generatedLinks.push({
            source: generatedNodes[i].id,
            target: generatedNodes[targetIndex].id,
            correlation: Math.random() * 0.4 + 0.1, // Lower correlation for trading flows
            volume: Math.random() * 2000000 + 500000,
            type: 'trading_flow'
          });
        }
      }
    }

    return { nodes: generatedNodes, links: generatedLinks };
  };

  useEffect(() => {
    const loadNetworkData = async () => {
      setIsLoading(true);
      await new Promise(resolve => setTimeout(resolve, 1000));
      const { nodes: newNodes, links: newLinks } = generateNetworkData();
      setNodes(newNodes);
      setLinks(newLinks);
      setIsLoading(false);
    };

    loadNetworkData();
    
    const interval = setInterval(loadNetworkData, 30000);
    return () => clearInterval(interval);
  }, []);

  // Render network visualization
  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 800;
    const height = 600;
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };

    // Create force simulation
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id((d: any) => d.id).distance(100))
      .force("charge", d3.forceManyBody().strength(-300))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(30));

    // Filter links based on network type
    const filteredLinks = links.filter(link => link.type === networkType);

    // Create links
    const link = svg.append("g")
      .selectAll("line")
      .data(filteredLinks)
      .enter().append("line")
      .attr("stroke", (d: BondLink) => {
        if (networkType === 'correlation') {
          return d.correlation > 0.8 ? "#22c55e" : d.correlation > 0.6 ? "#eab308" : "#ef4444";
        } else {
          return "#3b82f6";
        }
      })
      .attr("stroke-opacity", 0.6)
      .attr("stroke-width", (d: BondLink) => Math.sqrt(d.volume / 1000000) * 2);

    // Create nodes
    const node = svg.append("g")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("r", (d: BondNode) => Math.sqrt(d.volume / 1000000) * 3 + 8)
      .attr("fill", (d: BondNode) => {
        if (d.liquidity >= 80) return "#22c55e";
        if (d.liquidity >= 60) return "#eab308";
        if (d.liquidity >= 40) return "#f97316";
        return "#ef4444";
      })
      .attr("stroke", "#1f2937")
      .attr("stroke-width", 2)
      .style("cursor", "pointer")
      .on("click", (event, d: BondNode) => {
        setSelectedNode(d);
      })
      .call(d3.drag<any, BondNode>()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended) as any);

    // Add labels
    const label = svg.append("g")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
      .text((d: BondNode) => d.name)
      .attr("font-size", "10px")
      .attr("font-family", "sans-serif")
      .attr("fill", "#f9fafb")
      .attr("text-anchor", "middle")
      .attr("dy", "0.35em");

    // Update positions on simulation tick
    simulation.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);

      node
        .attr("cx", (d: any) => d.x)
        .attr("cy", (d: any) => d.y);

      label
        .attr("x", (d: any) => d.x)
        .attr("y", (d: any) => d.y);
    });

    function dragstarted(event: any, d: BondNode) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: BondNode) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: any, d: BondNode) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    return () => {
      simulation.stop();
    };
  }, [nodes, links, networkType]);

  const getSectorColor = (sector: string) => {
    const colors: { [key: string]: string } = {
      'Oil & Gas': 'text-blue-400',
      'Banking': 'text-green-400',
      'Infrastructure': 'text-orange-400',
      'Power': 'text-yellow-400',
      'Finance': 'text-purple-400',
      'Steel': 'text-gray-400'
    };
    return colors[sector] || 'text-gray-400';
  };

  const getRatingColor = (rating: string) => {
    if (rating.startsWith('AAA')) return 'text-green-400';
    if (rating.startsWith('AA')) return 'text-blue-400';
    if (rating.startsWith('A')) return 'text-yellow-400';
    return 'text-red-400';
  };

  const filteredLinks = links.filter(link => link.type === networkType);
  const totalVolume = filteredLinks.reduce((sum, link) => sum + link.volume, 0);
  const avgCorrelation = filteredLinks.reduce((sum, link) => sum + link.correlation, 0) / filteredLinks.length;

  if (isLoading) {
    return (
      <div className={`bg-gray-900/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 ${className}`}>
        <div className="flex items-center justify-center h-64">
          <div className="flex flex-col items-center space-y-4">
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            <p className="text-gray-400">Loading network data...</p>
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
            <Network className="w-6 h-6 text-blue-400" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-white">Liquidity Flow & Correlation Networks</h3>
            <p className="text-sm text-gray-400">Network visualization of bond correlations and trading flows</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setNetworkType('correlation')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              networkType === 'correlation'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Correlations
          </button>
          <button
            onClick={() => setNetworkType('trading_flow')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              networkType === 'trading_flow'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Trading Flows
          </button>
        </div>
      </div>

      {/* Network Stats */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Activity className="w-4 h-4 text-blue-400" />
            <span className="text-sm text-gray-400">Total Nodes</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">{nodes.length}</p>
          <p className="text-xs text-gray-500">bonds</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Zap className="w-4 h-4 text-green-400" />
            <span className="text-sm text-gray-400">Active Links</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">{filteredLinks.length}</p>
          <p className="text-xs text-gray-500">connections</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-gray-400">Total Volume</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">₹{(totalVolume / 1000000).toFixed(0)}M</p>
          <p className="text-xs text-gray-500">trading volume</p>
        </div>
        
        <div className="bg-gray-800/50 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <Target className="w-4 h-4 text-yellow-400" />
            <span className="text-sm text-gray-400">Avg Correlation</span>
          </div>
          <p className="text-2xl font-bold text-white mt-1">{(avgCorrelation * 100).toFixed(0)}%</p>
          <p className="text-xs text-gray-500">strength</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Network Visualization */}
        <div className="lg:col-span-3">
          <div className="bg-gray-800/30 rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-semibold text-white">
                {networkType === 'correlation' ? 'Correlation Network' : 'Trading Flow Network'}
              </h4>
              <div className="flex items-center space-x-4 text-xs text-gray-400">
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-3 bg-green-400 rounded-full"></div>
                  <span>High Liquidity</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
                  <span>Medium</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-3 h-3 bg-red-400 rounded-full"></div>
                  <span>Low Liquidity</span>
                </div>
              </div>
            </div>
            
            <svg
              ref={svgRef}
              width="100%"
              height="600"
              className="border border-gray-700 rounded-lg"
            />
          </div>
        </div>

        {/* Node Details */}
        <div className="lg:col-span-1">
          <h4 className="text-lg font-semibold text-white mb-4">Bond Details</h4>
          
          {selectedNode ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gray-800/50 rounded-lg p-4"
            >
              <div className="space-y-4">
                <div>
                  <h5 className="text-sm font-medium text-white">{selectedNode.name}</h5>
                  <p className="text-xs text-gray-400">{selectedNode.sector}</p>
                </div>
                
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Rating</span>
                    <span className={`text-xs font-medium ${getRatingColor(selectedNode.rating)}`}>
                      {selectedNode.rating}
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Liquidity</span>
                    <span className={`text-xs font-medium ${
                      selectedNode.liquidity >= 80 ? 'text-green-400' :
                      selectedNode.liquidity >= 60 ? 'text-yellow-400' :
                      selectedNode.liquidity >= 40 ? 'text-orange-400' : 'text-red-400'
                    }`}>
                      {selectedNode.liquidity.toFixed(0)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Volume</span>
                    <span className="text-xs font-medium text-white">
                      ₹{(selectedNode.volume / 1000000).toFixed(1)}M
                    </span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-xs text-gray-400">Sector</span>
                    <span className={`text-xs font-medium ${getSectorColor(selectedNode.sector)}`}>
                      {selectedNode.sector}
                    </span>
                  </div>
                </div>
                
                <div className="pt-4 border-t border-gray-700">
                  <h6 className="text-xs font-medium text-gray-400 mb-2">Connected Bonds</h6>
                  <div className="space-y-1">
                    {filteredLinks
                      .filter(link => link.source === selectedNode.id || link.target === selectedNode.id)
                      .slice(0, 3)
                      .map((link, index) => {
                        const connectedNode = nodes.find(n => 
                          n.id === (link.source === selectedNode.id ? link.target : link.source)
                        );
                        return (
                          <div key={index} className="flex justify-between items-center">
                            <span className="text-xs text-gray-300 truncate">
                              {connectedNode?.name}
                            </span>
                            <span className="text-xs text-gray-400">
                              {(link.correlation * 100).toFixed(0)}%
                            </span>
                          </div>
                        );
                      })}
                  </div>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="bg-gray-800/30 rounded-lg p-4 text-center">
              <Network className="w-8 h-8 text-gray-500 mx-auto mb-2" />
              <p className="text-gray-400 text-sm">Click on a node to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LiquidityFlowNetworks;
