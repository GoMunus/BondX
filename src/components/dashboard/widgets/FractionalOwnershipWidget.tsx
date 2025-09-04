import React, { useState, useEffect } from 'react'
import { 
  ShoppingCart, 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Percent, 
  Building, 
  Star,
  AlertCircle,
  CheckCircle,
  Loader2
} from 'lucide-react'
import { apiService } from '@/services/api'

interface FractionalPosition {
  bond_id: string
  bond_name: string
  issuer: string
  sector: string
  face_value: number
  current_price: number
  yield_to_maturity: number
  owned_fraction: number
  owned_value: number
  purchase_price: number
  current_value: number
  unrealized_pnl: number
  ownership_percentage: number
}

interface BondWithFractions {
  bond_id: string
  bond_name: string
  issuer: string
  sector: string
  face_value: number
  current_price: number
  yield_to_maturity: number
  rating: string
  maturity_date: string
  available_fraction: number
  min_investment: number
  total_volume: number
  daily_trades: number
}

interface Transaction {
  transaction_id: string
  status: string
  message: string
  bond_id: string
  fraction_amount: number
  price: number
  total_cost: number
  timestamp: string
}

const FractionalOwnershipWidget: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'available' | 'positions'>('available')
  const [availableBonds, setAvailableBonds] = useState<BondWithFractions[]>([])
  const [positions, setPositions] = useState<FractionalPosition[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [buyModal, setBuyModal] = useState<{ bond: BondWithFractions | null; isOpen: boolean }>({ bond: null, isOpen: false })
  const [sellModal, setSellModal] = useState<{ position: FractionalPosition | null; isOpen: boolean }>({ position: null, isOpen: false })
  const [transaction, setTransaction] = useState<Transaction | null>(null)

  // Form states
  const [buyAmount, setBuyAmount] = useState<number>(0.1)
  const [sellAmount, setSellAmount] = useState<number>(0.1)
  const [maxPrice, setMaxPrice] = useState<number>(0)
  const [minPrice, setMinPrice] = useState<number>(0)

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 30000) // Refresh every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchData = async () => {
    try {
      setLoading(true)
      const [bondsData, positionsData] = await Promise.all([
        apiService.getAvailableBonds(),
        apiService.getFractionalPositions()
      ])
      setAvailableBonds(bondsData)
      setPositions(positionsData)
      setError(null)
    } catch (err) {
      setError('Failed to fetch data')
      console.error('Error fetching fractional bonds data:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleBuyFraction = async (bond: BondWithFractions) => {
    try {
      const response = await apiService.buyBondFraction({
        bond_id: bond.bond_id,
        fraction_amount: buyAmount,
        max_price: maxPrice || undefined
      })
      
      setTransaction(response)
      setBuyModal({ bond: null, isOpen: false })
      setBuyAmount(0.1)
      setMaxPrice(0)
      
      // Refresh data
      await fetchData()
      
      // Clear transaction after 5 seconds
      setTimeout(() => setTransaction(null), 5000)
    } catch (err) {
      setError('Failed to buy bond fraction')
      console.error('Error buying bond fraction:', err)
    }
  }

  const handleSellFraction = async (position: FractionalPosition) => {
    try {
      const response = await apiService.sellBondFraction({
        bond_id: position.bond_id,
        fraction_amount: sellAmount,
        min_price: minPrice || undefined
      })
      
      setTransaction(response)
      setSellModal({ position: null, isOpen: false })
      setSellAmount(0.1)
      setMinPrice(0)
      
      // Refresh data
      await fetchData()
      
      // Clear transaction after 5 seconds
      setTimeout(() => setTransaction(null), 5000)
    } catch (err) {
      setError('Failed to sell bond fraction')
      console.error('Error selling bond fraction:', err)
    }
  }

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-IN', { 
      style: 'currency', 
      currency: 'INR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount)
  }

  const formatPercentage = (value: number) => {
    return `${value.toFixed(2)}%`
  }

  const getSectorColor = (sector: string) => {
    const colors: Record<string, string> = {
      'Financial Services': 'bg-blue-500/20 text-blue-400',
      'Energy': 'bg-green-500/20 text-green-400',
      'Infrastructure': 'bg-purple-500/20 text-purple-400',
      'Telecommunications': 'bg-orange-500/20 text-orange-400',
      'Manufacturing': 'bg-red-500/20 text-red-400',
      'Unknown': 'bg-gray-500/20 text-gray-400'
    }
    return colors[sector] || colors['Unknown']
  }

  const getRatingColor = (rating: string) => {
    if (rating.startsWith('AAA') || rating.startsWith('AA')) return 'text-green-400'
    if (rating.startsWith('A') || rating.startsWith('BBB')) return 'text-yellow-400'
    if (rating.startsWith('BB') || rating.startsWith('B')) return 'text-orange-400'
    return 'text-red-400'
  }

  return (
    <div className="bg-background-secondary rounded-xl border border-border p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-text-primary">Fractional Bond Ownership</h3>
          <p className="text-sm text-text-secondary">Buy and manage fractions of corporate bonds</p>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setActiveTab('available')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'available'
                ? 'bg-accent-blue text-white'
                : 'text-text-secondary hover:text-text-primary hover:bg-background-tertiary'
            }`}
          >
            Available Bonds
          </button>
          <button
            onClick={() => setActiveTab('positions')}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeTab === 'positions'
                ? 'bg-accent-blue text-white'
                : 'text-text-secondary hover:text-text-primary hover:bg-background-tertiary'
            }`}
          >
            My Positions ({positions.length})
          </button>
        </div>
      </div>

      {/* Transaction Status */}
      {transaction && (
        <div className="mb-4 p-4 rounded-lg bg-accent-green/10 border border-accent-green/20">
          <div className="flex items-center space-x-2">
            <CheckCircle className="h-5 w-5 text-accent-green" />
            <span className="text-accent-green font-medium">{transaction.message}</span>
          </div>
          <div className="text-sm text-text-secondary mt-1">
            Transaction ID: {transaction.transaction_id}
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mb-4 p-4 rounded-lg bg-accent-red/10 border border-accent-red/20">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-5 w-5 text-accent-red" />
            <span className="text-accent-red">{error}</span>
          </div>
        </div>
      )}

      {/* Loading State */}
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-accent-blue" />
          <span className="ml-2 text-text-secondary">Loading bonds...</span>
        </div>
      ) : (
        <>
          {/* Available Bonds Tab */}
          {activeTab === 'available' && (
            <div className="space-y-4">
              <div className="text-sm text-text-secondary mb-4">
                Showing {availableBonds.length} bonds available for fractional investment
              </div>
              
              {availableBonds.map((bond) => (
                <div key={bond.bond_id} className="p-4 rounded-lg border border-border bg-background hover:bg-background-tertiary transition-colors">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3 mb-2">
                        <h4 className="font-medium text-text-primary">{bond.issuer}</h4>
                        <span className={`px-2 py-1 rounded text-xs ${getSectorColor(bond.sector)}`}>
                          {bond.sector}
                        </span>
                        <span className={`text-xs font-medium ${getRatingColor(bond.rating)}`}>
                          {bond.rating}
                        </span>
                      </div>
                      
                      <p className="text-sm text-text-secondary mb-3 line-clamp-1">
                        {bond.bond_name}
                      </p>
                      
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-text-secondary">Current Price</span>
                          <div className="text-text-primary font-medium">{formatCurrency(bond.current_price)}</div>
                        </div>
                        <div>
                          <span className="text-text-secondary">YTM</span>
                          <div className="text-accent-green font-medium">{formatPercentage(bond.yield_to_maturity)}</div>
                        </div>
                        <div>
                          <span className="text-text-secondary">Available</span>
                          <div className="text-text-primary font-medium">{formatPercentage(bond.available_fraction * 100)}</div>
                        </div>
                        <div>
                          <span className="text-text-secondary">Min Investment</span>
                          <div className="text-text-primary font-medium">{formatCurrency(bond.min_investment)}</div>
                        </div>
                      </div>
                    </div>
                    
                    <button
                      onClick={() => setBuyModal({ bond, isOpen: true })}
                      className="ml-4 flex items-center space-x-2 px-4 py-2 bg-accent-blue hover:bg-accent-blue/90 text-white rounded-lg transition-colors"
                    >
                      <ShoppingCart className="h-4 w-4" />
                      <span>Buy Fraction</span>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Positions Tab */}
          {activeTab === 'positions' && (
            <div className="space-y-4">
              {positions.length === 0 ? (
                <div className="text-center py-12">
                  <Building className="h-12 w-12 text-text-muted mx-auto mb-4" />
                  <h4 className="text-lg font-medium text-text-primary mb-2">No Positions Yet</h4>
                  <p className="text-text-secondary">Start by buying fractional bonds from the Available Bonds tab</p>
                </div>
              ) : (
                <>
                  <div className="text-sm text-text-secondary mb-4">
                    You own fractions of {positions.length} different bonds
                  </div>
                  
                  {positions.map((position) => (
                    <div key={position.bond_id} className="p-4 rounded-lg border border-border bg-background hover:bg-background-tertiary transition-colors">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-2">
                            <h4 className="font-medium text-text-primary">{position.issuer}</h4>
                            <span className={`px-2 py-1 rounded text-xs ${getSectorColor(position.sector)}`}>
                              {position.sector}
                            </span>
                          </div>
                          
                          <p className="text-sm text-text-secondary mb-3 line-clamp-1">
                            {position.bond_name}
                          </p>
                          
                          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 text-sm">
                            <div>
                              <span className="text-text-secondary">Ownership</span>
                              <div className="text-text-primary font-medium">{formatPercentage(position.ownership_percentage)}</div>
                            </div>
                            <div>
                              <span className="text-text-secondary">Current Value</span>
                              <div className="text-text-primary font-medium">{formatCurrency(position.current_value)}</div>
                            </div>
                            <div>
                              <span className="text-text-secondary">Purchase Value</span>
                              <div className="text-text-secondary">{formatCurrency(position.owned_value)}</div>
                            </div>
                            <div>
                              <span className="text-text-secondary">P&L</span>
                              <div className={`font-medium flex items-center space-x-1 ${
                                position.unrealized_pnl >= 0 ? 'text-accent-green' : 'text-accent-red'
                              }`}>
                                {position.unrealized_pnl >= 0 ? (
                                  <TrendingUp className="h-4 w-4" />
                                ) : (
                                  <TrendingDown className="h-4 w-4" />
                                )}
                                <span>{formatCurrency(Math.abs(position.unrealized_pnl))}</span>
                              </div>
                            </div>
                            <div>
                              <span className="text-text-secondary">YTM</span>
                              <div className="text-accent-green font-medium">{formatPercentage(position.yield_to_maturity)}</div>
                            </div>
                          </div>
                        </div>
                        
                        <button
                          onClick={() => setSellModal({ position, isOpen: true })}
                          className="ml-4 flex items-center space-x-2 px-4 py-2 bg-accent-red hover:bg-accent-red/90 text-white rounded-lg transition-colors"
                        >
                          <DollarSign className="h-4 w-4" />
                          <span>Sell Fraction</span>
                        </button>
                      </div>
                    </div>
                  ))}
                </>
              )}
            </div>
          )}
        </>
      )}

      {/* Buy Modal */}
      {buyModal.isOpen && buyModal.bond && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-background-secondary rounded-xl p-6 w-full max-w-md border border-border">
            <h3 className="text-lg font-semibold text-text-primary mb-4">
              Buy Fraction - {buyModal.bond.issuer}
            </h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Fraction Amount (0-{buyModal.bond.available_fraction})
                </label>
                <input
                  type="number"
                  min="0.01"
                  max={buyModal.bond.available_fraction}
                  step="0.01"
                  value={buyAmount}
                  onChange={(e) => setBuyAmount(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-border rounded-lg bg-background text-text-primary"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Max Price (Optional)
                </label>
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  value={maxPrice}
                  onChange={(e) => setMaxPrice(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-border rounded-lg bg-background text-text-primary"
                  placeholder={`Current: ${formatCurrency(buyModal.bond.current_price)}`}
                />
              </div>
              
              <div className="p-3 bg-background rounded-lg">
                <div className="text-sm text-text-secondary">Estimated Cost:</div>
                <div className="text-lg font-semibold text-text-primary">
                  {formatCurrency(buyAmount * buyModal.bond.current_price * buyModal.bond.face_value / 100)}
                </div>
              </div>
            </div>
            
            <div className="flex space-x-3 mt-6">
              <button
                onClick={() => setBuyModal({ bond: null, isOpen: false })}
                className="flex-1 px-4 py-2 border border-border text-text-primary rounded-lg hover:bg-background-tertiary transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => handleBuyFraction(buyModal.bond!)}
                className="flex-1 px-4 py-2 bg-accent-blue hover:bg-accent-blue/90 text-white rounded-lg transition-colors"
              >
                Buy Fraction
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Sell Modal */}
      {sellModal.isOpen && sellModal.position && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-background-secondary rounded-xl p-6 w-full max-w-md border border-border">
            <h3 className="text-lg font-semibold text-text-primary mb-4">
              Sell Fraction - {sellModal.position.issuer}
            </h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Fraction Amount (0-{sellModal.position.owned_fraction})
                </label>
                <input
                  type="number"
                  min="0.01"
                  max={sellModal.position.owned_fraction}
                  step="0.01"
                  value={sellAmount}
                  onChange={(e) => setSellAmount(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-border rounded-lg bg-background text-text-primary"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Min Price (Optional)
                </label>
                <input
                  type="number"
                  min="0"
                  step="0.01"
                  value={minPrice}
                  onChange={(e) => setMinPrice(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-border rounded-lg bg-background text-text-primary"
                  placeholder={`Current: ${formatCurrency(sellModal.position.current_price)}`}
                />
              </div>
              
              <div className="p-3 bg-background rounded-lg">
                <div className="text-sm text-text-secondary">Estimated Proceeds:</div>
                <div className="text-lg font-semibold text-text-primary">
                  {formatCurrency(sellAmount * sellModal.position.current_price * sellModal.position.face_value / 100)}
                </div>
              </div>
            </div>
            
            <div className="flex space-x-3 mt-6">
              <button
                onClick={() => setSellModal({ position: null, isOpen: false })}
                className="flex-1 px-4 py-2 border border-border text-text-primary rounded-lg hover:bg-background-tertiary transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => handleSellFraction(sellModal.position!)}
                className="flex-1 px-4 py-2 bg-accent-red hover:bg-accent-red/90 text-white rounded-lg transition-colors"
              >
                Sell Fraction
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default FractionalOwnershipWidget
