import React, { useState, useEffect } from 'react'
import { 
  Clock, 
  Gavel, 
  TrendingUp, 
  Users, 
  Timer, 
  DollarSign,
  AlertCircle,
  CheckCircle,
  Loader2,
  Award
} from 'lucide-react'
import { apiService } from '@/services/api'

interface AuctionSummary {
  auction_id: string
  bond_id: string
  bond_name: string
  issuer: string
  current_highest_bid: number
  available_fraction: number
  end_time: string
  time_remaining: number
  total_bids: number
}

interface BidRequest {
  auction_id: string
  bid_amount: number
  fraction_amount: number
}

interface BidResponse {
  bid_id: string
  status: string
  message: string
  auction_id: string
  bid_amount: number
  fraction_amount: number
  timestamp: string
  position: number
}

const AuctionWidget: React.FC = () => {
  const [auctions, setAuctions] = useState<AuctionSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [bidModal, setBidModal] = useState<{ auction: AuctionSummary | null; isOpen: boolean }>({ auction: null, isOpen: false })
  const [bidResult, setBidResult] = useState<BidResponse | null>(null)
  
  // Form states
  const [bidAmount, setBidAmount] = useState<number>(0)
  const [fractionAmount, setFractionAmount] = useState<number>(0.1)

  useEffect(() => {
    fetchAuctions()
    const interval = setInterval(fetchAuctions, 10000) // Refresh every 10 seconds
    return () => clearInterval(interval)
  }, [])

  const fetchAuctions = async () => {
    try {
      setLoading(true)
      const data = await apiService.getActiveAuctions()
      setAuctions(data)
      setError(null)
    } catch (err) {
      setError('Failed to fetch auctions')
      console.error('Error fetching auctions:', err)
    } finally {
      setLoading(false)
    }
  }

  const handlePlaceBid = async (auction: AuctionSummary) => {
    try {
      const response = await apiService.placeBid({
        auction_id: auction.auction_id,
        bid_amount: bidAmount,
        fraction_amount: fractionAmount
      })
      
      setBidResult(response)
      setBidModal({ auction: null, isOpen: false })
      setBidAmount(0)
      setFractionAmount(0.1)
      
      // Refresh auctions
      await fetchAuctions()
      
      // Clear bid result after 5 seconds
      setTimeout(() => setBidResult(null), 5000)
    } catch (err) {
      setError('Failed to place bid')
      console.error('Error placing bid:', err)
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

  const formatTimeRemaining = (seconds: number) => {
    if (seconds <= 0) return 'Ended'
    
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const remainingSeconds = seconds % 60
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${remainingSeconds}s`
    } else if (minutes > 0) {
      return `${minutes}m ${remainingSeconds}s`
    } else {
      return `${remainingSeconds}s`
    }
  }

  const getUrgencyColor = (timeRemaining: number) => {
    if (timeRemaining <= 300) return 'text-accent-red' // Last 5 minutes
    if (timeRemaining <= 900) return 'text-accent-amber' // Last 15 minutes
    return 'text-accent-green'
  }

  const getUrgencyBg = (timeRemaining: number) => {
    if (timeRemaining <= 300) return 'bg-accent-red/10 border-accent-red/20'
    if (timeRemaining <= 900) return 'bg-accent-amber/10 border-accent-amber/20'
    return 'bg-accent-green/10 border-accent-green/20'
  }

  return (
    <div className="bg-background-secondary rounded-xl border border-border p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold text-text-primary flex items-center space-x-2">
            <Gavel className="h-5 w-5 text-accent-blue" />
            <span>Live Bond Auctions</span>
          </h3>
          <p className="text-sm text-text-secondary">Participate in real-time bond fraction auctions</p>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-accent-green animate-pulse"></div>
          <span className="text-sm text-text-secondary">Live</span>
        </div>
      </div>

      {/* Bid Result */}
      {bidResult && (
        <div className="mb-4 p-4 rounded-lg bg-accent-green/10 border border-accent-green/20">
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-5 w-5 text-accent-green" />
              <div>
                <span className="text-accent-green font-medium">{bidResult.message}</span>
                <div className="text-sm text-text-secondary mt-1">
                  Position #{bidResult.position} • Bid ID: {bidResult.bid_id}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-lg font-semibold text-text-primary">
                {formatCurrency(bidResult.bid_amount)}
              </div>
              <div className="text-sm text-text-secondary">
                {(bidResult.fraction_amount * 100).toFixed(1)}% fraction
              </div>
            </div>
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
          <span className="ml-2 text-text-secondary">Loading auctions...</span>
        </div>
      ) : (
        <div className="space-y-4">
          {auctions.length === 0 ? (
            <div className="text-center py-12">
              <Gavel className="h-12 w-12 text-text-muted mx-auto mb-4" />
              <h4 className="text-lg font-medium text-text-primary mb-2">No Active Auctions</h4>
              <p className="text-text-secondary">Check back later for new bond auction opportunities</p>
            </div>
          ) : (
            <>
              <div className="text-sm text-text-secondary mb-4">
                {auctions.length} active auction{auctions.length !== 1 ? 's' : ''} • Updated every 10 seconds
              </div>
              
              {auctions.map((auction) => (
                <div 
                  key={auction.auction_id} 
                  className={`p-4 rounded-lg border transition-all hover:shadow-lg ${getUrgencyBg(auction.time_remaining)}`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      {/* Auction Header */}
                      <div className="flex items-center justify-between mb-3">
                        <div>
                          <h4 className="font-semibold text-text-primary">{auction.issuer}</h4>
                          <p className="text-sm text-text-secondary line-clamp-1">{auction.bond_name}</p>
                        </div>
                        <div className="text-right">
                          <div className={`text-lg font-bold ${getUrgencyColor(auction.time_remaining)}`}>
                            {formatTimeRemaining(auction.time_remaining)}
                          </div>
                          <div className="text-xs text-text-secondary flex items-center justify-end space-x-1">
                            <Timer className="h-3 w-3" />
                            <span>Time Left</span>
                          </div>
                        </div>
                      </div>
                      
                      {/* Auction Stats */}
                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                        <div className="text-center p-3 bg-background rounded-lg">
                          <div className="text-xs text-text-secondary mb-1">Highest Bid</div>
                          <div className="text-lg font-bold text-accent-green">
                            {formatCurrency(auction.current_highest_bid)}
                          </div>
                          <TrendingUp className="h-4 w-4 text-accent-green mx-auto mt-1" />
                        </div>
                        
                        <div className="text-center p-3 bg-background rounded-lg">
                          <div className="text-xs text-text-secondary mb-1">Available</div>
                          <div className="text-lg font-bold text-accent-blue">
                            {(auction.available_fraction * 100).toFixed(1)}%
                          </div>
                          <DollarSign className="h-4 w-4 text-accent-blue mx-auto mt-1" />
                        </div>
                        
                        <div className="text-center p-3 bg-background rounded-lg">
                          <div className="text-xs text-text-secondary mb-1">Total Bids</div>
                          <div className="text-lg font-bold text-text-primary">
                            {auction.total_bids}
                          </div>
                          <Users className="h-4 w-4 text-text-primary mx-auto mt-1" />
                        </div>
                        
                        <div className="text-center p-3 bg-background rounded-lg">
                          <div className="text-xs text-text-secondary mb-1">Bond ID</div>
                          <div className="text-sm font-medium text-text-primary truncate">
                            {auction.bond_id}
                          </div>
                          <Award className="h-4 w-4 text-text-primary mx-auto mt-1" />
                        </div>
                      </div>
                      
                      {/* Progress Bar */}
                      <div className="mb-4">
                        <div className="flex items-center justify-between text-xs text-text-secondary mb-1">
                          <span>Auction Progress</span>
                          <span>{auction.total_bids} bids placed</span>
                        </div>
                        <div className="w-full bg-background rounded-full h-2">
                          <div 
                            className="bg-gradient-to-r from-accent-blue to-accent-green h-2 rounded-full transition-all duration-300"
                            style={{ 
                              width: `${Math.min(100, (auction.total_bids / 20) * 100)}%` 
                            }}
                          ></div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Bid Button */}
                    <div className="ml-4">
                      <button
                        onClick={() => {
                          setBidAmount(auction.current_highest_bid + 1)
                          setBidModal({ auction, isOpen: true })
                        }}
                        disabled={auction.time_remaining <= 0}
                        className={`flex flex-col items-center space-y-2 px-6 py-3 rounded-lg transition-all ${
                          auction.time_remaining <= 0
                            ? 'bg-gray-500/20 text-gray-400 cursor-not-allowed'
                            : auction.time_remaining <= 300
                            ? 'bg-accent-red hover:bg-accent-red/90 text-white animate-pulse'
                            : 'bg-accent-blue hover:bg-accent-blue/90 text-white hover:scale-105'
                        }`}
                      >
                        <Gavel className="h-5 w-5" />
                        <span className="text-sm font-medium">
                          {auction.time_remaining <= 0 ? 'Ended' : 'Place Bid'}
                        </span>
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      )}

      {/* Bid Modal */}
      {bidModal.isOpen && bidModal.auction && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-background-secondary rounded-xl p-6 w-full max-w-md border border-border">
            <div className="flex items-center space-x-2 mb-4">
              <Gavel className="h-5 w-5 text-accent-blue" />
              <h3 className="text-lg font-semibold text-text-primary">
                Place Bid - {bidModal.auction.issuer}
              </h3>
            </div>
            
            {/* Auction Info */}
            <div className="p-4 bg-background rounded-lg mb-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-text-secondary">Current Highest</span>
                  <div className="text-accent-green font-semibold">
                    {formatCurrency(bidModal.auction.current_highest_bid)}
                  </div>
                </div>
                <div>
                  <span className="text-text-secondary">Time Remaining</span>
                  <div className={`font-semibold ${getUrgencyColor(bidModal.auction.time_remaining)}`}>
                    {formatTimeRemaining(bidModal.auction.time_remaining)}
                  </div>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Your Bid Amount (Minimum: {formatCurrency(bidModal.auction.current_highest_bid + 1)})
                </label>
                <input
                  type="number"
                  min={bidModal.auction.current_highest_bid + 1}
                  step="1"
                  value={bidAmount}
                  onChange={(e) => setBidAmount(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-border rounded-lg bg-background text-text-primary text-lg font-semibold"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-text-primary mb-2">
                  Fraction Amount (0-{bidModal.auction.available_fraction})
                </label>
                <input
                  type="number"
                  min="0.01"
                  max={bidModal.auction.available_fraction}
                  step="0.01"
                  value={fractionAmount}
                  onChange={(e) => setFractionAmount(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 border border-border rounded-lg bg-background text-text-primary"
                />
                <div className="text-xs text-text-secondary mt-1">
                  Requesting {(fractionAmount * 100).toFixed(1)}% of available fraction
                </div>
              </div>
              
              <div className="p-3 bg-accent-blue/10 rounded-lg border border-accent-blue/20">
                <div className="text-sm text-text-secondary">Total Commitment:</div>
                <div className="text-xl font-bold text-accent-blue">
                  {formatCurrency(bidAmount * fractionAmount)}
                </div>
                <div className="text-xs text-text-secondary mt-1">
                  If your bid wins, you'll pay this amount for {(fractionAmount * 100).toFixed(1)}% fraction
                </div>
              </div>
            </div>
            
            <div className="flex space-x-3 mt-6">
              <button
                onClick={() => setBidModal({ auction: null, isOpen: false })}
                className="flex-1 px-4 py-2 border border-border text-text-primary rounded-lg hover:bg-background-tertiary transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => handlePlaceBid(bidModal.auction!)}
                disabled={bidAmount <= bidModal.auction.current_highest_bid || fractionAmount <= 0}
                className="flex-1 px-4 py-2 bg-accent-blue hover:bg-accent-blue/90 disabled:bg-gray-500/20 disabled:text-gray-400 text-white rounded-lg transition-colors flex items-center justify-center space-x-2"
              >
                <Gavel className="h-4 w-4" />
                <span>Place Bid</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default AuctionWidget
