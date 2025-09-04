import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

export interface Position {
  id: string
  isin: string
  bondName: string
  quantity: number
  averagePrice: number
  currentPrice: number
  marketValue: number
  unrealizedPnL: number
  unrealizedPnLPercent: number
  duration: number
  convexity: number
  yieldToMaturity: number
  creditRating: string
  sector: string
  maturityDate: string
  couponRate: number
  lastUpdated: string
}

export interface PortfolioSummary {
  totalValue: number
  totalUnrealizedPnL: number
  totalUnrealizedPnLPercent: number
  totalCash: number
  totalInvested: number
  averageDuration: number
  averageYield: number
  creditExposure: {
    [rating: string]: number
  }
  sectorExposure: {
    [sector: string]: number
  }
  maturityProfile: {
    [bucket: string]: number
  }
  riskMetrics: {
    var95: number
    var99: number
    expectedShortfall: number
    volatility: number
    sharpeRatio: number
    maxDrawdown: number
  }
}

export interface Transaction {
  id: string
  isin: string
  bondName: string
  type: 'BUY' | 'SELL'
  quantity: number
  price: number
  totalAmount: number
  fees: number
  netAmount: number
  settlementDate: string
  tradeDate: string
  status: 'PENDING' | 'SETTLED' | 'FAILED'
}

interface PortfolioState {
  positions: Position[]
  summary: PortfolioSummary | null
  transactions: Transaction[]
  selectedPosition: Position | null
  loading: boolean
  error: string | null
  filters: {
    sector?: string
    creditRating?: string
    maturityRange?: [string, string]
  }
  sortBy: string
  sortOrder: 'asc' | 'desc'
}

const initialState: PortfolioState = {
  positions: [],
  summary: null,
  transactions: [],
  selectedPosition: null,
  loading: false,
  error: null,
  filters: {},
  sortBy: 'marketValue',
  sortOrder: 'desc',
}

// Async thunks
export const fetchPortfolio = createAsyncThunk(
  'portfolio/fetchPortfolio',
  async (_, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/v1/portfolio')
      if (!response.ok) {
        throw new Error('Failed to fetch portfolio')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const fetchPortfolioSummary = createAsyncThunk(
  'portfolio/fetchPortfolioSummary',
  async (_, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/v1/portfolio/summary')
      if (!response.ok) {
        throw new Error('Failed to fetch portfolio summary')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const fetchTransactions = createAsyncThunk(
  'portfolio/fetchTransactions',
  async (params?: { limit?: number; offset?: number }, { rejectWithValue }) => {
    try {
      const searchParams = new URLSearchParams()
      if (params?.limit) searchParams.append('limit', params.limit.toString())
      if (params?.offset) searchParams.append('offset', params.offset.toString())
      
      const response = await fetch(`/api/v1/portfolio/transactions?${searchParams}`)
      if (!response.ok) {
        throw new Error('Failed to fetch transactions')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const addPosition = createAsyncThunk(
  'portfolio/addPosition',
  async (positionData: Partial<Position>, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/v1/portfolio/positions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(positionData),
      })
      if (!response.ok) {
        throw new Error('Failed to add position')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const updatePosition = createAsyncThunk(
  'portfolio/updatePosition',
  async ({ id, updates }: { id: string; updates: Partial<Position> }, { rejectWithValue }) => {
    try {
      const response = await fetch(`/api/v1/portfolio/positions/${id}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updates),
      })
      if (!response.ok) {
        throw new Error('Failed to update position')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

const portfolioSlice = createSlice({
  name: 'portfolio',
  initialState,
  reducers: {
    setSelectedPosition: (state, action: PayloadAction<Position | null>) => {
      state.selectedPosition = action.payload
    },
    setFilters: (state, action: PayloadAction<PortfolioState['filters']>) => {
      state.filters = { ...state.filters, ...action.payload }
    },
    clearFilters: (state) => {
      state.filters = {}
    },
    setSorting: (state, action: PayloadAction<{ sortBy: string; sortOrder: 'asc' | 'desc' }>) => {
      state.sortBy = action.payload.sortBy
      state.sortOrder = action.payload.sortOrder
    },
    updatePositionPrice: (state, action: PayloadAction<{ isin: string; price: number }>) => {
      const position = state.positions.find(p => p.isin === action.payload.isin)
      if (position) {
        position.currentPrice = action.payload.price
        position.marketValue = position.quantity * action.payload.price
        position.unrealizedPnL = position.marketValue - (position.quantity * position.averagePrice)
        position.unrealizedPnLPercent = (position.unrealizedPnL / (position.quantity * position.averagePrice)) * 100
        position.lastUpdated = new Date().toISOString()
      }
    },
    clearError: (state) => {
      state.error = null
    },
  },
  extraReducers: (builder) => {
    // Fetch portfolio
    builder
      .addCase(fetchPortfolio.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchPortfolio.fulfilled, (state, action) => {
        state.loading = false
        state.positions = action.payload
      })
      .addCase(fetchPortfolio.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Fetch portfolio summary
    builder
      .addCase(fetchPortfolioSummary.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchPortfolioSummary.fulfilled, (state, action) => {
        state.loading = false
        state.summary = action.payload
      })
      .addCase(fetchPortfolioSummary.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Fetch transactions
    builder
      .addCase(fetchTransactions.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchTransactions.fulfilled, (state, action) => {
        state.loading = false
        state.transactions = action.payload
      })
      .addCase(fetchTransactions.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Add position
    builder
      .addCase(addPosition.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(addPosition.fulfilled, (state, action) => {
        state.loading = false
        state.positions.push(action.payload)
      })
      .addCase(addPosition.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Update position
    builder
      .addCase(updatePosition.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(updatePosition.fulfilled, (state, action) => {
        state.loading = false
        const index = state.positions.findIndex(p => p.id === action.payload.id)
        if (index !== -1) {
          state.positions[index] = action.payload
        }
      })
      .addCase(updatePosition.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })
  },
})

export const {
  setSelectedPosition,
  setFilters,
  clearFilters,
  setSorting,
  updatePositionPrice,
  clearError,
} = portfolioSlice.actions

// Selectors
export const selectPositions = (state: { portfolio: PortfolioState }) => state.portfolio.positions
export const selectPortfolioSummary = (state: { portfolio: PortfolioState }) => state.portfolio.summary
export const selectTransactions = (state: { portfolio: PortfolioState }) => state.portfolio.transactions
export const selectSelectedPosition = (state: { portfolio: PortfolioState }) => state.portfolio.selectedPosition
export const selectPortfolioLoading = (state: { portfolio: PortfolioState }) => state.portfolio.loading
export const selectPortfolioError = (state: { portfolio: PortfolioState }) => state.portfolio.error
export const selectPortfolioFilters = (state: { portfolio: PortfolioState }) => state.portfolio.filters

export default portfolioSlice.reducer
