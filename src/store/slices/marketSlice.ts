import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

export interface Bond {
  isin: string
  name: string
  issuer: string
  couponRate: number
  maturityDate: string
  currentPrice: number
  yieldToMaturity: number
  duration: number
  convexity: number
  creditRating: string
  sector: string
  currency: string
  faceValue: number
  outstandingAmount: number
  lastTradeDate?: string
  volume?: number
  change?: number
  changePercent?: number
}

export interface MarketData {
  timestamp: string
  bonds: Bond[]
  indices: {
    [key: string]: {
      value: number
      change: number
      changePercent: number
    }
  }
  yieldCurve: {
    [tenure: string]: number
  }
}

interface MarketState {
  data: MarketData | null
  bonds: Bond[]
  selectedBond: Bond | null
  loading: boolean
  error: string | null
  lastUpdate: string | null
  filters: {
    sector?: string
    creditRating?: string
    maturityRange?: [string, string]
    yieldRange?: [number, number]
    search?: string
  }
  sortBy: string
  sortOrder: 'asc' | 'desc'
}

const initialState: MarketState = {
  data: null,
  bonds: [],
  selectedBond: null,
  loading: false,
  error: null,
  lastUpdate: null,
  filters: {},
  sortBy: 'name',
  sortOrder: 'asc',
}

// Async thunks
export const fetchMarketData = createAsyncThunk(
  'market/fetchMarketData',
  async (_, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/v1/market/data')
      if (!response.ok) {
        throw new Error('Failed to fetch market data')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const fetchBonds = createAsyncThunk(
  'market/fetchBonds',
  async (filters?: any, { rejectWithValue }) => {
    try {
      const params = new URLSearchParams(filters)
      const response = await fetch(`/api/v1/bonds?${params}`)
      if (!response.ok) {
        throw new Error('Failed to fetch bonds')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const fetchBondDetails = createAsyncThunk(
  'market/fetchBondDetails',
  async (isin: string, { rejectWithValue }) => {
    try {
      const response = await fetch(`/api/v1/bonds/${isin}`)
      if (!response.ok) {
        throw new Error('Failed to fetch bond details')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

const marketSlice = createSlice({
  name: 'market',
  initialState,
  reducers: {
    setSelectedBond: (state, action: PayloadAction<Bond | null>) => {
      state.selectedBond = action.payload
    },
    setFilters: (state, action: PayloadAction<MarketState['filters']>) => {
      state.filters = { ...state.filters, ...action.payload }
    },
    clearFilters: (state) => {
      state.filters = {}
    },
    setSorting: (state, action: PayloadAction<{ sortBy: string; sortOrder: 'asc' | 'desc' }>) => {
      state.sortBy = action.payload.sortBy
      state.sortOrder = action.payload.sortOrder
    },
    updateBondPrice: (state, action: PayloadAction<{ isin: string; price: number; change: number }>) => {
      const bond = state.bonds.find(b => b.isin === action.payload.isin)
      if (bond) {
        bond.currentPrice = action.payload.price
        bond.change = action.payload.change
        bond.changePercent = (action.payload.change / (action.payload.price - action.payload.change)) * 100
      }
    },
    clearError: (state) => {
      state.error = null
    },
  },
  extraReducers: (builder) => {
    // Fetch market data
    builder
      .addCase(fetchMarketData.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchMarketData.fulfilled, (state, action) => {
        state.loading = false
        state.data = action.payload
        state.lastUpdate = new Date().toISOString()
      })
      .addCase(fetchMarketData.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Fetch bonds
    builder
      .addCase(fetchBonds.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchBonds.fulfilled, (state, action) => {
        state.loading = false
        state.bonds = action.payload
      })
      .addCase(fetchBonds.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Fetch bond details
    builder
      .addCase(fetchBondDetails.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchBondDetails.fulfilled, (state, action) => {
        state.loading = false
        state.selectedBond = action.payload
      })
      .addCase(fetchBondDetails.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })
  },
})

export const {
  setSelectedBond,
  setFilters,
  clearFilters,
  setSorting,
  updateBondPrice,
  clearError,
} = marketSlice.actions

// Selectors
export const selectMarketData = (state: { market: MarketState }) => state.market.data
export const selectBonds = (state: { market: MarketState }) => state.market.bonds
export const selectSelectedBond = (state: { market: MarketState }) => state.market.selectedBond
export const selectMarketLoading = (state: { market: MarketState }) => state.market.loading
export const selectMarketError = (state: { market: MarketState }) => state.market.error
export const selectMarketFilters = (state: { market: MarketState }) => state.market.filters
export const selectMarketSorting = (state: { market: MarketState }) => ({
  sortBy: state.market.sortBy,
  sortOrder: state.market.sortOrder,
})

export default marketSlice.reducer
