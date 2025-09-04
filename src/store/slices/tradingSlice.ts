import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

export interface Order {
  id: string
  isin: string
  bondName: string
  side: 'BUY' | 'SELL'
  orderType: 'MARKET' | 'LIMIT' | 'STOP' | 'STOP_LIMIT'
  quantity: number
  price?: number
  stopPrice?: number
  timeInForce: 'DAY' | 'GTC' | 'IOC' | 'FOK'
  status: 'PENDING' | 'PARTIAL' | 'FILLED' | 'CANCELLED' | 'REJECTED'
  filledQuantity: number
  averageFillPrice?: number
  remainingQuantity: number
  orderTime: string
  lastUpdateTime: string
  fees?: number
  commission?: number
  notes?: string
}

export interface Trade {
  id: string
  orderId: string
  isin: string
  bondName: string
  side: 'BUY' | 'SELL'
  quantity: number
  price: number
  totalAmount: number
  fees: number
  commission: number
  tradeTime: string
  settlementDate: string
  counterparty?: string
}

export interface OrderBook {
  isin: string
  bids: Array<{
    price: number
    quantity: number
    count: number
  }>
  asks: Array<{
    price: number
    quantity: number
    count: number
  }>
  lastUpdate: string
}

export interface TradingLimits {
  maxOrderSize: number
  maxDailyVolume: number
  maxPositionSize: number
  availableBuyingPower: number
  usedBuyingPower: number
  marginRequirement: number
}

interface TradingState {
  orders: Order[]
  trades: Trade[]
  orderBooks: { [isin: string]: OrderBook }
  tradingLimits: TradingLimits | null
  selectedOrder: Order | null
  selectedTrade: Trade | null
  activeOrders: Order[]
  recentTrades: Trade[]
  loading: boolean
  error: string | null
  orderFormData: {
    isin?: string
    side?: 'BUY' | 'SELL'
    orderType?: 'MARKET' | 'LIMIT'
    quantity?: number
    price?: number
  }
}

const initialState: TradingState = {
  orders: [],
  trades: [],
  orderBooks: {},
  tradingLimits: null,
  selectedOrder: null,
  selectedTrade: null,
  activeOrders: [],
  recentTrades: [],
  loading: false,
  error: null,
  orderFormData: {},
}

// Async thunks
export const fetchOrders = createAsyncThunk(
  'trading/fetchOrders',
  async (params?: { status?: string; limit?: number }, { rejectWithValue }) => {
    try {
      const searchParams = new URLSearchParams()
      if (params?.status) searchParams.append('status', params.status)
      if (params?.limit) searchParams.append('limit', params.limit.toString())
      
      const response = await fetch(`/api/v1/trading/orders?${searchParams}`)
      if (!response.ok) {
        throw new Error('Failed to fetch orders')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const fetchTrades = createAsyncThunk(
  'trading/fetchTrades',
  async (params: { limit?: number; offset?: number } = {}, { rejectWithValue }) => {
    try {
      const searchParams = new URLSearchParams()
      if (params?.limit) searchParams.append('limit', params.limit.toString())
      if (params?.offset) searchParams.append('offset', params.offset.toString())
      
      const response = await fetch(`/api/v1/trading/trades?${searchParams}`)
      if (!response.ok) {
        throw new Error('Failed to fetch trades')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const fetchOrderBook = createAsyncThunk(
  'trading/fetchOrderBook',
  async (isin: string, { rejectWithValue }) => {
    try {
      const response = await fetch(`/api/v1/trading/orderbook/${isin}`)
      if (!response.ok) {
        throw new Error('Failed to fetch order book')
      }
      const data = await response.json()
      return { isin, orderBook: data }
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const fetchTradingLimits = createAsyncThunk(
  'trading/fetchTradingLimits',
  async (_, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/v1/trading/limits')
      if (!response.ok) {
        throw new Error('Failed to fetch trading limits')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const submitOrder = createAsyncThunk(
  'trading/submitOrder',
  async (orderData: Partial<Order>, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/v1/trading/orders', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(orderData),
      })
      if (!response.ok) {
        throw new Error('Failed to submit order')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const cancelOrder = createAsyncThunk(
  'trading/cancelOrder',
  async (orderId: string, { rejectWithValue }) => {
    try {
      const response = await fetch(`/api/v1/trading/orders/${orderId}/cancel`, {
        method: 'POST',
      })
      if (!response.ok) {
        throw new Error('Failed to cancel order')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

const tradingSlice = createSlice({
  name: 'trading',
  initialState,
  reducers: {
    setSelectedOrder: (state, action: PayloadAction<Order | null>) => {
      state.selectedOrder = action.payload
    },
    setSelectedTrade: (state, action: PayloadAction<Trade | null>) => {
      state.selectedTrade = action.payload
    },
    setOrderFormData: (state, action: PayloadAction<TradingState['orderFormData']>) => {
      state.orderFormData = { ...state.orderFormData, ...action.payload }
    },
    clearOrderFormData: (state) => {
      state.orderFormData = {}
    },
    updateOrderStatus: (state, action: PayloadAction<{ orderId: string; status: Order['status']; filledQuantity?: number; averageFillPrice?: number }>) => {
      const order = state.orders.find(o => o.id === action.payload.orderId)
      if (order) {
        order.status = action.payload.status
        if (action.payload.filledQuantity !== undefined) {
          order.filledQuantity = action.payload.filledQuantity
          order.remainingQuantity = order.quantity - action.payload.filledQuantity
        }
        if (action.payload.averageFillPrice !== undefined) {
          order.averageFillPrice = action.payload.averageFillPrice
        }
        order.lastUpdateTime = new Date().toISOString()
      }
    },
    addTrade: (state, action: PayloadAction<Trade>) => {
      state.trades.unshift(action.payload)
      state.recentTrades = state.trades.slice(0, 10)
    },
    updateOrderBook: (state, action: PayloadAction<{ isin: string; orderBook: OrderBook }>) => {
      state.orderBooks[action.payload.isin] = action.payload.orderBook
    },
    clearError: (state) => {
      state.error = null
    },
  },
  extraReducers: (builder) => {
    // Fetch orders
    builder
      .addCase(fetchOrders.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchOrders.fulfilled, (state, action) => {
        state.loading = false
        state.orders = action.payload
        state.activeOrders = action.payload.filter((order: Order) => 
          ['PENDING', 'PARTIAL'].includes(order.status)
        )
      })
      .addCase(fetchOrders.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Fetch trades
    builder
      .addCase(fetchTrades.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchTrades.fulfilled, (state, action) => {
        state.loading = false
        state.trades = action.payload
        state.recentTrades = action.payload.slice(0, 10)
      })
      .addCase(fetchTrades.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Fetch order book
    builder
      .addCase(fetchOrderBook.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchOrderBook.fulfilled, (state, action) => {
        state.loading = false
        state.orderBooks[action.payload.isin] = action.payload.orderBook
      })
      .addCase(fetchOrderBook.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Fetch trading limits
    builder
      .addCase(fetchTradingLimits.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchTradingLimits.fulfilled, (state, action) => {
        state.loading = false
        state.tradingLimits = action.payload
      })
      .addCase(fetchTradingLimits.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Submit order
    builder
      .addCase(submitOrder.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(submitOrder.fulfilled, (state, action) => {
        state.loading = false
        state.orders.unshift(action.payload)
        if (['PENDING', 'PARTIAL'].includes(action.payload.status)) {
          state.activeOrders.unshift(action.payload)
        }
      })
      .addCase(submitOrder.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Cancel order
    builder
      .addCase(cancelOrder.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(cancelOrder.fulfilled, (state, action) => {
        state.loading = false
        const order = state.orders.find(o => o.id === action.payload.id)
        if (order) {
          order.status = 'CANCELLED'
          order.lastUpdateTime = new Date().toISOString()
        }
        state.activeOrders = state.activeOrders.filter(o => o.id !== action.payload.id)
      })
      .addCase(cancelOrder.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })
  },
})

export const {
  setSelectedOrder,
  setSelectedTrade,
  setOrderFormData,
  clearOrderFormData,
  updateOrderStatus,
  addTrade,
  updateOrderBook,
  clearError,
} = tradingSlice.actions

// Selectors
export const selectOrders = (state: { trading: TradingState }) => state.trading.orders
export const selectTrades = (state: { trading: TradingState }) => state.trading.trades
export const selectOrderBooks = (state: { trading: TradingState }) => state.trading.orderBooks
export const selectTradingLimits = (state: { trading: TradingState }) => state.trading.tradingLimits
export const selectSelectedOrder = (state: { trading: TradingState }) => state.trading.selectedOrder
export const selectSelectedTrade = (state: { trading: TradingState }) => state.trading.selectedTrade
export const selectActiveOrders = (state: { trading: TradingState }) => state.trading.activeOrders
export const selectRecentTrades = (state: { trading: TradingState }) => state.trading.recentTrades
export const selectTradingLoading = (state: { trading: TradingState }) => state.trading.loading
export const selectTradingError = (state: { trading: TradingState }) => state.trading.error
export const selectOrderFormData = (state: { trading: TradingState }) => state.trading.orderFormData

export default tradingSlice.reducer
