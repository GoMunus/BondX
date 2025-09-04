import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

// Define minimal types locally to avoid import errors
interface WebSocketMessage {
  id: string
  type: string
  data: any
  timestamp: string
}

interface WebSocketConnection {
  id: string
  url: string
  status: ConnectionStatus
}

type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

// Async thunks
export const connectWebSocket = createAsyncThunk(
  'websocket/connect',
  async (_, { rejectWithValue }) => {
    try {
      // Mock WebSocket connection for demo
      const connection: WebSocketConnection = {
        id: '1',
        url: 'ws://localhost:8000/ws',
        status: 'connected'
      }
      return connection
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to connect to WebSocket')
    }
  }
)

export const disconnectWebSocket = createAsyncThunk(
  'websocket/disconnect',
  async (_, { rejectWithValue }) => {
    try {
      // Mock WebSocket disconnect for demo
      return true
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to disconnect WebSocket')
    }
  }
)

export const subscribeToStream = createAsyncThunk(
  'websocket/subscribe',
  async (stream: string, { rejectWithValue }) => {
    try {
      // Mock WebSocket subscribe for demo
      return stream
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to subscribe to stream')
    }
  }
)

export const unsubscribeFromStream = createAsyncThunk(
  'websocket/unsubscribe',
  async (stream: string, { rejectWithValue }) => {
    try {
      // Mock WebSocket unsubscribe for demo
      return stream
    } catch (error: any) {
      return rejectWithValue(error.message || 'Failed to unsubscribe from stream')
    }
  }
)

// State interface
interface WebSocketState {
  // Connection state
  connection: WebSocketConnection | null
  status: ConnectionStatus
  isConnecting: boolean
  isReconnecting: boolean
  
  // Message handling
  messages: WebSocketMessage[]
  lastMessage: WebSocketMessage | null
  messageQueue: WebSocketMessage[]
  
  // Subscriptions
  subscriptions: string[]
  activeStreams: string[]
  
  // Performance metrics
  latency: number
  messageCount: number
  errorCount: number
  lastHeartbeat: Date | null
  
  // Reconnection settings
  reconnectAttempts: number
  maxReconnectAttempts: number
  reconnectDelay: number
  reconnectTimer: number | null
  
  // Error handling
  lastError: string | null
  errorHistory: Array<{ timestamp: Date; error: string; code?: string }>
  
  // Message processing
  isProcessing: boolean
  processingQueue: WebSocketMessage[]
  processedMessages: Set<string>
}

// Initial state
const initialState: WebSocketState = {
  // Connection state
  connection: null,
  status: 'disconnected',
  isConnecting: false,
  isReconnecting: false,
  
  // Message handling
  messages: [],
  lastMessage: null,
  messageQueue: [],
  
  // Subscriptions
  subscriptions: [],
  activeStreams: [],
  
  // Performance metrics
  latency: 0,
  messageCount: 0,
  errorCount: 0,
  lastHeartbeat: null,
  
  // Reconnection settings
  reconnectAttempts: 0,
  maxReconnectAttempts: 5,
  reconnectDelay: 1000,
  reconnectTimer: null,
  
  // Error handling
  lastError: null,
  errorHistory: [],
  
  // Message processing
  isProcessing: false,
  processingQueue: [],
  processedMessages: new Set(),
}

// WebSocket slice
const websocketSlice = createSlice({
  name: 'websocket',
  initialState,
  reducers: {
    // Connection state actions
    setStatus: (state, action: PayloadAction<ConnectionStatus>) => {
      state.status = action.payload
      if (action.payload === 'connected') {
        state.isConnecting = false
        state.isReconnecting = false
        state.reconnectAttempts = 0
        state.lastError = null
      } else if (action.payload === 'connecting') {
        state.isConnecting = true
      } else if (action.payload === 'reconnecting') {
        state.isReconnecting = true
      }
    },
    
    setConnecting: (state, action: PayloadAction<boolean>) => {
      state.isConnecting = action.payload
    },
    
    setReconnecting: (state, action: PayloadAction<boolean>) => {
      state.isReconnecting = action.payload
    },
    
    // Message handling actions
    addMessage: (state, action: PayloadAction<WebSocketMessage>) => {
      const message = action.payload
      state.messages.push(message)
      state.lastMessage = message
      state.messageCount++
      
      // Keep only last 1000 messages to prevent memory issues
      if (state.messages.length > 1000) {
        state.messages = state.messages.slice(-1000)
      }
      
      // Add to processing queue if not already processed
      if (!state.processedMessages.has(message.id || `${new Date(message.timestamp).getTime()}-${message.type}`)) {
        state.processingQueue.push(message)
      }
    },
    
    addMessageToQueue: (state, action: PayloadAction<WebSocketMessage>) => {
      state.messageQueue.push(action.payload)
    },
    
    processMessage: (state, action: PayloadAction<string>) => {
      const messageId = action.payload
      state.processedMessages.add(messageId)
      state.processingQueue = state.processingQueue.filter(m => 
        (m.id || `${m.timestamp.getTime()}-${m.type}`) !== messageId
      )
    },
    
    clearMessages: (state) => {
      state.messages = []
      state.lastMessage = null
      state.messageQueue = []
      state.processingQueue = []
      state.processedMessages.clear()
    },
    
    // Subscription actions
    addSubscription: (state, action: PayloadAction<string>) => {
      if (!state.subscriptions.includes(action.payload)) {
        state.subscriptions.push(action.payload)
      }
    },
    
    removeSubscription: (state, action: PayloadAction<string>) => {
      state.subscriptions = state.subscriptions.filter(s => s !== action.payload)
    },
    
    setActiveStreams: (state, action: PayloadAction<string[]>) => {
      state.activeStreams = action.payload
    },
    
    // Performance actions
    updateLatency: (state, action: PayloadAction<number>) => {
      state.latency = action.payload
    },
    
    updateHeartbeat: (state) => {
      state.lastHeartbeat = new Date()
    },
    
    incrementErrorCount: (state) => {
      state.errorCount++
    },
    
    // Reconnection actions
    setReconnectAttempts: (state, action: PayloadAction<number>) => {
      state.reconnectAttempts = action.payload
    },
    
    setReconnectDelay: (state, action: PayloadAction<number>) => {
      state.reconnectDelay = action.payload
    },
    
    setReconnectTimer: (state, action: PayloadAction<number | null>) => {
      state.reconnectTimer = action.payload
    },
    
    // Error handling actions
    setError: (state, action: PayloadAction<{ error: string; code?: string }>) => {
      const { error, code } = action.payload
      state.lastError = error
      state.errorHistory.push({
        timestamp: new Date(),
        error,
        code,
      })
      
      // Keep only last 100 errors
      if (state.errorHistory.length > 100) {
        state.errorHistory = state.errorHistory.slice(-100)
      }
      
      state.errorCount++
    },
    
    clearError: (state) => {
      state.lastError = null
    },
    
    // Processing actions
    setProcessing: (state, action: PayloadAction<boolean>) => {
      state.isProcessing = action.payload
    },
    
    // Connection update actions
    updateConnection: (state, action: PayloadAction<Partial<WebSocketConnection>>) => {
      if (state.connection) {
        state.connection = { ...state.connection, ...action.payload }
      }
    },
    
    // Reset actions
    resetWebSocket: (state) => {
      state.connection = null
      state.status = 'disconnected'
      state.isConnecting = false
      state.isReconnecting = false
      state.messages = []
      state.lastMessage = null
      state.messageQueue = []
      state.subscriptions = []
      state.activeStreams = []
      state.latency = 0
      state.messageCount = 0
      state.errorCount = 0
      state.lastHeartbeat = null
      state.reconnectAttempts = 0
      state.reconnectTimer = null
      state.lastError = null
      state.errorHistory = []
      state.isProcessing = false
      state.processingQueue = []
      state.processedMessages.clear()
    },
  },
  extraReducers: (builder) => {
    // Connect WebSocket
    builder
      .addCase(connectWebSocket.pending, (state) => {
        state.isConnecting = true
        state.status = 'connecting'
        state.lastError = null
      })
      .addCase(connectWebSocket.fulfilled, (state, action) => {
        state.isConnecting = false
        state.status = 'connected'
        state.connection = action.payload
        state.reconnectAttempts = 0
        state.lastError = null
      })
      .addCase(connectWebSocket.rejected, (state, action) => {
        state.isConnecting = false
        state.status = 'error'
        state.lastError = action.payload as string
        state.errorCount++
      })

    // Disconnect WebSocket
    builder
      .addCase(disconnectWebSocket.pending, (state) => {
        state.status = 'disconnected'
      })
      .addCase(disconnectWebSocket.fulfilled, (state) => {
        state.status = 'disconnected'
        state.connection = null
        state.subscriptions = []
        state.activeStreams = []
      })
      .addCase(disconnectWebSocket.rejected, (state, action) => {
        state.lastError = action.payload as string
        state.errorCount++
      })

    // Subscribe to stream
    builder
      .addCase(subscribeToStream.fulfilled, (state, action) => {
        const stream = action.payload
        if (!state.subscriptions.includes(stream)) {
          state.subscriptions.push(stream)
        }
        if (!state.activeStreams.includes(stream)) {
          state.activeStreams.push(stream)
        }
      })
      .addCase(subscribeToStream.rejected, (state, action) => {
        state.lastError = action.payload as string
        state.errorCount++
      })

    // Unsubscribe from stream
    builder
      .addCase(unsubscribeFromStream.fulfilled, (state, action) => {
        const stream = action.payload
        state.subscriptions = state.subscriptions.filter(s => s !== stream)
        state.activeStreams = state.activeStreams.filter(s => s !== stream)
      })
      .addCase(unsubscribeFromStream.rejected, (state, action) => {
        state.lastError = action.payload as string
        state.errorCount++
      })
  },
})

// Selectors
export const selectWebSocket = (state: { websocket: WebSocketState }) => state.websocket
export const selectConnectionStatus = (state: { websocket: WebSocketState }) => state.websocket.status
export const selectIsConnected = (state: { websocket: WebSocketState }) => state.websocket.status === 'connected'
export const selectIsConnecting = (state: { websocket: WebSocketState }) => state.websocket.isConnecting
export const selectIsReconnecting = (state: { websocket: WebSocketState }) => state.websocket.isReconnecting
export const selectConnection = (state: { websocket: WebSocketState }) => state.websocket.connection
export const selectSubscriptions = (state: { websocket: WebSocketState }) => state.websocket.subscriptions
export const selectActiveStreams = (state: { websocket: WebSocketState }) => state.websocket.activeStreams
export const selectLatency = (state: { websocket: WebSocketState }) => state.websocket.latency
export const selectMessageCount = (state: { websocket: WebSocketState }) => state.websocket.messageCount
export const selectErrorCount = (state: { websocket: WebSocketState }) => state.websocket.errorCount
export const selectLastMessage = (state: { websocket: WebSocketState }) => state.websocket.lastMessage
export const selectLastError = (state: { websocket: WebSocketState }) => state.websocket.lastError
export const selectReconnectAttempts = (state: { websocket: WebSocketState }) => state.websocket.reconnectAttempts
export const selectIsProcessing = (state: { websocket: WebSocketState }) => state.websocket.isProcessing

// Actions
export const {
  setStatus,
  setConnecting,
  setReconnecting,
  addMessage,
  addMessageToQueue,
  processMessage,
  clearMessages,
  addSubscription,
  removeSubscription,
  setActiveStreams,
  updateLatency,
  updateHeartbeat,
  incrementErrorCount,
  setReconnectAttempts,
  setReconnectDelay,
  setReconnectTimer,
  setError,
  clearError,
  setProcessing,
  updateConnection,
  resetWebSocket,
} = websocketSlice.actions

export default websocketSlice.reducer
