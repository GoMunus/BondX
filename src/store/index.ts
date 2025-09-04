import { configureStore } from '@reduxjs/toolkit'
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux'
import authReducer from './slices/authSlice'
import marketReducer from './slices/marketSlice'
import portfolioReducer from './slices/portfolioSlice'
import tradingReducer from './slices/tradingSlice'
import riskReducer from './slices/riskSlice'
import uiReducer from './slices/uiSlice'
import websocketReducer from './slices/websocketSlice'
import notificationReducer from './slices/notificationSlice'

export const store = configureStore({
  reducer: {
    auth: authReducer,
    market: marketReducer,
    portfolio: portfolioReducer,
    trading: tradingReducer,
    risk: riskReducer,
    ui: uiReducer,
    websocket: websocketReducer,
    notifications: notificationReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: [
          'persist/PERSIST',
          'persist/REHYDRATE',
          'websocket/connectionEstablished',
          'websocket/messageReceived',
        ],
        ignoredPaths: ['websocket.connection', 'websocket.lastMessage'],
      },
    }),
  devTools: process.env.NODE_ENV !== 'production',
})

export type RootState = ReturnType<typeof store.getState>
export type AppDispatch = typeof store.dispatch

// Use throughout your app instead of plain `useDispatch` and `useSelector`
export const useAppDispatch: () => AppDispatch = useDispatch
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector
