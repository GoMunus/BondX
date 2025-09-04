import { createSlice, PayloadAction } from '@reduxjs/toolkit'

export interface Notification {
  id: string
  type: 'success' | 'error' | 'warning' | 'info'
  title: string
  message: string
  timestamp: string
  read: boolean
  persistent?: boolean
  actionLabel?: string
  actionCallback?: () => void
  autoClose?: boolean
  autoCloseDelay?: number
}

interface NotificationState {
  notifications: Notification[]
  unreadCount: number
  toasts: Notification[]
}

const initialState: NotificationState = {
  notifications: [],
  unreadCount: 0,
  toasts: [],
}

const notificationSlice = createSlice({
  name: 'notifications',
  initialState,
  reducers: {
    addNotification: (state, action: PayloadAction<Omit<Notification, 'id' | 'timestamp' | 'read'>>) => {
      const notification: Notification = {
        ...action.payload,
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        timestamp: new Date().toISOString(),
        read: false,
      }
      
      state.notifications.unshift(notification)
      state.unreadCount += 1
      
      // Add to toasts if not persistent
      if (!notification.persistent) {
        state.toasts.push(notification)
      }
    },
    
    markAsRead: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(n => n.id === action.payload)
      if (notification && !notification.read) {
        notification.read = true
        state.unreadCount = Math.max(0, state.unreadCount - 1)
      }
    },
    
    markAllAsRead: (state) => {
      state.notifications.forEach(notification => {
        notification.read = true
      })
      state.unreadCount = 0
    },
    
    removeNotification: (state, action: PayloadAction<string>) => {
      const index = state.notifications.findIndex(n => n.id === action.payload)
      if (index !== -1) {
        const notification = state.notifications[index]
        if (!notification.read) {
          state.unreadCount = Math.max(0, state.unreadCount - 1)
        }
        state.notifications.splice(index, 1)
      }
    },
    
    removeToast: (state, action: PayloadAction<string>) => {
      const index = state.toasts.findIndex(t => t.id === action.payload)
      if (index !== -1) {
        state.toasts.splice(index, 1)
      }
    },
    
    clearAllNotifications: (state) => {
      state.notifications = []
      state.unreadCount = 0
    },
    
    clearToasts: (state) => {
      state.toasts = []
    },
    
    // Convenience action creators
    addSuccessNotification: (state, action: PayloadAction<{ title: string; message: string; persistent?: boolean }>) => {
      const notification: Notification = {
        ...action.payload,
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        type: 'success',
        timestamp: new Date().toISOString(),
        read: false,
      }
      
      state.notifications.unshift(notification)
      state.unreadCount += 1
      
      if (!notification.persistent) {
        state.toasts.push(notification)
      }
    },
    
    addErrorNotification: (state, action: PayloadAction<{ title: string; message: string; persistent?: boolean }>) => {
      const notification: Notification = {
        ...action.payload,
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        type: 'error',
        timestamp: new Date().toISOString(),
        read: false,
        persistent: action.payload.persistent ?? true, // Errors are persistent by default
      }
      
      state.notifications.unshift(notification)
      state.unreadCount += 1
      
      if (!notification.persistent) {
        state.toasts.push(notification)
      }
    },
    
    addWarningNotification: (state, action: PayloadAction<{ title: string; message: string; persistent?: boolean }>) => {
      const notification: Notification = {
        ...action.payload,
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        type: 'warning',
        timestamp: new Date().toISOString(),
        read: false,
      }
      
      state.notifications.unshift(notification)
      state.unreadCount += 1
      
      if (!notification.persistent) {
        state.toasts.push(notification)
      }
    },
    
    addInfoNotification: (state, action: PayloadAction<{ title: string; message: string; persistent?: boolean }>) => {
      const notification: Notification = {
        ...action.payload,
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        type: 'info',
        timestamp: new Date().toISOString(),
        read: false,
      }
      
      state.notifications.unshift(notification)
      state.unreadCount += 1
      
      if (!notification.persistent) {
        state.toasts.push(notification)
      }
    },
  },
})

export const {
  addNotification,
  markAsRead,
  markAllAsRead,
  removeNotification,
  removeToast,
  clearAllNotifications,
  clearToasts,
  addSuccessNotification,
  addErrorNotification,
  addWarningNotification,
  addInfoNotification,
} = notificationSlice.actions

// Selectors
export const selectNotifications = (state: { notifications: NotificationState }) => state.notifications.notifications
export const selectUnreadCount = (state: { notifications: NotificationState }) => state.notifications.unreadCount
export const selectToasts = (state: { notifications: NotificationState }) => state.notifications.toasts
export const selectUnreadNotifications = (state: { notifications: NotificationState }) => 
  state.notifications.notifications.filter(n => !n.read)
export const selectRecentNotifications = (state: { notifications: NotificationState }) => 
  state.notifications.notifications.slice(0, 10)

export default notificationSlice.reducer
