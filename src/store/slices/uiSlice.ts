import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'
import { DashboardLayout, Widget, WidgetType, WidgetPosition, WidgetSize } from '@/types'
import { uiService } from '@/services/uiService'

// Async thunks
export const fetchDashboardLayouts = createAsyncThunk(
  'ui/fetchDashboardLayouts',
  async (_, { rejectWithValue }) => {
    try {
      const response = await uiService.getDashboardLayouts()
      return response.data
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch dashboard layouts')
    }
  }
)

export const saveDashboardLayout = createAsyncThunk(
  'ui/saveDashboardLayout',
  async (layout: DashboardLayout, { rejectWithValue }) => {
    try {
      const response = await uiService.saveDashboardLayout(layout)
      return response.data
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to save dashboard layout')
    }
  }
)

export const deleteDashboardLayout = createAsyncThunk(
  'ui/deleteDashboardLayout',
  async (layoutId: string, { rejectWithValue }) => {
    try {
      await uiService.deleteDashboardLayout(layoutId)
      return layoutId
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to delete dashboard layout')
    }
  }
)

// State interface
interface UIState {
  // Dashboard management
  currentLayout: DashboardLayout | null
  availableLayouts: DashboardLayout[]
  layoutsLoading: boolean
  layoutsError: string | null
  
  // Widget management
  widgets: Widget[]
  selectedWidget: Widget | null
  widgetSettings: Record<string, any>
  
  // Layout state
  isDragging: boolean
  isResizing: boolean
  gridBreakpoint: 'lg' | 'md' | 'sm' | 'xs'
  
  // UI state
  sidebarCollapsed: boolean
  theme: 'light' | 'dark' | 'auto'
  sidebarWidth: number
  headerHeight: number
  
  // Modal and overlay state
  activeModals: string[]
  activeOverlays: string[]
  
  // Search and navigation
  globalSearchQuery: string
  searchResults: any[]
  searchLoading: boolean
  
  // Notifications
  toastNotifications: any[]
  systemAlerts: any[]
  
  // Performance metrics
  renderTime: number
  lastUpdate: Date | null
}

// Initial state
const initialState: UIState = {
  // Dashboard management
  currentLayout: null,
  availableLayouts: [],
  layoutsLoading: false,
  layoutsError: null,
  
  // Widget management
  widgets: [
    {
      id: 'portfolio_summary',
      type: 'portfolio-summary',
      title: 'Portfolio Summary',
      position: { x: 0, y: 0, w: 6, h: 3 },
      size: { width: 6, height: 3 },
      config: {
        title: 'Portfolio Summary',
        refreshInterval: 30,
        showHeader: true,
        showFooter: false
      }
    },
    {
      id: 'market_overview',
      type: 'market-overview',
      title: 'Market Overview',
      position: { x: 6, y: 0, w: 6, h: 3 },
      size: { width: 6, height: 3 },
      config: {
        title: 'Market Overview',
        refreshInterval: 30,
        showHeader: true,
        showFooter: false
      }
    },
    {
      id: 'risk_metrics',
      type: 'risk-metrics',
      title: 'Risk Metrics',
      position: { x: 0, y: 3, w: 6, h: 3 },
      size: { width: 6, height: 3 },
      config: {
        title: 'Risk Metrics',
        refreshInterval: 60,
        showHeader: true,
        showFooter: false
      }
    },
    {
      id: 'trading_activity',
      type: 'trading_activity',
      title: 'Trading Activity',
      position: { x: 6, y: 3 },
      size: { width: 6, height: 3 },
      config: {
        title: 'Trading Activity',
        refreshInterval: 10,
        showHeader: true,
        showFooter: false
      }
    },
    {
      id: 'fractional_ownership',
      type: 'fractional_ownership',
      title: 'Fractional Bond Ownership',
      position: { x: 0, y: 6 },
      size: { width: 6, height: 4 },
      config: {
        title: 'Fractional Bond Ownership',
        refreshInterval: 30,
        showHeader: true,
        showFooter: false
      }
    },
    {
      id: 'auction',
      type: 'auction',
      title: 'Live Bond Auctions',
      position: { x: 6, y: 6 },
      size: { width: 6, height: 4 },
      config: {
        title: 'Live Bond Auctions',
        refreshInterval: 10,
        showHeader: true,
        showFooter: false
      }
    },
    {
      id: 'yield_curve',
      type: 'yield_curve',
      title: 'Yield Curve',
      position: { x: 0, y: 10 },
      size: { width: 6, height: 3 },
      config: {
        title: 'Yield Curve',
        refreshInterval: 60,
        showHeader: true,
        showFooter: false
      }
    },
    {
      id: 'corporate_bonds',
      type: 'corporate_bonds',
      title: 'Corporate Bonds',
      position: { x: 6, y: 10 },
      size: { width: 6, height: 3 },
      config: {
        title: 'Corporate Bonds',
        refreshInterval: 30,
        showHeader: true,
        showFooter: false
      }
    }
  ],
  selectedWidget: null,
  widgetSettings: {},
  
  // Layout state
  isDragging: false,
  isResizing: false,
  gridBreakpoint: 'lg',
  
  // UI state
  sidebarCollapsed: false,
  theme: 'dark',
  sidebarWidth: 280,
  headerHeight: 64,
  
  // Modal and overlay state
  activeModals: [],
  activeOverlays: [],
  
  // Search and navigation
  globalSearchQuery: '',
  searchResults: [],
  searchLoading: false,
  
  // Notifications
  toastNotifications: [],
  systemAlerts: [],
  
  // Performance metrics
  renderTime: 0,
  lastUpdate: null,
}

// UI slice
const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    // Dashboard layout actions
    setCurrentLayout: (state, action: PayloadAction<DashboardLayout>) => {
      state.currentLayout = action.payload
      state.widgets = action.payload.widgets
    },
    
    updateWidgetPosition: (state, action: PayloadAction<{ widgetId: string; position: WidgetPosition }>) => {
      const { widgetId, position } = action.payload
      const widget = state.widgets.find(w => w.id === widgetId)
      if (widget) {
        widget.position = position
      }
    },
    
    updateWidgetSize: (state, action: PayloadAction<{ widgetId: string; size: WidgetSize }>) => {
      const { widgetId, size } = action.payload
      const widget = state.widgets.find(w => w.id === widgetId)
      if (widget) {
        widget.size = size
      }
    },
    
    addWidget: (state, action: PayloadAction<Widget>) => {
      state.widgets.push(action.payload)
      if (state.currentLayout) {
        state.currentLayout.widgets = state.widgets
      }
    },
    
    removeWidget: (state, action: PayloadAction<string>) => {
      state.widgets = state.widgets.filter(w => w.id !== action.payload)
      if (state.currentLayout) {
        state.currentLayout.widgets = state.widgets
      }
    },
    
    updateWidgetConfig: (state, action: PayloadAction<{ widgetId: string; config: Partial<Widget['config']> }>) => {
      const { widgetId, config } = action.payload
      const widget = state.widgets.find(w => w.id === widgetId)
      if (widget) {
        widget.config = { ...widget.config, ...config }
      }
    },
    
    selectWidget: (state, action: PayloadAction<Widget | null>) => {
      state.selectedWidget = action.payload
    },
    
    setWidgetSettings: (state, action: PayloadAction<{ widgetId: string; settings: any }>) => {
      const { widgetId, settings } = action.payload
      state.widgetSettings[widgetId] = { ...state.widgetSettings[widgetId], ...settings }
    },
    
    // Layout state actions
    setDragging: (state, action: PayloadAction<boolean>) => {
      state.isDragging = action.payload
    },
    
    setResizing: (state, action: PayloadAction<boolean>) => {
      state.isResizing = action.payload
    },
    
    setGridBreakpoint: (state, action: PayloadAction<'lg' | 'md' | 'sm' | 'xs'>) => {
      state.gridBreakpoint = action.payload
    },
    
    // UI state actions
    toggleSidebar: (state) => {
      state.sidebarCollapsed = !state.sidebarCollapsed
    },
    
    setSidebarCollapsed: (state, action: PayloadAction<boolean>) => {
      state.sidebarCollapsed = action.payload
    },
    
    setTheme: (state, action: PayloadAction<'light' | 'dark' | 'auto'>) => {
      state.theme = action.payload
    },
    
    setSidebarWidth: (state, action: PayloadAction<number>) => {
      state.sidebarWidth = action.payload
    },
    
    setHeaderHeight: (state, action: PayloadAction<number>) => {
      state.headerHeight = action.payload
    },
    
    // Modal and overlay actions
    openModal: (state, action: PayloadAction<string>) => {
      if (!state.activeModals.includes(action.payload)) {
        state.activeModals.push(action.payload)
      }
    },
    
    closeModal: (state, action: PayloadAction<string>) => {
      state.activeModals = state.activeModals.filter(id => id !== action.payload)
    },
    
    openOverlay: (state, action: PayloadAction<string>) => {
      if (!state.activeOverlays.includes(action.payload)) {
        state.activeOverlays.push(action.payload)
      }
    },
    
    closeOverlay: (state, action: PayloadAction<string>) => {
      state.activeOverlays = state.activeOverlays.filter(id => id !== action.payload)
    },
    
    // Search actions
    setGlobalSearchQuery: (state, action: PayloadAction<string>) => {
      state.globalSearchQuery = action.payload
    },
    
    setSearchResults: (state, action: PayloadAction<any[]>) => {
      state.searchResults = action.payload
    },
    
    setSearchLoading: (state, action: PayloadAction<boolean>) => {
      state.searchLoading = action.payload
    },
    
    // Notification actions
    addToastNotification: (state, action: PayloadAction<any>) => {
      state.toastNotifications.push(action.payload)
    },
    
    removeToastNotification: (state, action: PayloadAction<string>) => {
      state.toastNotifications = state.toastNotifications.filter(n => n.id !== action.payload)
    },
    
    addSystemAlert: (state, action: PayloadAction<any>) => {
      state.systemAlerts.push(action.payload)
    },
    
    removeSystemAlert: (state, action: PayloadAction<string>) => {
      state.systemAlerts = state.systemAlerts.filter(a => a.id !== action.payload)
    },
    
    // Performance actions
    setRenderTime: (state, action: PayloadAction<number>) => {
      state.renderTime = action.payload
      state.lastUpdate = new Date()
    },
    
    // Reset actions
    resetUI: (state) => {
      state.widgets = []
      state.selectedWidget = null
      state.widgetSettings = {}
      state.isDragging = false
      state.isResizing = false
      state.activeModals = []
      state.activeOverlays = []
      state.globalSearchQuery = ''
      state.searchResults = []
      state.toastNotifications = []
      state.systemAlerts = []
    },
  },
  extraReducers: (builder) => {
    // Fetch dashboard layouts
    builder
      .addCase(fetchDashboardLayouts.pending, (state) => {
        state.layoutsLoading = true
        state.layoutsError = null
      })
      .addCase(fetchDashboardLayouts.fulfilled, (state, action) => {
        state.layoutsLoading = false
        state.availableLayouts = action.payload
        if (!state.currentLayout && action.payload.length > 0) {
          const defaultLayout = action.payload.find(l => l.isDefault)
          state.currentLayout = defaultLayout || action.payload[0]
          state.widgets = state.currentLayout.widgets
        }
      })
      .addCase(fetchDashboardLayouts.rejected, (state, action) => {
        state.layoutsLoading = false
        state.layoutsError = action.payload as string
      })

    // Save dashboard layout
    builder
      .addCase(saveDashboardLayout.fulfilled, (state, action) => {
        const savedLayout = action.payload
        const existingIndex = state.availableLayouts.findIndex(l => l.id === savedLayout.id)
        if (existingIndex >= 0) {
          state.availableLayouts[existingIndex] = savedLayout
        } else {
          state.availableLayouts.push(savedLayout)
        }
        if (state.currentLayout?.id === savedLayout.id) {
          state.currentLayout = savedLayout
        }
      })

    // Delete dashboard layout
    builder
      .addCase(deleteDashboardLayout.fulfilled, (state, action) => {
        const deletedLayoutId = action.payload
        state.availableLayouts = state.availableLayouts.filter(l => l.id !== deletedLayoutId)
        if (state.currentLayout?.id === deletedLayoutId) {
          state.currentLayout = state.availableLayouts[0] || null
          state.widgets = state.currentLayout?.widgets || []
        }
      })
  },
})

// Selectors
export const selectUI = (state: { ui: UIState }) => state.ui
export const selectCurrentLayout = (state: { ui: UIState }) => state.ui.currentLayout
export const selectAvailableLayouts = (state: { ui: UIState }) => state.ui.availableLayouts
export const selectWidgets = (state: { ui: UIState }) => state.ui.widgets
export const selectSelectedWidget = (state: { ui: UIState }) => state.ui.selectedWidget
export const selectSidebarCollapsed = (state: { ui: UIState }) => state.ui.sidebarCollapsed
export const selectTheme = (state: { ui: UIState }) => state.ui.theme
export const selectActiveModals = (state: { ui: UIState }) => state.ui.activeModals
export const selectGlobalSearchQuery = (state: { ui: UIState }) => state.ui.globalSearchQuery
export const selectSearchResults = (state: { ui: UIState }) => state.ui.searchResults
export const selectSearchLoading = (state: { ui: UIState }) => state.ui.searchLoading
export const selectToastNotifications = (state: { ui: UIState }) => state.ui.toastNotifications
export const selectSystemAlerts = (state: { ui: UIState }) => state.ui.systemAlerts

// Actions
export const {
  setCurrentLayout,
  updateWidgetPosition,
  updateWidgetSize,
  addWidget,
  removeWidget,
  updateWidgetConfig,
  selectWidget,
  setWidgetSettings,
  setDragging,
  setResizing,
  setGridBreakpoint,
  toggleSidebar,
  setSidebarCollapsed,
  setTheme,
  setSidebarWidth,
  setHeaderHeight,
  openModal,
  closeModal,
  openOverlay,
  closeOverlay,
  setGlobalSearchQuery,
  setSearchResults,
  setSearchLoading,
  addToastNotification,
  removeToastNotification,
  addSystemAlert,
  removeSystemAlert,
  setRenderTime,
  resetUI,
} = uiSlice.actions

export default uiSlice.reducer
