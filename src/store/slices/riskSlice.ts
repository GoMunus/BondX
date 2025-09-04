import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'

export interface RiskMetrics {
  portfolioVar95: number
  portfolioVar99: number
  expectedShortfall: number
  beta: number
  sharpeRatio: number
  informationRatio: number
  trackingError: number
  maxDrawdown: number
  volatility: number
  correlationMatrix: { [key: string]: { [key: string]: number } }
  durationRisk: number
  creditRisk: number
  liquidityRisk: number
  concentrationRisk: number
  stressTestResults: {
    scenario: string
    pnlImpact: number
    description: string
  }[]
}

export interface RiskLimits {
  maxVar95: number
  maxVar99: number
  maxConcentration: number
  maxDuration: number
  maxCreditExposure: { [rating: string]: number }
  maxSectorExposure: { [sector: string]: number }
  maxSingleIssuer: number
  maxLeverageRatio: number
}

export interface RiskAlert {
  id: string
  type: 'WARNING' | 'CRITICAL'
  category: 'VAR' | 'CONCENTRATION' | 'DURATION' | 'CREDIT' | 'LIQUIDITY'
  title: string
  description: string
  currentValue: number
  limitValue: number
  threshold: number
  timestamp: string
  acknowledged: boolean
  resolved: boolean
}

export interface StressTestScenario {
  id: string
  name: string
  description: string
  shocks: {
    yieldCurveShift: number
    creditSpreadWidening: number
    liquidityPremium: number
    volatilityIncrease: number
  }
  results?: {
    portfolioPnL: number
    varChange: number
    positionImpacts: Array<{
      isin: string
      pnlImpact: number
      priceImpact: number
    }>
  }
}

interface RiskState {
  metrics: RiskMetrics | null
  limits: RiskLimits | null
  alerts: RiskAlert[]
  scenarios: StressTestScenario[]
  selectedScenario: StressTestScenario | null
  historical: {
    dates: string[]
    var95: number[]
    var99: number[]
    sharpeRatio: number[]
    volatility: number[]
  }
  loading: boolean
  error: string | null
  lastUpdate: string | null
}

const initialState: RiskState = {
  metrics: null,
  limits: null,
  alerts: [],
  scenarios: [],
  selectedScenario: null,
  historical: {
    dates: [],
    var95: [],
    var99: [],
    sharpeRatio: [],
    volatility: [],
  },
  loading: false,
  error: null,
  lastUpdate: null,
}

// Async thunks
export const fetchRiskMetrics = createAsyncThunk(
  'risk/fetchRiskMetrics',
  async (_, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/v1/risk/metrics')
      if (!response.ok) {
        throw new Error('Failed to fetch risk metrics')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const fetchRiskLimits = createAsyncThunk(
  'risk/fetchRiskLimits',
  async (_, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/v1/risk/limits')
      if (!response.ok) {
        throw new Error('Failed to fetch risk limits')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const fetchRiskAlerts = createAsyncThunk(
  'risk/fetchRiskAlerts',
  async (_, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/v1/risk/alerts')
      if (!response.ok) {
        throw new Error('Failed to fetch risk alerts')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const fetchStressTestScenarios = createAsyncThunk(
  'risk/fetchStressTestScenarios',
  async (_, { rejectWithValue }) => {
    try {
      const response = await fetch('/api/v1/risk/stress-scenarios')
      if (!response.ok) {
        throw new Error('Failed to fetch stress test scenarios')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const runStressTest = createAsyncThunk(
  'risk/runStressTest',
  async (scenarioId: string, { rejectWithValue }) => {
    try {
      const response = await fetch(`/api/v1/risk/stress-test/${scenarioId}`, {
        method: 'POST',
      })
      if (!response.ok) {
        throw new Error('Failed to run stress test')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const fetchHistoricalRisk = createAsyncThunk(
  'risk/fetchHistoricalRisk',
  async (params: { days?: number } = {}, { rejectWithValue }) => {
    try {
      const searchParams = new URLSearchParams()
      if (params.days) searchParams.append('days', params.days.toString())
      
      const response = await fetch(`/api/v1/risk/historical?${searchParams}`)
      if (!response.ok) {
        throw new Error('Failed to fetch historical risk data')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

export const acknowledgeAlert = createAsyncThunk(
  'risk/acknowledgeAlert',
  async (alertId: string, { rejectWithValue }) => {
    try {
      const response = await fetch(`/api/v1/risk/alerts/${alertId}/acknowledge`, {
        method: 'POST',
      })
      if (!response.ok) {
        throw new Error('Failed to acknowledge alert')
      }
      return await response.json()
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Unknown error')
    }
  }
)

const riskSlice = createSlice({
  name: 'risk',
  initialState,
  reducers: {
    setSelectedScenario: (state, action: PayloadAction<StressTestScenario | null>) => {
      state.selectedScenario = action.payload
    },
    addAlert: (state, action: PayloadAction<RiskAlert>) => {
      state.alerts.unshift(action.payload)
    },
    updateAlert: (state, action: PayloadAction<{ id: string; updates: Partial<RiskAlert> }>) => {
      const alert = state.alerts.find(a => a.id === action.payload.id)
      if (alert) {
        Object.assign(alert, action.payload.updates)
      }
    },
    clearError: (state) => {
      state.error = null
    },
  },
  extraReducers: (builder) => {
    // Fetch risk metrics
    builder
      .addCase(fetchRiskMetrics.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchRiskMetrics.fulfilled, (state, action) => {
        state.loading = false
        state.metrics = action.payload
        state.lastUpdate = new Date().toISOString()
      })
      .addCase(fetchRiskMetrics.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Fetch risk limits
    builder
      .addCase(fetchRiskLimits.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchRiskLimits.fulfilled, (state, action) => {
        state.loading = false
        state.limits = action.payload
      })
      .addCase(fetchRiskLimits.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Fetch risk alerts
    builder
      .addCase(fetchRiskAlerts.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchRiskAlerts.fulfilled, (state, action) => {
        state.loading = false
        state.alerts = action.payload
      })
      .addCase(fetchRiskAlerts.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Fetch stress test scenarios
    builder
      .addCase(fetchStressTestScenarios.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchStressTestScenarios.fulfilled, (state, action) => {
        state.loading = false
        state.scenarios = action.payload
      })
      .addCase(fetchStressTestScenarios.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Run stress test
    builder
      .addCase(runStressTest.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(runStressTest.fulfilled, (state, action) => {
        state.loading = false
        const scenario = state.scenarios.find(s => s.id === action.payload.scenarioId)
        if (scenario) {
          scenario.results = action.payload.results
        }
      })
      .addCase(runStressTest.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Fetch historical risk
    builder
      .addCase(fetchHistoricalRisk.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(fetchHistoricalRisk.fulfilled, (state, action) => {
        state.loading = false
        state.historical = action.payload
      })
      .addCase(fetchHistoricalRisk.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })

    // Acknowledge alert
    builder
      .addCase(acknowledgeAlert.pending, (state) => {
        state.loading = true
        state.error = null
      })
      .addCase(acknowledgeAlert.fulfilled, (state, action) => {
        state.loading = false
        const alert = state.alerts.find(a => a.id === action.payload.id)
        if (alert) {
          alert.acknowledged = true
        }
      })
      .addCase(acknowledgeAlert.rejected, (state, action) => {
        state.loading = false
        state.error = action.payload as string
      })
  },
})

export const {
  setSelectedScenario,
  addAlert,
  updateAlert,
  clearError,
} = riskSlice.actions

// Selectors
export const selectRiskMetrics = (state: { risk: RiskState }) => state.risk.metrics
export const selectRiskLimits = (state: { risk: RiskState }) => state.risk.limits
export const selectRiskAlerts = (state: { risk: RiskState }) => state.risk.alerts
export const selectStressTestScenarios = (state: { risk: RiskState }) => state.risk.scenarios
export const selectSelectedScenario = (state: { risk: RiskState }) => state.risk.selectedScenario
export const selectHistoricalRisk = (state: { risk: RiskState }) => state.risk.historical
export const selectRiskLoading = (state: { risk: RiskState }) => state.risk.loading
export const selectRiskError = (state: { risk: RiskState }) => state.risk.error
export const selectUnacknowledgedAlerts = (state: { risk: RiskState }) => 
  state.risk.alerts.filter(alert => !alert.acknowledged && !alert.resolved)

export default riskSlice.reducer
