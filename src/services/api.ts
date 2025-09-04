/**
 * API Service for BondX Frontend
 * 
 * Provides centralized API communication with the backend,
 * including REST endpoints and WebSocket connections.
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000'

// Request timeout in milliseconds
const REQUEST_TIMEOUT = 30000

// Create axios instance with default configuration
const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL: API_BASE_URL,
    timeout: REQUEST_TIMEOUT,
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
  })

  // Request interceptor to add auth token
  client.interceptors.request.use(
    (config) => {
      const token = localStorage.getItem('auth_token')
      if (token) {
        config.headers.Authorization = `Bearer ${token}`
      }
      return config
    },
    (error) => {
      return Promise.reject(error)
    }
  )

  // Response interceptor to handle errors
  client.interceptors.response.use(
    (response) => response,
    (error) => {
      if (error.response?.status === 401) {
        // Unauthorized - redirect to login
        localStorage.removeItem('auth_token')
        window.location.href = '/login'
      }
      return Promise.reject(error)
    }
  )

  return client
}

const apiClient = createApiClient()

// Type definitions
export interface ApiResponse<T = any> {
  success?: boolean
  data: T
  message?: string
  timestamp: string
  metadata?: Record<string, any>
}

export interface PaginatedResponse<T = any> extends ApiResponse<T[]> {
  pagination: {
    total_count: number
    limit: number
    offset: number
    has_more: boolean
  }
}

export interface PortfolioSummary {
  summary: {
    portfolio_id: string
    total_aum: number
    daily_pnl: number
    daily_pnl_percent: number
    active_positions: number
    total_bonds: number
    average_rating: string
    weighted_duration: number
    cash_balance: number
    leverage_ratio: number
    last_updated: string
  }
  risk_metrics: {
    var_95_1d: number
    portfolio_volatility: number
    modified_duration: number
    credit_spread_risk: string
  }
  performance: {
    mtd_return: number
    qtd_return: number
    ytd_return: number
    trailing_volatility: number
  }
  allocation: {
    government: number
    corporate: number
    psu: number
  }
}

export interface MarketStatus {
  timestamp: string
  local_time: string
  markets: {
    nse: MarketInfo
    bse: MarketInfo
    rbi: MarketInfo
  }
  market_indicators: {
    '10y_gsec': number
    '10y_gsec_change': number
    call_money_rate: number
    repo_rate: number
    reverse_repo_rate: number
  }
}

export interface MarketInfo {
  name: string
  status: 'OPEN' | 'CLOSED'
  next_session: string
  total_volume: number
  total_trades: number
  avg_yield: number
}

export interface RiskMetrics {
  portfolio_id: string
  timestamp: string
  var_metrics: {
    var_95_1d: number
    var_99_1d: number
    cvar_95_1d: number
  }
  duration_risk: {
    modified_duration: number
    effective_duration: number
    convexity: number
  }
  concentration_risk: {
    issuer_concentration: number
    sector_concentration: number
    rating_concentration: number
  }
  liquidity_metrics: {
    liquidity_score: number
    liquidity_risk: string
    days_to_liquidate: number
  }
  stress_scenarios: Record<string, number>
}

export interface Trade {
  trade_id: string
  timestamp: string
  bond_id: string
  bond_name: string
  side: 'BUY' | 'SELL'
  quantity: number
  price: number
  yield: number
  trade_value: number
  counterparty: string
  venue: string
}

export interface TradingActivity {
  timestamp: string
  recent_trades: Trade[]
  summary: {
    total_trades: number
    total_volume: number
    avg_trade_size: number
    buy_trades: number
    sell_trades: number
    buy_volume: number
    sell_volume: number
  }
}

export interface SystemHealth {
  timestamp: string
  overall_status: 'HEALTHY' | 'DEGRADED' | 'DOWN'
  uptime: string
  components: Record<string, {
    status: 'HEALTHY' | 'DEGRADED' | 'DOWN'
    response_time: number
    [key: string]: any
  }>
  performance: {
    cpu_usage: number
    memory_usage: number
    disk_usage: number
    network_io: string
    database_connections?: number
  }
}

export interface PlatformStats {
  timestamp: string
  aum: {
    total: number
    growth_mtd: number
    growth_ytd: number
  }
  institutions: {
    total_onboarded: number
    active_today: number
    new_this_month: number
  }
  trading: {
    trades_today: number
    volume_today: number
    avg_trade_size: number
  }
  performance: {
    uptime: string
    avg_latency: string
    orders_per_second: number
    data_points_processed: number
  }
  compliance: {
    risk_limit_breaches: number
    failed_trades: number
    compliance_score: number
  }
}

export interface YieldCurvePoint {
  tenor: number
  rate: number
  tenor_label: string
}

export interface YieldCurveData {
  currency: string
  date: string
  timestamp: string
  data: YieldCurvePoint[]
  metadata: {
    source: string
    construction_method: string
    last_updated: string
  }
}

export interface CorporateBond {
  isin: string
  descriptor: string
  issuer_name: string
  sector: string
  bond_type: string
  coupon_rate: number | null
  maturity_date: string | null
  weighted_avg_price: number
  last_trade_price: number
  weighted_avg_yield: number
  last_trade_yield: number
  value_lakhs: number
  num_trades: number
  face_value: number | null
}

export interface CorporateBondsData {
  timestamp: string
  total_bonds: number
  bonds: CorporateBond[]
  market_summary: {
    total_bonds: number
    total_value_lakhs: number
    total_trades: number
    average_yield: number
    sector_breakdown: {
      counts: Record<string, number>
      values: Record<string, number>
    }
  }
  filters_applied: {
    sector: string | null
    min_yield: number | null
    max_yield: number | null
    sort_by: string
    limit: number
  }
}

export interface BondSectorsData {
  timestamp: string
  sectors: string[]
  sector_statistics: Record<string, {
    count: number
    total_value_lakhs: number
    avg_yield: number
    total_trades: number
  }>
}

export interface TopPerformingBondsData {
  timestamp: string
  metric: string
  top_bonds: Array<{
    isin: string
    issuer_name: string
    descriptor: string
    sector: string
    last_trade_price: number
    last_trade_yield: number
    value_lakhs: number
    num_trades: number
    coupon_rate: number | null
  }>
}

// API Service Class
export class ApiService {
  private client: AxiosInstance

  constructor() {
    this.client = apiClient
  }

  // Generic API request method
  private async request<T>(config: AxiosRequestConfig): Promise<T> {
    try {
      const response: AxiosResponse<T> = await this.client.request(config)
      return response.data
    } catch (error) {
      console.error('API Request Error:', error)
      throw error
    }
  }

  // Dashboard API endpoints
  async getDashboardSummary(userId?: string, portfolioId?: string): Promise<ApiResponse> {
    return this.request({
      method: 'GET',
      url: '/api/v1/dashboard/summary',
      params: { user_id: userId, portfolio_id: portfolioId }
    })
  }

  async getPortfolioSummary(portfolioId?: string): Promise<PortfolioSummary> {
    return this.request({
      method: 'GET',
      url: '/api/v1/dashboard/portfolio/summary',
      params: { portfolio_id: portfolioId }
    })
  }

  async getMarketStatus(): Promise<MarketStatus> {
    return this.request({
      method: 'GET',
      url: '/api/v1/dashboard/market/status'
    })
  }

  async getRiskMetrics(portfolioId?: string): Promise<RiskMetrics> {
    return this.request({
      method: 'GET',
      url: '/api/v1/dashboard/risk/metrics',
      params: { portfolio_id: portfolioId }
    })
  }

  async getTradingActivity(limit: number = 20): Promise<TradingActivity> {
    return this.request({
      method: 'GET',
      url: '/api/v1/dashboard/trading/activity',
      params: { limit }
    })
  }

  async getSystemHealth(): Promise<SystemHealth> {
    return this.request({
      method: 'GET',
      url: '/api/v1/dashboard/system/health'
    })
  }

  async getPlatformStats(): Promise<PlatformStats> {
    return this.request({
      method: 'GET',
      url: '/api/v1/dashboard/platform/stats'
    })
  }

  async getYieldCurve(currency: string = 'INR', date?: string): Promise<YieldCurveData> {
    return this.request({
      method: 'GET',
      url: '/api/v1/dashboard/yield-curve',
      params: { currency, date }
    })
  }

  // Portfolio Analytics endpoints
  async getPortfolioMetrics(portfolioId: string): Promise<ApiResponse> {
    return this.request({
      method: 'GET',
      url: `/api/v1/portfolio-analytics/metrics`,
      params: { portfolio_id: portfolioId }
    })
  }

  // Risk Management endpoints
  async getPortfolioRisk(portfolioId: string): Promise<ApiResponse> {
    return this.request({
      method: 'GET',
      url: `/api/v1/risk/portfolios/${portfolioId}/risk`
    })
  }

  async performStressTest(portfolioId: string, scenarioType: string): Promise<ApiResponse> {
    return this.request({
      method: 'POST',
      url: `/api/v1/risk/portfolios/${portfolioId}/stress-test/default`,
      data: { scenario_type: scenarioType }
    })
  }

  // Trading endpoints
  async getOrders(filters?: Record<string, any>): Promise<PaginatedResponse> {
    return this.request({
      method: 'GET',
      url: '/api/v1/trading/orders',
      params: filters
    })
  }

  async getTrades(filters?: Record<string, any>): Promise<PaginatedResponse> {
    return this.request({
      method: 'GET',
      url: '/api/v1/trading/trades',
      params: filters
    })
  }

  async getMarketData(bondId: string): Promise<ApiResponse> {
    return this.request({
      method: 'GET',
      url: `/api/v1/trading/market-data/${bondId}`
    })
  }

  // Authentication endpoints
  async login(credentials: { username: string; password: string }): Promise<ApiResponse> {
    return this.request({
      method: 'POST',
      url: '/api/v1/auth/login',
      data: credentials
    })
  }

  async logout(): Promise<void> {
    localStorage.removeItem('auth_token')
    return this.request({
      method: 'POST',
      url: '/api/v1/auth/logout'
    })
  }

  async getUserProfile(): Promise<ApiResponse> {
    return this.request({
      method: 'GET',
      url: '/api/v1/auth/profile'
    })
  }

  // Health check
  async healthCheck(): Promise<ApiResponse> {
    return this.request({
      method: 'GET',
      url: '/api/v1/health'
    })
  }

  // Corporate Bonds API methods
  async getCorporateBonds(params?: {
    sector?: string
    min_yield?: number
    max_yield?: number
    sort_by?: string
    limit?: number
  }): Promise<CorporateBondsData> {
    const queryParams = new URLSearchParams()
    if (params?.sector) queryParams.append('sector', params.sector)
    if (params?.min_yield !== undefined) queryParams.append('min_yield', params.min_yield.toString())
    if (params?.max_yield !== undefined) queryParams.append('max_yield', params.max_yield.toString())
    if (params?.sort_by) queryParams.append('sort_by', params.sort_by)
    if (params?.limit) queryParams.append('limit', params.limit.toString())

    const response = await this.request({
      method: 'GET',
      url: `/api/v1/dashboard/corporate-bonds?${queryParams}`
    })
    return (response as AxiosResponse<CorporateBondsData>).data
  }

  async getBondSectors(): Promise<BondSectorsData> {
    const response = await this.request({
      method: 'GET',
      url: '/api/v1/dashboard/bonds/sectors'
    })
    return response.data as BondSectorsData
  }

  async getTopPerformingBonds(metric: string = 'volume', limit: number = 10): Promise<TopPerformingBondsData> {
    const response = await this.request({
      method: 'GET',
      url: `/api/v1/dashboard/bonds/top-performers?metric=${metric}&limit=${limit}`
    })
    return response.data as TopPerformingBondsData
  }

  // Fractional Bonds API methods
  async getAvailableBonds(params?: {
    sector?: string
    min_yield?: number
    max_yield?: number
    limit?: number
  }): Promise<any[]> {
    return this.request({
      method: 'GET',
      url: '/api/v1/bonds/',
      params
    })
  }

  async getFractionalPositions(): Promise<any[]> {
    return this.request({
      method: 'GET',
      url: '/api/v1/bonds/positions'
    })
  }

  async buyBondFraction(data: {
    bond_id: string
    fraction_amount: number
    max_price?: number
  }): Promise<any> {
    return this.request({
      method: 'POST',
      url: '/api/v1/bonds/buy-fraction',
      data
    })
  }

  async sellBondFraction(data: {
    bond_id: string
    fraction_amount: number
    min_price?: number
  }): Promise<any> {
    return this.request({
      method: 'POST',
      url: '/api/v1/bonds/sell-fraction',
      data
    })
  }

  // Auction API methods
  async getActiveAuctions(): Promise<any[]> {
    return this.request({
      method: 'GET',
      url: '/api/v1/simple-auctions/'
    })
  }

  async placeBid(data: {
    auction_id: string
    bid_amount: number
    fraction_amount: number
  }): Promise<any> {
    return this.request({
      method: 'POST',
      url: '/api/v1/simple-auctions/place-bid',
      data
    })
  }

  async getUserBids(): Promise<any[]> {
    return this.request({
      method: 'GET',
      url: '/api/v1/simple-auctions/user/bids'
    })
  }
}

// WebSocket Service Class
export class WebSocketService {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private subscriptions: Set<string> = new Set()
  private messageHandlers: Map<string, Function[]> = new Map()

  constructor(private clientId: string) {}

  connect(userId?: string, portfolioId?: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        const params = new URLSearchParams({
          client_id: this.clientId,
          ...(userId && { user_id: userId }),
          ...(portfolioId && { portfolio_id: portfolioId })
        })

        const wsUrl = `${WS_BASE_URL}/api/v1/ws/dashboard/connect?${params}`
        this.ws = new WebSocket(wsUrl)

        this.ws.onopen = () => {
          console.log('Dashboard WebSocket connected')
          this.reconnectAttempts = 0
          resolve()
        }

        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data)
            this.handleMessage(message)
          } catch (error) {
            console.error('Error parsing WebSocket message:', error)
          }
        }

        this.ws.onclose = () => {
          console.log('Dashboard WebSocket disconnected')
          this.attemptReconnect(userId, portfolioId)
        }

        this.ws.onerror = (error) => {
          console.error('Dashboard WebSocket error:', error)
          reject(error)
        }
      } catch (error) {
        reject(error)
      }
    })
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.subscriptions.clear()
    this.messageHandlers.clear()
  }

  subscribe(subscriptionTypes: string[]): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        subscriptions: subscriptionTypes
      }))
      
      subscriptionTypes.forEach(type => this.subscriptions.add(type))
    }
  }

  unsubscribe(subscriptionTypes: string[]): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'unsubscribe',
        subscriptions: subscriptionTypes
      }))
      
      subscriptionTypes.forEach(type => this.subscriptions.delete(type))
    }
  }

  onMessage(messageType: string, handler: Function): void {
    if (!this.messageHandlers.has(messageType)) {
      this.messageHandlers.set(messageType, [])
    }
    this.messageHandlers.get(messageType)!.push(handler)
  }

  private handleMessage(message: any): void {
    const handlers = this.messageHandlers.get(message.type) || []
    handlers.forEach(handler => {
      try {
        handler(message)
      } catch (error) {
        console.error('Error in message handler:', error)
      }
    })
  }

  private attemptReconnect(userId?: string, portfolioId?: string): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
      
      console.log(`Attempting to reconnect WebSocket in ${delay}ms (attempt ${this.reconnectAttempts})`)
      
      setTimeout(() => {
        this.connect(userId, portfolioId)
          .then(() => {
            // Re-subscribe to previous subscriptions
            if (this.subscriptions.size > 0) {
              this.subscribe(Array.from(this.subscriptions))
            }
          })
          .catch(error => {
            console.error('Reconnection failed:', error)
          })
      }, delay)
    } else {
      console.error('Max reconnection attempts reached')
    }
  }

  ping(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'ping' }))
    }
  }

  getSnapshot(dataType: string): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'get_snapshot',
        data_type: dataType
      }))
    }
  }
}

// Export singleton instances
export const apiService = new ApiService()

// WebSocket factory function
export const createWebSocketService = (clientId: string): WebSocketService => {
  return new WebSocketService(clientId)
}

// Utility functions
export const formatCurrency = (amount: number, currency: string = 'INR'): string => {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: currency,
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  }).format(amount)
}

export const formatPercentage = (value: number, decimals: number = 2): string => {
  return `${value.toFixed(decimals)}%`
}

export const formatNumber = (value: number, decimals: number = 2): string => {
  return new Intl.NumberFormat('en-IN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  }).format(value)
}

export default apiService
