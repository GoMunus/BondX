import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit'
import { User, UserRole, Permission, UserPreferences } from '@/types'
import { authService } from '@/services/authService'

// Async thunks
export const login = createAsyncThunk(
  'auth/login',
  async (credentials: { email: string; password: string; mfaCode?: string }, { rejectWithValue }) => {
    try {
      // For demo purposes, simulate login
      if (credentials.email === 'demo@bondx.com' && credentials.password === 'demo123') {
        return {
          user: {
            id: '1',
            email: 'demo@bondx.com',
            firstName: 'Demo',
            lastName: 'User',
            role: 'trader',
            organization: 'BondX Demo',
            avatar: null,
          },
          token: 'demo-jwt-token',
          refreshToken: 'demo-refresh-token'
        }
      }
      throw new Error('Invalid credentials')
    } catch (error: any) {
      return rejectWithValue(error.message || 'Login failed')
    }
  }
)

export const register = createAsyncThunk(
  'auth/register',
  async (userData: { 
    firstName: string; 
    lastName: string; 
    email: string; 
    password: string; 
    organization: string;
    phone: string;
  }, { rejectWithValue }) => {
    try {
      // For demo purposes, simulate registration
      await new Promise(resolve => setTimeout(resolve, 1000))
      return {
        user: {
          id: '2',
          email: userData.email,
          firstName: userData.firstName,
          lastName: userData.lastName,
          role: 'trader',
          organization: userData.organization,
          avatar: null,
        },
        token: 'demo-jwt-token',
        refreshToken: 'demo-refresh-token'
      }
    } catch (error: any) {
      return rejectWithValue(error.message || 'Registration failed')
    }
  }
)

export const loginUser = createAsyncThunk(
  'auth/loginUser',
  async (credentials: { email: string; password: string; mfaCode?: string }, { rejectWithValue }) => {
    try {
      const response = await authService.login(credentials)
      return response.data
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Login failed')
    }
  }
)

export const refreshToken = createAsyncThunk(
  'auth/refreshToken',
  async (_, { rejectWithValue }) => {
    try {
      const response = await authService.refreshToken()
      return response.data
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Token refresh failed')
    }
  }
)

export const logoutUser = createAsyncThunk(
  'auth/logoutUser',
  async (_, { rejectWithValue }) => {
    try {
      await authService.logout()
      return true
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Logout failed')
    }
  }
)

export const fetchUserProfile = createAsyncThunk(
  'auth/fetchUserProfile',
  async (_, { rejectWithValue }) => {
    try {
      const response = await authService.getProfile()
      return response.data
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to fetch profile')
    }
  }
)

export const updateUserPreferences = createAsyncThunk(
  'auth/updateUserPreferences',
  async (preferences: Partial<UserPreferences>, { rejectWithValue }) => {
    try {
      const response = await authService.updatePreferences(preferences)
      return response.data
    } catch (error: any) {
      return rejectWithValue(error.response?.data?.message || 'Failed to update preferences')
    }
  }
)

// State interface
interface AuthState {
  user: User | null
  token: string | null
  refreshToken: string | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  mfaRequired: boolean
  sessionExpiry: Date | null
  permissions: Permission[]
  lastActivity: Date | null
}

// Initial state
const initialState: AuthState = {
  user: null,
  token: localStorage.getItem('bondx_token'),
  refreshToken: localStorage.getItem('bondx_refresh_token'),
  isAuthenticated: false,
  isLoading: false,
  error: null,
  mfaRequired: false,
  sessionExpiry: null,
  permissions: [],
  lastActivity: null,
}

// Auth slice
const authSlice = createSlice({
  name: 'auth',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null
    },
    setMfaRequired: (state, action: PayloadAction<boolean>) => {
      state.mfaRequired = action.payload
    },
    updateLastActivity: (state) => {
      state.lastActivity = new Date()
    },
    setSessionExpiry: (state, action: PayloadAction<Date>) => {
      state.sessionExpiry = action.payload
    },
    clearAuth: (state) => {
      state.user = null
      state.token = null
      state.refreshToken = null
      state.isAuthenticated = false
      state.permissions = []
      state.sessionExpiry = null
      state.lastActivity = null
      localStorage.removeItem('bondx_token')
      localStorage.removeItem('bondx_refresh_token')
    },
    updateUser: (state, action: PayloadAction<Partial<User>>) => {
      if (state.user) {
        state.user = { ...state.user, ...action.payload }
      }
    },
  },
  extraReducers: (builder) => {
    // Login (demo)
    builder
      .addCase(login.pending, (state) => {
        state.isLoading = true
        state.error = null
        state.mfaRequired = false
      })
      .addCase(login.fulfilled, (state, action) => {
        state.isLoading = false
        state.user = action.payload.user
        state.token = action.payload.token
        state.refreshToken = action.payload.refreshToken
        state.isAuthenticated = true
        state.permissions = []
        state.lastActivity = new Date()
        state.sessionExpiry = new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours
        
        // Store tokens in localStorage
        localStorage.setItem('bondx_token', action.payload.token)
        localStorage.setItem('bondx_refresh_token', action.payload.refreshToken)
      })
      .addCase(login.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.payload as string
      })

    // Register (demo)
    builder
      .addCase(register.pending, (state) => {
        state.isLoading = true
        state.error = null
      })
      .addCase(register.fulfilled, (state, action) => {
        state.isLoading = false
        state.user = action.payload.user
        state.token = action.payload.token
        state.refreshToken = action.payload.refreshToken
        state.isAuthenticated = true
        state.permissions = []
        state.lastActivity = new Date()
        state.sessionExpiry = new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours
        
        // Store tokens in localStorage
        localStorage.setItem('bondx_token', action.payload.token)
        localStorage.setItem('bondx_refresh_token', action.payload.refreshToken)
      })
      .addCase(register.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.payload as string
      })

    // Login
    builder
      .addCase(loginUser.pending, (state) => {
        state.isLoading = true
        state.error = null
        state.mfaRequired = false
      })
      .addCase(loginUser.fulfilled, (state, action) => {
        state.isLoading = false
        state.user = action.payload.user
        state.token = action.payload.accessToken
        state.refreshToken = action.payload.refreshToken
        state.isAuthenticated = true
        state.permissions = action.payload.user.permissions
        state.lastActivity = new Date()
        state.sessionExpiry = new Date(Date.now() + action.payload.expiresIn * 1000)
        
        // Store tokens in localStorage
        localStorage.setItem('bondx_token', action.payload.accessToken)
        localStorage.setItem('bondx_refresh_token', action.payload.refreshToken)
      })
      .addCase(loginUser.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.payload as string
        if (action.payload === 'MFA_REQUIRED') {
          state.mfaRequired = true
        }
      })

    // Refresh token
    builder
      .addCase(refreshToken.pending, (state) => {
        state.isLoading = true
        state.error = null
      })
      .addCase(refreshToken.fulfilled, (state, action) => {
        state.isLoading = false
        state.token = action.payload.accessToken
        state.refreshToken = action.payload.refreshToken
        state.lastActivity = new Date()
        state.sessionExpiry = new Date(Date.now() + action.payload.expiresIn * 1000)
        
        localStorage.setItem('bondx_token', action.payload.accessToken)
        localStorage.setItem('bondx_refresh_token', action.payload.refreshToken)
      })
      .addCase(refreshToken.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.payload as string
        // If refresh fails, clear auth state
        state.user = null
        state.token = null
        state.refreshToken = null
        state.isAuthenticated = false
        state.permissions = []
        localStorage.removeItem('bondx_token')
        localStorage.removeItem('bondx_refresh_token')
      })

    // Logout
    builder
      .addCase(logoutUser.pending, (state) => {
        state.isLoading = true
      })
      .addCase(logoutUser.fulfilled, (state) => {
        state.isLoading = false
        // clearAuth reducer will handle the rest
      })
      .addCase(logoutUser.rejected, (state) => {
        state.isLoading = false
        // Even if logout fails on server, clear local state
      })

    // Fetch user profile
    builder
      .addCase(fetchUserProfile.pending, (state) => {
        state.isLoading = true
        state.error = null
      })
      .addCase(fetchUserProfile.fulfilled, (state, action) => {
        state.isLoading = false
        state.user = action.payload
        state.permissions = action.payload.permissions
        state.lastActivity = new Date()
      })
      .addCase(fetchUserProfile.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.payload as string
      })

    // Update preferences
    builder
      .addCase(updateUserPreferences.pending, (state) => {
        state.isLoading = true
        state.error = null
      })
      .addCase(updateUserPreferences.fulfilled, (state, action) => {
        state.isLoading = false
        if (state.user) {
          state.user.preferences = { ...state.user.preferences, ...action.payload }
        }
      })
      .addCase(updateUserPreferences.rejected, (state, action) => {
        state.isLoading = false
        state.error = action.payload as string
      })
  },
})

// Selectors
export const selectAuth = (state: { auth: AuthState }) => state.auth
export const selectUser = (state: { auth: AuthState }) => state.auth.user
export const selectIsAuthenticated = (state: { auth: AuthState }) => state.auth.isAuthenticated
export const selectUserRole = (state: { auth: AuthState }) => state.auth.user?.role
export const selectPermissions = (state: { auth: AuthState }) => state.auth.permissions
export const selectToken = (state: { auth: AuthState }) => state.auth.token
export const selectIsLoading = (state: { auth: AuthState }) => state.auth.isLoading
export const selectError = (state: { auth: AuthState }) => state.auth.error
export const selectMfaRequired = (state: { auth: AuthState }) => state.auth.mfaRequired
export const selectSessionExpiry = (state: { auth: AuthState }) => state.auth.sessionExpiry

// Permission helpers
export const hasPermission = (permissions: Permission[], resource: string, action: string): boolean => {
  return permissions.some(
    (permission) => permission.resource === resource && permission.actions.includes(action)
  )
}

export const hasRole = (userRole: UserRole, requiredRoles: UserRole[]): boolean => {
  return requiredRoles.includes(userRole)
}

// Actions
export const { 
  clearError, 
  setMfaRequired, 
  updateLastActivity, 
  setSessionExpiry, 
  clearAuth, 
  updateUser 
} = authSlice.actions

export default authSlice.reducer
