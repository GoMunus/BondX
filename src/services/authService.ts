import { User } from '@/types'
import { apiService } from './api'

export const authService = {
  login: async (credentials: { email: string; password: string; mfaCode?: string }) => {
    try {
      // Use username field for login (backend expects username, not email)
      const loginData = {
        username: credentials.email.split('@')[0] || credentials.email, // Extract username from email
        password: credentials.password
      }
      
      const response = await apiService.login(loginData)
      
      // Store the token
      if (response.access_token) {
        localStorage.setItem('auth_token', response.access_token)
        if (response.refresh_token) {
          localStorage.setItem('refresh_token', response.refresh_token)
        }
      }
      
      return {
        data: {
          user: {
            id: response.user.id,
            email: response.user.email,
            firstName: response.user.first_name,
            lastName: response.user.last_name,
            role: response.user.role as const,
            organization: response.user.organization,
            permissions: response.user.permissions
          },
          accessToken: response.access_token,
          refreshToken: response.refresh_token,
          expiresIn: response.expires_in
        }
      }
    } catch (error) {
      console.error('Login failed:', error)
      throw error
    }
  },

  refreshToken: async () => {
    try {
      const refreshToken = localStorage.getItem('refresh_token')
      if (!refreshToken) {
        throw new Error('No refresh token available')
      }
      
      const response = await fetch(`${import.meta.env.VITE_API_BASE_URL}/api/v1/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ refresh_token: refreshToken })
      })
      
      if (!response.ok) {
        throw new Error('Token refresh failed')
      }
      
      const data = await response.json()
      
      // Store the new token
      if (data.access_token) {
        localStorage.setItem('auth_token', data.access_token)
      }
      
      return {
        data: {
          accessToken: data.access_token,
          refreshToken: refreshToken, // Keep the same refresh token
          expiresIn: data.expires_in
        }
      }
    } catch (error) {
      console.error('Token refresh failed:', error)
      // Clear tokens on refresh failure
      localStorage.removeItem('auth_token')
      localStorage.removeItem('refresh_token')
      throw error
    }
  },

  logout: async () => {
    try {
      await apiService.logout()
    } catch (error) {
      console.error('Logout API call failed:', error)
      // Continue with logout even if API call fails
    } finally {
      // Always clear local storage
      localStorage.removeItem('auth_token')
      localStorage.removeItem('refresh_token')
    }
    
    return {
      data: { success: true }
    }
  },

  getProfile: async () => {
    try {
      const response = await apiService.getUserProfile()
      
      return {
        data: {
          id: response.data.id,
          email: response.data.email,
          firstName: response.data.first_name,
          lastName: response.data.last_name,
          role: response.data.role as const,
          organization: response.data.organization,
          permissions: response.data.permissions || []
        } as User
      }
    } catch (error) {
      console.error('Get profile failed:', error)
      throw error
    }
  },

  // Helper method to check if user is authenticated
  isAuthenticated: () => {
    const token = localStorage.getItem('auth_token')
    return !!token
  },

  // Helper method to get current token
  getToken: () => {
    return localStorage.getItem('auth_token')
  }
}
