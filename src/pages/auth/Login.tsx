import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { useAppDispatch } from '@/store'
import { login } from '@/store/slices/authSlice'
import { Eye, EyeOff, ArrowRight, Shield } from 'lucide-react'

const Login: React.FC = () => {
  const dispatch = useAppDispatch()
  const [formData, setFormData] = useState({
    email: '',
    password: '',
  })
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError('')

    try {
      await dispatch(login(formData)).unwrap()
    } catch (err: any) {
      setError(err.message || 'Login failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }))
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-background-secondary to-background-tertiary">
      <div className="w-full max-w-md">
        <div className="card">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="flex items-center justify-center mb-4">
              <div className="p-3 bg-accent-blue/10 rounded-xl">
                <Shield className="h-8 w-8 text-accent-blue" />
              </div>
            </div>
            <h1 className="text-2xl font-bold text-text-primary mb-2">
              Welcome back to BondX
            </h1>
            <p className="text-text-secondary">
              Sign in to your institutional bond trading account
            </p>
          </div>

          {/* Error Alert */}
          {error && (
            <div className="alert-error mb-6">
              <p className="text-sm">{error}</p>
            </div>
          )}

          {/* Login Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="form-group">
              <label htmlFor="email" className="form-label">
                Email Address
              </label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                className="input"
                placeholder="Enter your email"
                required
                disabled={isLoading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="password" className="form-label">
                Password
              </label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  id="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  className="input pr-10"
                  placeholder="Enter your password"
                  required
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary"
                  disabled={isLoading}
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  className="rounded border-border text-accent-blue focus:ring-accent-blue/50"
                />
                <span className="ml-2 text-sm text-text-secondary">
                  Remember me
                </span>
              </label>
              <Link
                to="/forgot-password"
                className="text-sm text-accent-blue hover:text-accent-blue/80 transition-colors"
              >
                Forgot password?
              </Link>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="btn-primary w-full flex items-center justify-center"
            >
              {isLoading ? (
                <div className="spinner h-4 w-4" />
              ) : (
                <>
                  Sign In
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </button>
          </form>

          {/* Demo Credentials */}
          <div className="mt-6 p-4 bg-background-tertiary rounded-lg">
            <p className="text-sm text-text-secondary mb-2">Demo Credentials:</p>
            <p className="text-xs text-text-muted">
              Email: demo@bondx.com<br />
              Password: demo123
            </p>
          </div>

          {/* Footer */}
          <div className="mt-6 text-center">
            <p className="text-sm text-text-secondary">
              Don't have an account?{' '}
              <Link
                to="/register"
                className="text-accent-blue hover:text-accent-blue/80 transition-colors font-medium"
              >
                Sign up here
              </Link>
            </p>
          </div>
        </div>

        {/* Features */}
        <div className="mt-8 grid grid-cols-2 gap-4 text-center">
          <div className="p-4">
            <div className="text-2xl font-bold text-accent-green mb-1">99.9%</div>
            <div className="text-xs text-text-secondary">Uptime</div>
          </div>
          <div className="p-4">
            <div className="text-2xl font-bold text-accent-blue mb-1">$50B+</div>
            <div className="text-xs text-text-secondary">Assets Managed</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Login
