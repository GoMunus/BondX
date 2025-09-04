import React, { useState, useEffect } from 'react'
import { Link, useParams, useNavigate } from 'react-router-dom'
import { Eye, EyeOff, Shield, CheckCircle, AlertCircle } from 'lucide-react'

const ResetPassword: React.FC = () => {
  const { token } = useParams<{ token: string }>()
  const navigate = useNavigate()
  const [formData, setFormData] = useState({
    password: '',
    confirmPassword: '',
  })
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isValidating, setIsValidating] = useState(true)
  const [isValidToken, setIsValidToken] = useState(false)
  const [isSuccess, setIsSuccess] = useState(false)
  const [error, setError] = useState('')
  const [passwordStrength, setPasswordStrength] = useState(0)

  useEffect(() => {
    const validateToken = async () => {
      if (!token) {
        setIsValidToken(false)
        setIsValidating(false)
        return
      }

      try {
        // Simulate token validation
        await new Promise(resolve => setTimeout(resolve, 1500))
        setIsValidToken(true)
      } catch (err) {
        setIsValidToken(false)
      } finally {
        setIsValidating(false)
      }
    }

    validateToken()
  }, [token])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError('')

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match')
      setIsLoading(false)
      return
    }

    if (passwordStrength < 75) {
      setError('Password is not strong enough')
      setIsLoading(false)
      return
    }

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000))
      setIsSuccess(true)
      
      // Redirect to login after 3 seconds
      setTimeout(() => {
        navigate('/login')
      }, 3000)
    } catch (err: any) {
      setError(err.message || 'Failed to reset password. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))

    // Check password strength
    if (name === 'password') {
      const strength = calculatePasswordStrength(value)
      setPasswordStrength(strength)
    }
  }

  const calculatePasswordStrength = (password: string): number => {
    let strength = 0
    if (password.length >= 8) strength += 25
    if (/[A-Z]/.test(password)) strength += 25
    if (/[a-z]/.test(password)) strength += 25
    if (/[0-9]/.test(password)) strength += 25
    return strength
  }

  const getPasswordStrengthColor = () => {
    if (passwordStrength < 50) return 'bg-accent-red'
    if (passwordStrength < 75) return 'bg-accent-amber'
    return 'bg-accent-green'
  }

  const getPasswordStrengthText = () => {
    if (passwordStrength < 25) return 'Very Weak'
    if (passwordStrength < 50) return 'Weak'
    if (passwordStrength < 75) return 'Good'
    return 'Strong'
  }

  // Loading state
  if (isValidating) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-background-secondary to-background-tertiary">
        <div className="w-full max-w-md">
          <div className="card text-center">
            <div className="flex items-center justify-center mb-6">
              <div className="spinner h-12 w-12" />
            </div>
            <h1 className="text-xl font-semibold text-text-primary mb-2">
              Validating Reset Link
            </h1>
            <p className="text-text-secondary">
              Please wait while we verify your reset token...
            </p>
          </div>
        </div>
      </div>
    )
  }

  // Invalid token
  if (!isValidToken) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-background-secondary to-background-tertiary">
        <div className="w-full max-w-md">
          <div className="card text-center">
            <div className="flex items-center justify-center mb-6">
              <div className="p-4 bg-accent-red/10 rounded-full">
                <AlertCircle className="h-12 w-12 text-accent-red" />
              </div>
            </div>
            
            <h1 className="text-2xl font-bold text-text-primary mb-4">
              Invalid Reset Link
            </h1>
            
            <p className="text-text-secondary mb-6">
              This password reset link is invalid or has expired. 
              Please request a new one.
            </p>
            
            <div className="space-y-4">
              <Link
                to="/forgot-password"
                className="btn-primary w-full"
              >
                Request New Reset Link
              </Link>
              
              <Link
                to="/login"
                className="btn-secondary w-full"
              >
                Back to Sign In
              </Link>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Success state
  if (isSuccess) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-background-secondary to-background-tertiary">
        <div className="w-full max-w-md">
          <div className="card text-center">
            <div className="flex items-center justify-center mb-6">
              <div className="p-4 bg-accent-green/10 rounded-full">
                <CheckCircle className="h-12 w-12 text-accent-green" />
              </div>
            </div>
            
            <h1 className="text-2xl font-bold text-text-primary mb-4">
              Password Reset Successful
            </h1>
            
            <p className="text-text-secondary mb-6">
              Your password has been successfully updated. 
              You will be redirected to the sign-in page shortly.
            </p>
            
            <Link
              to="/login"
              className="btn-primary w-full"
            >
              Continue to Sign In
            </Link>
          </div>
        </div>
      </div>
    )
  }

  // Reset password form
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
              Create New Password
            </h1>
            <p className="text-text-secondary">
              Enter a new password for your BondX account
            </p>
          </div>

          {/* Error Alert */}
          {error && (
            <div className="alert-error mb-6">
              <p className="text-sm">{error}</p>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="form-group">
              <label htmlFor="password" className="form-label">
                New Password
              </label>
              <div className="relative">
                <input
                  type={showPassword ? 'text' : 'password'}
                  id="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  className="input pr-10"
                  placeholder="Enter your new password"
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
              {formData.password && (
                <div className="mt-2">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-text-secondary">Password Strength</span>
                    <span className={`font-medium ${passwordStrength >= 75 ? 'text-accent-green' : passwordStrength >= 50 ? 'text-accent-amber' : 'text-accent-red'}`}>
                      {getPasswordStrengthText()}
                    </span>
                  </div>
                  <div className="progress">
                    <div
                      className={`progress-bar ${getPasswordStrengthColor()}`}
                      style={{ width: `${passwordStrength}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="confirmPassword" className="form-label">
                Confirm New Password
              </label>
              <div className="relative">
                <input
                  type={showConfirmPassword ? 'text' : 'password'}
                  id="confirmPassword"
                  name="confirmPassword"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  className="input pr-10"
                  placeholder="Confirm your new password"
                  required
                  disabled={isLoading}
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary"
                  disabled={isLoading}
                >
                  {showConfirmPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </button>
              </div>
              {formData.confirmPassword && formData.password !== formData.confirmPassword && (
                <p className="form-error">Passwords do not match</p>
              )}
            </div>

            <button
              type="submit"
              disabled={isLoading || passwordStrength < 75 || formData.password !== formData.confirmPassword}
              className="btn-primary w-full flex items-center justify-center"
            >
              {isLoading ? (
                <div className="spinner h-4 w-4" />
              ) : (
                'Update Password'
              )}
            </button>
          </form>

          {/* Password Requirements */}
          <div className="mt-6 p-4 bg-background-tertiary rounded-lg">
            <h3 className="text-sm font-medium text-text-primary mb-2">
              Password Requirements
            </h3>
            <ul className="text-xs text-text-secondary space-y-1">
              <li className={formData.password.length >= 8 ? 'text-accent-green' : ''}>
                • At least 8 characters long
              </li>
              <li className={/[A-Z]/.test(formData.password) ? 'text-accent-green' : ''}>
                • Contains uppercase letter
              </li>
              <li className={/[a-z]/.test(formData.password) ? 'text-accent-green' : ''}>
                • Contains lowercase letter
              </li>
              <li className={/[0-9]/.test(formData.password) ? 'text-accent-green' : ''}>
                • Contains number
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ResetPassword
