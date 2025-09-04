import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { useAppDispatch } from '@/store'
import { register } from '@/store/slices/authSlice'
import { Eye, EyeOff, ArrowRight, Shield, Check } from 'lucide-react'

const Register: React.FC = () => {
  const dispatch = useAppDispatch()
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: '',
    organization: '',
    phone: '',
    agreeToTerms: false,
  })
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [passwordStrength, setPasswordStrength] = useState(0)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError('')

    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match')
      setIsLoading(false)
      return
    }

    if (!formData.agreeToTerms) {
      setError('Please agree to the terms and conditions')
      setIsLoading(false)
      return
    }

    try {
      await dispatch(register(formData)).unwrap()
    } catch (err: any) {
      setError(err.message || 'Registration failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
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

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-background-secondary to-background-tertiary py-8">
      <div className="w-full max-w-lg">
        <div className="card">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="flex items-center justify-center mb-4">
              <div className="p-3 bg-accent-blue/10 rounded-xl">
                <Shield className="h-8 w-8 text-accent-blue" />
              </div>
            </div>
            <h1 className="text-2xl font-bold text-text-primary mb-2">
              Join BondX Platform
            </h1>
            <p className="text-text-secondary">
              Create your institutional trading account
            </p>
          </div>

          {/* Error Alert */}
          {error && (
            <div className="alert-error mb-6">
              <p className="text-sm">{error}</p>
            </div>
          )}

          {/* Registration Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="form-group">
                <label htmlFor="firstName" className="form-label">
                  First Name
                </label>
                <input
                  type="text"
                  id="firstName"
                  name="firstName"
                  value={formData.firstName}
                  onChange={handleChange}
                  className="input"
                  placeholder="First name"
                  required
                  disabled={isLoading}
                />
              </div>

              <div className="form-group">
                <label htmlFor="lastName" className="form-label">
                  Last Name
                </label>
                <input
                  type="text"
                  id="lastName"
                  name="lastName"
                  value={formData.lastName}
                  onChange={handleChange}
                  className="input"
                  placeholder="Last name"
                  required
                  disabled={isLoading}
                />
              </div>
            </div>

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
              <label htmlFor="organization" className="form-label">
                Organization
              </label>
              <input
                type="text"
                id="organization"
                name="organization"
                value={formData.organization}
                onChange={handleChange}
                className="input"
                placeholder="Company or institution name"
                required
                disabled={isLoading}
              />
            </div>

            <div className="form-group">
              <label htmlFor="phone" className="form-label">
                Phone Number
              </label>
              <input
                type="tel"
                id="phone"
                name="phone"
                value={formData.phone}
                onChange={handleChange}
                className="input"
                placeholder="+1 (555) 123-4567"
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
                  placeholder="Create a strong password"
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
                Confirm Password
              </label>
              <div className="relative">
                <input
                  type={showConfirmPassword ? 'text' : 'password'}
                  id="confirmPassword"
                  name="confirmPassword"
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  className="input pr-10"
                  placeholder="Confirm your password"
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

            <div className="form-group">
              <label className="flex items-start">
                <input
                  type="checkbox"
                  name="agreeToTerms"
                  checked={formData.agreeToTerms}
                  onChange={handleChange}
                  className="rounded border-border text-accent-blue focus:ring-accent-blue/50 mt-1"
                  required
                  disabled={isLoading}
                />
                <span className="ml-2 text-sm text-text-secondary">
                  I agree to the{' '}
                  <Link to="/terms" className="text-accent-blue hover:text-accent-blue/80">
                    Terms of Service
                  </Link>{' '}
                  and{' '}
                  <Link to="/privacy" className="text-accent-blue hover:text-accent-blue/80">
                    Privacy Policy
                  </Link>
                </span>
              </label>
            </div>

            <button
              type="submit"
              disabled={isLoading || !formData.agreeToTerms}
              className="btn-primary w-full flex items-center justify-center"
            >
              {isLoading ? (
                <div className="spinner h-4 w-4" />
              ) : (
                <>
                  Create Account
                  <ArrowRight className="ml-2 h-4 w-4" />
                </>
              )}
            </button>
          </form>

          {/* Features */}
          <div className="mt-6 grid grid-cols-3 gap-4 p-4 bg-background-tertiary rounded-lg">
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Check className="h-4 w-4 text-accent-green" />
              </div>
              <p className="text-xs text-text-secondary">Bank-Grade Security</p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Check className="h-4 w-4 text-accent-green" />
              </div>
              <p className="text-xs text-text-secondary">Real-time Analytics</p>
            </div>
            <div className="text-center">
              <div className="flex items-center justify-center mb-2">
                <Check className="h-4 w-4 text-accent-green" />
              </div>
              <p className="text-xs text-text-secondary">24/7 Support</p>
            </div>
          </div>

          {/* Footer */}
          <div className="mt-6 text-center">
            <p className="text-sm text-text-secondary">
              Already have an account?{' '}
              <Link
                to="/login"
                className="text-accent-blue hover:text-accent-blue/80 transition-colors font-medium"
              >
                Sign in here
              </Link>
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Register
