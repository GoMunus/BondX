import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowLeft, Mail, Shield, CheckCircle } from 'lucide-react'

const ForgotPassword: React.FC = () => {
  const [email, setEmail] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isSubmitted, setIsSubmitted] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError('')

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000))
      setIsSubmitted(true)
    } catch (err: any) {
      setError(err.message || 'Failed to send reset email. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  if (isSubmitted) {
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
              Check Your Email
            </h1>
            
            <p className="text-text-secondary mb-6">
              We've sent a password reset link to{' '}
              <span className="text-text-primary font-medium">{email}</span>
            </p>
            
            <div className="space-y-4">
              <p className="text-sm text-text-muted">
                Didn't receive the email? Check your spam folder or{' '}
                <button
                  onClick={() => setIsSubmitted(false)}
                  className="text-accent-blue hover:text-accent-blue/80 font-medium"
                >
                  try again
                </button>
              </p>
              
              <Link
                to="/login"
                className="btn-secondary w-full flex items-center justify-center"
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Sign In
              </Link>
            </div>
          </div>
        </div>
      </div>
    )
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
              Reset Your Password
            </h1>
            <p className="text-text-secondary">
              Enter your email address and we'll send you a link to reset your password
            </p>
          </div>

          {/* Error Alert */}
          {error && (
            <div className="alert-error mb-6">
              <p className="text-sm">{error}</p>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="form-group">
              <label htmlFor="email" className="form-label">
                Email Address
              </label>
              <div className="relative">
                <input
                  type="email"
                  id="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="input pl-10"
                  placeholder="Enter your email address"
                  required
                  disabled={isLoading}
                />
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-text-muted" />
              </div>
            </div>

            <button
              type="submit"
              disabled={isLoading || !email}
              className="btn-primary w-full flex items-center justify-center"
            >
              {isLoading ? (
                <div className="spinner h-4 w-4" />
              ) : (
                <>
                  <Mail className="mr-2 h-4 w-4" />
                  Send Reset Link
                </>
              )}
            </button>
          </form>

          {/* Security Note */}
          <div className="mt-6 p-4 bg-background-tertiary rounded-lg">
            <h3 className="text-sm font-medium text-text-primary mb-2">
              Security Note
            </h3>
            <p className="text-xs text-text-secondary">
              For your security, the reset link will expire in 15 minutes. 
              If you don't receive the email within a few minutes, please check your spam folder.
            </p>
          </div>

          {/* Footer */}
          <div className="mt-6 flex items-center justify-between">
            <Link
              to="/login"
              className="text-sm text-text-secondary hover:text-text-primary transition-colors flex items-center"
            >
              <ArrowLeft className="mr-1 h-4 w-4" />
              Back to Sign In
            </Link>
            <Link
              to="/register"
              className="text-sm text-accent-blue hover:text-accent-blue/80 transition-colors"
            >
              Create Account
            </Link>
          </div>
        </div>

        {/* Help */}
        <div className="mt-8 text-center">
          <p className="text-sm text-text-muted">
            Need help?{' '}
            <a
              href="mailto:support@bondx.com"
              className="text-accent-blue hover:text-accent-blue/80 transition-colors"
            >
              Contact Support
            </a>
          </p>
        </div>
      </div>
    </div>
  )
}

export default ForgotPassword
