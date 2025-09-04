import React from 'react'
import { AlertTriangle, RefreshCw, Home } from 'lucide-react'

interface ErrorFallbackProps {
  error: Error
  resetErrorBoundary: () => void
}

const ErrorFallback: React.FC<ErrorFallbackProps> = ({ error, resetErrorBoundary }) => {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6">
      <div className="max-w-md w-full text-center">
        {/* Error Icon */}
        <div className="mx-auto w-16 h-16 bg-accent-red/20 rounded-full flex items-center justify-center mb-6">
          <AlertTriangle className="h-8 w-8 text-accent-red" />
        </div>

        {/* Error Title */}
        <h1 className="text-2xl font-bold text-text-primary mb-4">
          Something went wrong
        </h1>

        {/* Error Message */}
        <p className="text-text-secondary mb-6">
          We encountered an unexpected error. Please try refreshing the page or contact support if the problem persists.
        </p>

        {/* Error Details (Development only) */}
        {process.env.NODE_ENV === 'development' && (
          <details className="mb-6 text-left">
            <summary className="cursor-pointer text-sm text-text-secondary hover:text-text-primary mb-2">
              Error Details
            </summary>
            <div className="bg-background-secondary border border-border rounded-lg p-4 text-xs font-mono text-text-secondary overflow-auto">
              <p className="mb-2 font-semibold">{error.name}</p>
              <p className="mb-2">{error.message}</p>
              {error.stack && (
                <pre className="whitespace-pre-wrap">{error.stack}</pre>
              )}
            </div>
          </details>
        )}

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-3 justify-center">
          <button
            onClick={resetErrorBoundary}
            className="btn btn-primary flex items-center justify-center space-x-2"
          >
            <RefreshCw className="h-4 w-4" />
            <span>Try Again</span>
          </button>

          <button
            onClick={() => window.location.href = '/dashboard'}
            className="btn btn-secondary flex items-center justify-center space-x-2"
          >
            <Home className="h-4 w-4" />
            <span>Go Home</span>
          </button>
        </div>

        {/* Support Information */}
        <div className="mt-8 pt-6 border-t border-border">
          <p className="text-sm text-text-muted mb-2">
            Need help? Contact our support team
          </p>
          <div className="flex justify-center space-x-4 text-sm">
            <a
              href="mailto:support@bondx.com"
              className="text-accent-blue hover:text-accent-blue/80 transition-colors duration-200"
            >
              support@bondx.com
            </a>
            <span className="text-text-muted">|</span>
            <a
              href="tel:+1-800-BONDX"
              className="text-accent-blue hover:text-accent-blue/80 transition-colors duration-200"
            >
              +1-800-BONDX
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ErrorFallback
