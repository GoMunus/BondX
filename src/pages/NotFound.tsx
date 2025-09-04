import React from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Home, ArrowLeft, Search, HelpCircle } from 'lucide-react'

const NotFound: React.FC = () => {
  const navigate = useNavigate()

  const handleGoBack = () => {
    navigate(-1)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background-secondary to-background-tertiary flex items-center justify-center p-4">
      <div className="w-full max-w-2xl text-center">
        {/* 404 Illustration */}
        <div className="mb-8">
          <div className="text-8xl font-bold text-accent-blue/20 mb-4">404</div>
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-accent-blue to-accent-purple opacity-20 rounded-full blur-3xl"></div>
            <div className="relative bg-background-secondary rounded-full p-8 inline-block">
              <Search className="h-16 w-16 text-accent-blue" />
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-text-primary mb-4">
            Page Not Found
          </h1>
          <p className="text-lg text-text-secondary mb-2">
            Oops! The page you're looking for doesn't exist.
          </p>
          <p className="text-text-muted">
            It might have been moved, deleted, or you entered the wrong URL.
          </p>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-8">
          <Link
            to="/dashboard"
            className="btn-primary flex items-center"
          >
            <Home className="mr-2 h-4 w-4" />
            Go to Dashboard
          </Link>
          
          <button
            onClick={handleGoBack}
            className="btn-secondary flex items-center"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Go Back
          </button>
        </div>

        {/* Quick Links */}
        <div className="card max-w-md mx-auto">
          <h2 className="text-lg font-semibold text-text-primary mb-4">
            Quick Links
          </h2>
          <div className="space-y-3">
            <Link
              to="/dashboard"
              className="block w-full text-left p-3 rounded-lg bg-background-tertiary hover:bg-background text-text-secondary hover:text-text-primary transition-colors"
            >
              <div className="font-medium">Dashboard</div>
              <div className="text-sm text-text-muted">View your portfolio overview</div>
            </Link>
            
            <Link
              to="/trading"
              className="block w-full text-left p-3 rounded-lg bg-background-tertiary hover:bg-background text-text-secondary hover:text-text-primary transition-colors"
            >
              <div className="font-medium">Trading</div>
              <div className="text-sm text-text-muted">Access bond trading platform</div>
            </Link>
            
            <Link
              to="/analytics"
              className="block w-full text-left p-3 rounded-lg bg-background-tertiary hover:bg-background text-text-secondary hover:text-text-primary transition-colors"
            >
              <div className="font-medium">Analytics</div>
              <div className="text-sm text-text-muted">AI-powered market insights</div>
            </Link>
            
            <Link
              to="/risk"
              className="block w-full text-left p-3 rounded-lg bg-background-tertiary hover:bg-background text-text-secondary hover:text-text-primary transition-colors"
            >
              <div className="font-medium">Risk Management</div>
              <div className="text-sm text-text-muted">Portfolio risk analysis</div>
            </Link>
          </div>
        </div>

        {/* Help Section */}
        <div className="mt-8">
          <div className="flex items-center justify-center text-text-muted mb-4">
            <HelpCircle className="h-4 w-4 mr-2" />
            <span className="text-sm">Need help?</span>
          </div>
          <div className="flex items-center justify-center space-x-6 text-sm">
            <a
              href="mailto:support@bondx.com"
              className="text-accent-blue hover:text-accent-blue/80 transition-colors"
            >
              Contact Support
            </a>
            <a
              href="/docs"
              className="text-accent-blue hover:text-accent-blue/80 transition-colors"
            >
              Documentation
            </a>
            <a
              href="/status"
              className="text-accent-blue hover:text-accent-blue/80 transition-colors"
            >
              System Status
            </a>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-text-muted text-sm">
          <p>© 2024 BondX. All rights reserved.</p>
        </div>
      </div>
    </div>
  )
}

export default NotFound
