import React from 'react'
import { Zap } from 'lucide-react'

interface AuthLayoutProps {
  children: React.ReactNode
}

const AuthLayout: React.FC<AuthLayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen bg-background flex">
      {/* Left side - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-gradient-to-br from-accent-blue/20 via-accent-purple/20 to-accent-cyan/20 items-center justify-center">
        <div className="text-center max-w-md">
          <div className="flex items-center justify-center mb-8">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-accent-blue">
              <Zap className="h-8 w-8 text-white" />
            </div>
          </div>
          
          <h1 className="text-4xl font-bold text-text-primary mb-4">
            BondX Platform
          </h1>
          
          <p className="text-xl text-text-secondary mb-8">
            Next-generation institutional bond trading platform with AI-powered analytics and real-time risk management
          </p>
          
          <div className="grid grid-cols-2 gap-6 text-center">
            <div>
              <div className="text-3xl font-bold text-accent-green mb-2">$2.5T+</div>
                  <div className="text-sm text-text-secondary">Trading Volume</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-accent-blue mb-2">50+</div>
                  <div className="text-sm text-text-secondary">Countries</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-accent-purple mb-2">99.99%</div>
                  <div className="text-sm text-text-secondary">Uptime</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-accent-cyan mb-2">&lt;50ms</div>
                  <div className="text-sm text-text-secondary">Latency</div>
            </div>
          </div>
        </div>
      </div>

      {/* Right side - Auth form */}
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="w-full max-w-md">
          {/* Mobile logo */}
          <div className="lg:hidden flex items-center justify-center mb-8">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-accent-blue">
              <Zap className="h-6 w-6 text-white" />
            </div>
            <span className="ml-3 text-2xl font-bold text-text-primary">BondX</span>
          </div>
          
          {children}
        </div>
      </div>
    </div>
  )
}

export default AuthLayout
