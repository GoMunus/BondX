import React from 'react'
import { UserRole } from '@/types'
import { Zap, TrendingUp, Shield, Brain, Building2 } from 'lucide-react'

interface WelcomeMessageProps {
  userRole: UserRole
}

const WelcomeMessage: React.FC<WelcomeMessageProps> = ({ userRole }) => {
  const getWelcomeContent = () => {
    switch (userRole) {
      case 'retail':
        return {
          title: 'Welcome to BondX',
          subtitle: 'Your gateway to institutional-grade bond trading',
          description: 'Access professional tools and insights to make informed investment decisions.',
          icon: Building2,
          color: 'text-accent-blue'
        }
      case 'professional':
        return {
          title: 'Professional Trading Suite',
          subtitle: 'Advanced tools for institutional traders',
          description: 'Execute trades with precision using our professional-grade trading interface.',
          icon: TrendingUp,
          color: 'text-accent-green'
        }
      case 'risk_manager':
        return {
          title: 'Risk Management Dashboard',
          subtitle: 'Comprehensive risk monitoring and analysis',
          description: 'Monitor portfolio risk, set limits, and analyze stress scenarios in real-time.',
          icon: Shield,
          color: 'text-accent-amber'
        }
      case 'ai_analyst':
        return {
          title: 'AI Analytics Platform',
          subtitle: 'Machine learning-powered insights',
          description: 'Leverage advanced AI models for yield prediction and market analysis.',
          icon: Brain,
          color: 'text-accent-purple'
        }
      case 'compliance':
        return {
          title: 'Compliance & Regulatory Hub',
          subtitle: 'Stay compliant with real-time monitoring',
          description: 'Monitor regulatory requirements and maintain audit trails effortlessly.',
          icon: Shield,
          color: 'text-accent-cyan'
        }
      case 'admin':
        return {
          title: 'System Administration',
          subtitle: 'Full platform control and monitoring',
          description: 'Manage users, monitor system health, and configure platform settings.',
          icon: Zap,
          color: 'text-accent-blue'
        }
      default:
        return {
          title: 'Welcome to BondX',
          subtitle: 'Next-generation bond trading platform',
          description: 'Experience the future of institutional bond trading with AI-powered insights.',
          icon: Zap,
          color: 'text-accent-blue'
        }
    }
  }

  const content = getWelcomeContent()
  const Icon = content.icon

  return (
    <div className="bg-gradient-to-r from-background-secondary to-background-tertiary border border-border rounded-xl p-6">
      <div className="flex items-start space-x-4">
        <div className={`p-3 rounded-lg bg-background ${content.color} bg-opacity-10`}>
          <Icon className="h-8 w-8" />
        </div>
        
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-text-primary mb-2">
            {content.title}
          </h1>
          <p className="text-lg text-text-secondary mb-3">
            {content.subtitle}
          </p>
          <p className="text-text-muted mb-4">
            {content.description}
          </p>
          
          <div className="flex flex-wrap gap-2">
            <span className="badge badge-info">Real-time Data</span>
            <span className="badge badge-success">AI Analytics</span>
            <span className="badge badge-warning">Risk Management</span>
            <span className="badge badge-neutral">Professional Tools</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default WelcomeMessage
