import React from 'react'
import { UserRole } from '@/types'
import { 
  TrendingUp, 
  BarChart3, 
  Shield, 
  Brain, 
  FileText, 
  Settings,
  Plus,
  Search,
  Download,
  Upload
} from 'lucide-react'

interface QuickActionsProps {
  userRole: UserRole
}

const QuickActions: React.FC<QuickActionsProps> = ({ userRole }) => {
  const getQuickActions = () => {
    const baseActions = [
      {
        title: 'Search Bonds',
        description: 'Find specific bonds',
        icon: Search,
        color: 'bg-accent-blue/20 text-accent-blue',
        href: '/search'
      },
      {
        title: 'View Reports',
        description: 'Access reports',
        icon: FileText,
        color: 'bg-accent-green/20 text-accent-green',
        href: '/reports'
      }
    ]

    const roleActions = {
      retail: [
        {
          title: 'Start Trading',
          description: 'Begin bond trading',
          icon: TrendingUp,
          color: 'bg-accent-green/20 text-accent-green',
          href: '/trading'
        },
        {
          title: 'Portfolio View',
          description: 'View your portfolio',
          icon: BarChart3,
          color: 'bg-accent-blue/20 text-accent-blue',
          href: '/portfolio'
        }
      ],
      professional: [
        {
          title: 'Advanced Trading',
          description: 'Professional trading tools',
          icon: TrendingUp,
          color: 'bg-accent-green/20 text-accent-green',
          href: '/trading/advanced'
        },
        {
          title: 'Market Analysis',
          description: 'Deep market insights',
          icon: BarChart3,
          color: 'bg-accent-purple/20 text-accent-purple',
          href: '/analytics/market'
        }
      ],
      risk_manager: [
        {
          title: 'Risk Dashboard',
          description: 'Monitor risk metrics',
          icon: Shield,
          color: 'bg-accent-amber/20 text-accent-amber',
          href: '/risk'
        },
        {
          title: 'Stress Testing',
          description: 'Run stress scenarios',
          icon: Shield,
          color: 'bg-accent-red/20 text-accent-red',
          href: '/risk/stress'
        }
      ],
      ai_analyst: [
        {
          title: 'AI Models',
          description: 'Manage ML models',
          icon: Brain,
          color: 'bg-accent-purple/20 text-accent-purple',
          href: '/analytics/models'
        },
        {
          title: 'Predictions',
          description: 'View AI predictions',
          icon: Brain,
          color: 'bg-accent-cyan/20 text-accent-cyan',
          href: '/analytics/predictions'
        }
      ],
      compliance: [
        {
          title: 'Compliance Check',
          description: 'Run compliance checks',
          icon: Shield,
          color: 'bg-accent-amber/20 text-accent-amber',
          href: '/compliance'
        },
        {
          title: 'Audit Trail',
          description: 'View audit logs',
          icon: FileText,
          color: 'bg-accent-blue/20 text-accent-blue',
          href: '/compliance/audit'
        }
      ],
      admin: [
        {
          title: 'User Management',
          description: 'Manage users',
          icon: Settings,
          color: 'bg-accent-blue/20 text-accent-blue',
          href: '/admin/users'
        },
        {
          title: 'System Health',
          description: 'Monitor system',
          icon: Settings,
          color: 'bg-accent-green/20 text-accent-green',
          href: '/admin/health'
        }
      ]
    }

    return [...baseActions, ...(roleActions[userRole] || [])]
  }

  const actions = getQuickActions()

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-text-primary">Quick Actions</h2>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {actions.map((action, index) => {
          const Icon = action.icon
          return (
            <button
              key={index}
              className="group p-4 bg-background-secondary border border-border rounded-lg hover:border-accent-blue hover:bg-accent-blue/5 transition-all duration-200 text-left"
              onClick={() => {
                // TODO: Implement navigation
                console.log('Navigate to:', action.href)
              }}
            >
              <div className={`w-12 h-12 rounded-lg ${action.color} flex items-center justify-center mb-3 group-hover:scale-110 transition-transform duration-200`}>
                <Icon className="h-6 w-6" />
              </div>
              
              <h3 className="font-medium text-text-primary mb-1 group-hover:text-accent-blue transition-colors duration-200">
                {action.title}
              </h3>
              
              <p className="text-sm text-text-secondary">
                {action.description}
              </p>
            </button>
          )
        })}
      </div>
    </div>
  )
}

export default QuickActions
