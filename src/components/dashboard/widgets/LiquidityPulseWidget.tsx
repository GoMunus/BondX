import React from 'react'
import { Widget } from '@/types'

interface LiquidityPulseWidgetProps {
  widget: Widget
  userRole: string
  isExpanded: boolean
}

const LiquidityPulseWidget: React.FC<LiquidityPulseWidgetProps> = ({ widget, userRole, isExpanded }) => {
  return (
    <div className="h-full flex items-center justify-center text-text-secondary">
      <p>Liquidity Pulse Widget - Coming Soon</p>
    </div>
  )
}

export default LiquidityPulseWidget
