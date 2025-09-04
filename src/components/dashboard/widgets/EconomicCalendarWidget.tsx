import React from 'react'
import { Widget } from '@/types'

interface EconomicCalendarWidgetProps {
  widget: Widget
  userRole: string
  isExpanded: boolean
}

const EconomicCalendarWidget: React.FC<EconomicCalendarWidgetProps> = ({ widget, userRole, isExpanded }) => {
  return (
    <div className="h-full flex items-center justify-center text-text-secondary">
      <p>Economic Calendar Widget - Coming Soon</p>
    </div>
  )
}

export default EconomicCalendarWidget
