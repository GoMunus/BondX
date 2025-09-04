import React from 'react'
import { Widget } from '@/types'

interface AlertsWidgetProps {
  widget: Widget
  userRole: string
  isExpanded: boolean
}

const AlertsWidget: React.FC<AlertsWidgetProps> = ({ widget, userRole, isExpanded }) => {
  return (
    <div className="h-full flex items-center justify-center text-text-secondary">
      <p>Alerts Widget - Coming Soon</p>
    </div>
  )
}

export default AlertsWidget
