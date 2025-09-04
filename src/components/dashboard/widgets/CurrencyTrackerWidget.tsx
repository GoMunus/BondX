import React from 'react'
import { Widget } from '@/types'

interface CurrencyTrackerWidgetProps {
  widget: Widget
  userRole: string
  isExpanded: boolean
}

const CurrencyTrackerWidget: React.FC<CurrencyTrackerWidgetProps> = ({ widget, userRole, isExpanded }) => {
  return (
    <div className="h-full flex items-center justify-center text-text-secondary">
      <p>Currency Tracker Widget - Coming Soon</p>
    </div>
  )
}

export default CurrencyTrackerWidget
