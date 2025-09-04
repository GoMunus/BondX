import React from 'react'
import { Widget } from '@/types'

interface NewsFeedWidgetProps {
  widget: Widget
  userRole: string
  isExpanded: boolean
}

const NewsFeedWidget: React.FC<NewsFeedWidgetProps> = ({ widget, userRole, isExpanded }) => {
  return (
    <div className="h-full flex items-center justify-center text-text-secondary">
      <p>News Feed Widget - Coming Soon</p>
    </div>
  )
}

export default NewsFeedWidget
