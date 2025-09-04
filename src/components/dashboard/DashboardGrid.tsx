import React, { useCallback, useState } from 'react'
import { Responsive, WidthProvider } from 'react-grid-layout'
import { useAppDispatch, useAppSelector } from '@/store'
import { selectWidgets, updateWidgetPosition, updateWidgetSize } from '@/store/slices/uiSlice'
import { Widget, DashboardLayout } from '@/types'

import WidgetComponent from './WidgetComponent'
import 'react-grid-layout/css/styles.css'
import 'react-resizable/css/styles.css'

const ResponsiveGridLayout = WidthProvider(Responsive)

interface DashboardGridProps {
  widgets: Widget[]
  layout: DashboardLayout | null
  userRole: string
}

const DashboardGrid: React.FC<DashboardGridProps> = ({
  widgets,
  layout,
  userRole
}) => {
  const dispatch = useAppDispatch()
  const [isDragging, setIsDragging] = useState(false)

  // Convert widgets to grid layout format
  const generateLayout = useCallback(() => {
    return widgets.map((widget) => ({
      i: widget.id,
      x: widget.position.x,
      y: widget.position.y,
      w: widget.size.width,
      h: widget.size.height,
      minW: 2,
      minH: 2,
      maxW: 12,
      maxH: 8,
      isDraggable: true,
      isResizable: true,
      isBounded: true,
      static: false,
    }))
  }, [widgets])

  // Handle layout change
  const onLayoutChange = useCallback((currentLayout: any, allLayouts: any) => {
    const breakpoint = Object.keys(allLayouts)[0] || 'lg'
    const layoutItems = allLayouts[breakpoint] || currentLayout

    layoutItems.forEach((item: any) => {
      const widget = widgets.find(w => w.id === item.i)
      if (widget) {
        // Update position
        dispatch(updateWidgetPosition({
          widgetId: item.i,
          position: { x: item.x, y: item.y }
        }))

        // Update size
        dispatch(updateWidgetSize({
          widgetId: item.i,
          size: { width: item.w, height: item.h }
        }))
      }
    })
  }, [dispatch, widgets])

  // Handle drag start
  const onDragStart = useCallback(() => {
    setIsDragging(true)
  }, [])

  // Handle drag stop
  const onDragStop = useCallback(() => {
    setIsDragging(false)
  }, [])

  // Handle resize start
  const onResizeStart = useCallback(() => {
    setIsDragging(true)
  }, [])

  // Handle resize stop
  const onResizeStop = useCallback(() => {
    setIsDragging(false)
  }, [])

  // Breakpoint configurations
  const breakpoints = { lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }
  const cols = { lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }

  // Generate default layouts for different breakpoints
  const layouts = {
    lg: generateLayout(),
    md: generateLayout(),
    sm: generateLayout(),
    xs: generateLayout(),
    xxs: generateLayout()
  }

  if (!widgets || widgets.length === 0) {
    return null
  }

  return (
    <div className="relative">
      {/* Grid layout */}
      <ResponsiveGridLayout
        className={`dashboard-grid ${isDragging ? 'dragging' : ''}`}
        layouts={layouts}
        breakpoints={breakpoints}
        cols={cols}
        rowHeight={60}
        margin={[16, 16]}
        containerPadding={[0, 0]}
        onLayoutChange={onLayoutChange}
        onDragStart={onDragStart}
        onDragStop={onDragStop}
        onResizeStart={onResizeStart}
        onResizeStop={onResizeStop}
        isDraggable={true}
        isResizable={true}
        isBounded={true}
        preventCollision={false}
        compactType="vertical"
        useCSSTransforms={true}
        transformScale={1}
        draggableHandle=".widget-drag-handle"
        resizeHandles={['se', 'sw', 'ne', 'nw']}
      >
        {widgets.map((widget) => (
          <div key={widget.id} className="widget-container">
            <WidgetComponent
              widget={widget}
              userRole={userRole}
            />
          </div>
        ))}
      </ResponsiveGridLayout>

      {/* Drag overlay indicator */}
      {isDragging && (
        <div className="fixed inset-0 pointer-events-none z-50">
          <div className="absolute top-4 right-4 bg-background-secondary border border-border rounded-lg px-3 py-2 text-sm text-text-secondary">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-accent-blue rounded-full animate-pulse"></div>
              <span>Dragging widget...</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default DashboardGrid
