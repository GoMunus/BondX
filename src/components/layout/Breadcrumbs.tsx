import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { ChevronRight, Home } from 'lucide-react'

const Breadcrumbs: React.FC = () => {
  const location = useLocation()

  const generateBreadcrumbs = () => {
    const pathnames = location.pathname.split('/').filter((x) => x)
    
    if (pathnames.length === 0) {
      return []
    }

    const breadcrumbs = [
      {
        name: 'Home',
        path: '/dashboard',
        icon: Home
      }
    ]

    let currentPath = ''
    pathnames.forEach((name, index) => {
      currentPath += `/${name}`
      
      // Convert path to readable name
      const readableName = name
        .split('-')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ')

      breadcrumbs.push({
        name: readableName,
        path: currentPath,
        icon: null
      })
    })

    return breadcrumbs
  }

  const breadcrumbs = generateBreadcrumbs()

  if (breadcrumbs.length <= 1) {
    return null
  }

  return (
    <nav className="border-b border-border bg-background-secondary px-6 py-3">
      <div className="mx-auto max-w-7xl">
        <ol className="flex items-center space-x-2 text-sm">
          {breadcrumbs.map((breadcrumb, index) => {
            const isLast = index === breadcrumbs.length - 1
            const Icon = breadcrumb.icon

            return (
              <li key={breadcrumb.path} className="flex items-center">
                {index > 0 && (
                  <ChevronRight className="mx-2 h-4 w-4 text-text-muted" />
                )}
                
                {isLast ? (
                  <span className="text-text-primary font-medium">
                    {Icon && <Icon className="mr-2 h-4 w-4 inline" />}
                    {breadcrumb.name}
                  </span>
                ) : (
                  <Link
                    to={breadcrumb.path}
                    className="flex items-center text-text-secondary hover:text-text-primary transition-colors duration-200"
                  >
                    {Icon && <Icon className="mr-2 h-4 w-4" />}
                    {breadcrumb.name}
                  </Link>
                )}
              </li>
            )
          })}
        </ol>
      </div>
    </nav>
  )
}

export default Breadcrumbs
