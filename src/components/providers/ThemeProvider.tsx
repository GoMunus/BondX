import React, { createContext, useContext, useEffect, useState } from 'react'

type Theme = 'light' | 'dark' | 'auto'

interface ThemeContextType {
  theme: Theme
  setTheme: (theme: Theme) => void
  isDark: boolean
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export const useTheme = () => {
  const context = useContext(ThemeContext)
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}

interface ThemeProviderProps {
  children: React.ReactNode
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setTheme] = useState<Theme>(() => {
    // Check localStorage first
    const savedTheme = localStorage.getItem('bondx-theme') as Theme
    if (savedTheme && ['light', 'dark', 'auto'].includes(savedTheme)) {
      return savedTheme
    }
    
    // Check system preference
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark'
    }
    
    return 'dark' // Default to dark theme for BondX
  })

  const [isDark, setIsDark] = useState(false)

  useEffect(() => {
    const updateTheme = () => {
      let effectiveTheme: 'light' | 'dark'
      
      if (theme === 'auto') {
        effectiveTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
      } else {
        effectiveTheme = theme
      }
      
      setIsDark(effectiveTheme === 'dark')
      
      // Update document classes
      if (effectiveTheme === 'dark') {
        document.documentElement.classList.add('dark')
        document.documentElement.classList.remove('light')
      } else {
        document.documentElement.classList.add('light')
        document.documentElement.classList.remove('dark')
      }
      
      // Update CSS custom properties for theme colors
      const root = document.documentElement
      if (effectiveTheme === 'dark') {
        root.style.setProperty('--color-background', '#1a1d29')
        root.style.setProperty('--color-background-secondary', '#2a2f3d')
        root.style.setProperty('--color-background-tertiary', '#3a3f4d')
        root.style.setProperty('--color-text-primary', '#f8fafc')
        root.style.setProperty('--color-text-secondary', '#94a3b8')
        root.style.setProperty('--color-text-muted', '#64748b')
        root.style.setProperty('--color-border', '#374151')
        root.style.setProperty('--color-border-secondary', '#4b5563')
      } else {
        root.style.setProperty('--color-background', '#ffffff')
        root.style.setProperty('--color-background-secondary', '#f8fafc')
        root.style.setProperty('--color-background-tertiary', '#f1f5f9')
        root.style.setProperty('--color-text-primary', '#0f172a')
        root.style.setProperty('--color-text-secondary', '#475569')
        root.style.setProperty('--color-text-muted', '#94a3b8')
        root.style.setProperty('--color-border', '#e2e8f0')
        root.style.setProperty('--color-border-secondary', '#cbd5e1')
      }
    }

    updateTheme()

    // Listen for system theme changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)')
    const handleChange = () => {
      if (theme === 'auto') {
        updateTheme()
      }
    }

    mediaQuery.addEventListener('change', handleChange)

    return () => {
      mediaQuery.removeEventListener('change', handleChange)
    }
  }, [theme])

  useEffect(() => {
    // Save theme preference to localStorage
    localStorage.setItem('bondx-theme', theme)
  }, [theme])

  const handleSetTheme = (newTheme: Theme) => {
    setTheme(newTheme)
  }

  const value: ThemeContextType = {
    theme,
    setTheme: handleSetTheme,
    isDark,
  }

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  )
}
