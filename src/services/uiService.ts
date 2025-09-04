import { DashboardLayout } from '@/types'

export const uiService = {
  getDashboardLayouts: async () => {
    // Mock implementation for demo
    return {
      data: [] as DashboardLayout[]
    }
  },

  saveDashboardLayout: async (layout: DashboardLayout) => {
    // Mock implementation for demo
    return {
      data: layout
    }
  },

  updateDashboardLayout: async (id: string, updates: Partial<DashboardLayout>) => {
    // Mock implementation for demo
    return {
      data: { id, ...updates }
    }
  },

  deleteDashboardLayout: async (id: string) => {
    // Mock implementation for demo
    return {
      data: { success: true }
    }
  }
}
