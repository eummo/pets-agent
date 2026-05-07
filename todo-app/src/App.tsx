import { useState, useEffect, useCallback, useRef, useMemo } from 'react'

// 类型定义
interface Todo {
  id: string
  text: string
  completed: boolean
  createdAt: number
  priority: 'high' | 'medium' | 'low'
  tags: string[]
  dueDate: string | null
  description: string
}

type FilterType = 'all' | 'pending' | 'completed'
type SortType = 'created' | 'priority' | 'dueDate'

interface EditModalProps {
  todo: Todo | null
  isOpen: boolean
  onClose: () => void
  onSave: (todo: Todo) => void
  allTags: string[]
}

interface SearchFilterProps {
  searchQuery: string
  setSearchQuery: (query: string) => void
  filter: FilterType
  setFilter: (filter: FilterType) => void
  sortBy: SortType
  setSortBy: (sort: SortType) => void
  priorityFilter: 'all' | 'high' | 'medium' | 'low'
  setPriorityFilter: (p: 'all' | 'high' | 'medium' | 'low') => void
  tagFilter: string
  setTagFilter: (tag: string) => void
}

// 常量
const STORAGE_KEY = 'react-todo-app-todos'
const PRIORITY_LABELS = { high: '高', medium: '中', low: '低' }
const PRIORITY_COLORS = { high: '#ef4444', medium: '#f59e0b', low: '#22c55e' }
const PRIORITY_BG = { high: '#fef2f2', medium: '#fffbeb', low: '#f0fdf4' }

// 工具函数
function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).substring(2)
}

function formatDate(dateStr: string | null): string {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}

function isDueSoon(dueDate: string | null): boolean {
  if (!dueDate) return false
  const due = new Date(dueDate)
  const now = new Date()
  const diff = due.getTime() - now.getTime()
  const days = diff / (1000 * 60 * 60 * 24)
  return days >= 0 && days <= 2
}

function isOverdue(dueDate: string | null): boolean {
  if (!dueDate) return false
  return new Date(dueDate) < new Date()
}
