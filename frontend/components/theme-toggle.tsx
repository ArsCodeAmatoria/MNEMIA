'use client'

import { useState, useEffect } from 'react'
import { Sun, Moon } from 'lucide-react'

export function ThemeToggle() {
  const [isDark, setIsDark] = useState(false)

  useEffect(() => {
    // Check if user has a theme preference stored
    const stored = localStorage.getItem('theme')
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    
    if (stored === 'dark' || (!stored && prefersDark)) {
      setIsDark(true)
      document.documentElement.classList.add('dark')
    } else {
      setIsDark(false)
      document.documentElement.classList.remove('dark')
    }
  }, [])

  const toggleTheme = () => {
    const newIsDark = !isDark
    setIsDark(newIsDark)
    
    if (newIsDark) {
      document.documentElement.classList.add('dark')
      localStorage.setItem('theme', 'dark')
    } else {
      document.documentElement.classList.remove('dark')
      localStorage.setItem('theme', 'light')
    }
  }

  return (
    <button
      onClick={toggleTheme}
      className="relative p-3 rounded-xl bg-gradient-to-r from-accent/10 to-purple-600/10 border border-accent/20 hover:from-accent/20 hover:to-purple-600/20 transition-all duration-200 group overflow-hidden"
      aria-label="Toggle theme"
    >
      <div className="absolute inset-0 bg-gradient-to-r from-accent/20 to-purple-600/20 opacity-0 group-hover:opacity-100 transition-opacity duration-200"></div>
      <div className="relative">
        {isDark ? (
          <Sun className="h-4 w-4 text-accent transition-transform group-hover:scale-110 group-hover:rotate-12" />
        ) : (
          <Moon className="h-4 w-4 text-accent transition-transform group-hover:scale-110 group-hover:-rotate-12" />
        )}
      </div>
    </button>
  )
} 