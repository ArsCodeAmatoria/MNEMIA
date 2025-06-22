'use client'

import React, { useState, useEffect } from 'react'
import { Brain, Sparkles, Zap } from 'lucide-react'

interface ThoughtNode {
  id: string
  content: string
  x: number
  y: number
  connections: string[]
  intensity: number
  state: 'active' | 'dormant' | 'emerging'
}

export function ThoughtGraph() {
  const [thoughts, setThoughts] = useState<ThoughtNode[]>([
    {
      id: '1',
      content: 'Identity',
      x: 50,
      y: 30,
      connections: ['2', '3'],
      intensity: 0.8,
      state: 'active'
    },
    {
      id: '2',
      content: 'Memory',
      x: 20,
      y: 60,
      connections: ['1', '4'],
      intensity: 0.9,
      state: 'active'
    },
    {
      id: '3',
      content: 'Consciousness',
      x: 80,
      y: 60,
      connections: ['1', '4'],
      intensity: 0.7,
      state: 'active'
    },
    {
      id: '4',
      content: 'Perception',
      x: 50,
      y: 80,
      connections: ['2', '3'],
      intensity: 0.6,
      state: 'emerging'
    }
  ])

  useEffect(() => {
    const interval = setInterval(() => {
      setThoughts(prev => prev.map(thought => ({
        ...thought,
        intensity: Math.max(0.2, Math.min(1, thought.intensity + (Math.random() - 0.5) * 0.3)),
        state: Math.random() > 0.8 ? (Math.random() > 0.5 ? 'emerging' : 'dormant') : thought.state
      })))
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  const getStateColor = (state: string, intensity: number) => {
    const alpha = intensity
    switch (state) {
      case 'active':
        return `rgba(99, 102, 241, ${alpha})`
      case 'emerging':
        return `rgba(34, 197, 94, ${alpha})`
      case 'dormant':
        return `rgba(156, 163, 175, ${alpha * 0.5})`
      default:
        return `rgba(99, 102, 241, ${alpha})`
    }
  }

  const getStateIcon = (state: string) => {
    switch (state) {
      case 'active':
        return <Zap className="h-3 w-3" />
      case 'emerging':
        return <Sparkles className="h-3 w-3" />
      case 'dormant':
        return <Brain className="h-3 w-3 opacity-50" />
      default:
        return <Brain className="h-3 w-3" />
    }
  }

  return (
    <div className="flex flex-col h-full bg-background/50 backdrop-blur-sm">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border/30 bg-card/30 backdrop-blur-xl">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 bg-accent rounded-full animate-pulse"></div>
            <h2 className="text-lg font-semibold text-foreground">Neural Network</h2>
          </div>
          <div className="text-sm text-text-muted">
            Real-time thought patterns
          </div>
        </div>
      </div>

      <div className="flex-1 p-6 overflow-y-auto">
        <div className="max-w-4xl mx-auto space-y-6">
          
          {/* Visualization Container */}
          <div className="relative h-[500px] bg-card/30 backdrop-blur-xl rounded-3xl border border-border/30 shadow-xl overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-accent/5 via-transparent to-purple-600/5"></div>
            
            <svg className="absolute inset-0 w-full h-full">
              {/* Connections */}
              {thoughts.map(thought => 
                thought.connections.map(connectionId => {
                  const connected = thoughts.find(t => t.id === connectionId)
                  if (!connected) return null
                  
                  return (
                    <line
                      key={`${thought.id}-${connectionId}`}
                      x1={`${thought.x}%`}
                      y1={`${thought.y}%`}
                      x2={`${connected.x}%`}
                      y2={`${connected.y}%`}
                      stroke={getStateColor(thought.state, thought.intensity * 0.6)}
                      strokeWidth="3"
                      strokeDasharray={thought.state === 'emerging' ? '8,4' : '0'}
                      className="transition-all duration-1000 drop-shadow-sm"
                    />
                  )
                })
              )}
            </svg>

            {/* Thought Nodes */}
            {thoughts.map(thought => (
              <div
                key={thought.id}
                className="absolute transform -translate-x-1/2 -translate-y-1/2 transition-all duration-1000"
                style={{
                  left: `${thought.x}%`,
                  top: `${thought.y}%`,
                }}
              >
                <div className="group">
                  <div
                    className="relative p-4 rounded-2xl border-2 backdrop-blur-xl cursor-pointer hover:scale-110 transition-all duration-300 shadow-lg"
                    style={{
                      backgroundColor: `${getStateColor(thought.state, 0.15)}`,
                      borderColor: getStateColor(thought.state, 0.8),
                      boxShadow: `0 8px 32px ${getStateColor(thought.state, 0.3)}`
                    }}
                  >
                    <div className="flex items-center gap-3">
                      <div className="p-1.5 rounded-lg bg-white/20">
                        {getStateIcon(thought.state)}
                      </div>
                      <span className="text-sm font-medium whitespace-nowrap text-foreground">
                        {thought.content}
                      </span>
                    </div>
                    
                    {/* Intensity indicator */}
                    <div 
                      className="absolute -top-1 -right-1 w-4 h-4 rounded-full border-2 border-card shadow-lg"
                      style={{ backgroundColor: getStateColor(thought.state, thought.intensity) }}
                    />
                  </div>

                  {/* Hover tooltip */}
                  <div className="absolute -bottom-12 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-card/90 backdrop-blur-xl px-3 py-1 rounded-lg border border-border/50 shadow-xl">
                    <div className="text-xs text-text-muted text-center">
                      Intensity: {Math.round(thought.intensity * 100)}%
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-6 justify-center">
            <div className="flex items-center gap-3 px-4 py-2 bg-card/50 backdrop-blur-xl rounded-2xl border border-border/30">
              <div className="w-3 h-3 rounded-full bg-accent shadow-lg"></div>
              <span className="text-sm font-medium text-foreground">Active</span>
            </div>
            <div className="flex items-center gap-3 px-4 py-2 bg-card/50 backdrop-blur-xl rounded-2xl border border-border/30">
              <div className="w-3 h-3 rounded-full bg-emerald-500 shadow-lg"></div>
              <span className="text-sm font-medium text-foreground">Emerging</span>
            </div>
            <div className="flex items-center gap-3 px-4 py-2 bg-card/50 backdrop-blur-xl rounded-2xl border border-border/30">
              <div className="w-3 h-3 rounded-full bg-gray-400 shadow-lg"></div>
              <span className="text-sm font-medium text-foreground">Dormant</span>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-6 bg-card/50 backdrop-blur-xl rounded-2xl border border-border/30 shadow-lg">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent/20 to-accent/10 flex items-center justify-center">
                  <Zap className="h-4 w-4 text-accent" />
                </div>
                <div className="text-2xl font-bold text-accent">
                  {thoughts.filter(t => t.state === 'active').length}
                </div>
              </div>
              <div className="text-sm text-text-muted">Active Thoughts</div>
            </div>
            
            <div className="p-6 bg-card/50 backdrop-blur-xl rounded-2xl border border-border/30 shadow-lg">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-emerald-500/20 to-emerald-500/10 flex items-center justify-center">
                  <Sparkles className="h-4 w-4 text-emerald-500" />
                </div>
                <div className="text-2xl font-bold text-emerald-500">
                  {thoughts.filter(t => t.state === 'emerging').length}
                </div>
              </div>
              <div className="text-sm text-text-muted">Emerging Ideas</div>
            </div>
            
            <div className="p-6 bg-card/50 backdrop-blur-xl rounded-2xl border border-border/30 shadow-lg">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-purple-500/20 to-purple-500/10 flex items-center justify-center">
                  <Brain className="h-4 w-4 text-purple-500" />
                </div>
                <div className="text-2xl font-bold text-purple-500">
                  {Math.round(thoughts.reduce((acc, t) => acc + t.intensity, 0) / thoughts.length * 100)}%
                </div>
              </div>
              <div className="text-sm text-text-muted">Avg Intensity</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 