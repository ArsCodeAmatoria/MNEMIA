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
    <div className="h-full p-6">
      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Thought Patterns</h2>
        <p className="text-sm text-muted-foreground">
          Real-time visualization of MNEMIA's cognitive processes
        </p>
      </div>

      <div className="relative h-[400px] bg-neural-900/20 rounded-lg border border-neural-600/30 overflow-hidden">
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
                  stroke={getStateColor(thought.state, thought.intensity * 0.5)}
                  strokeWidth="2"
                  strokeDasharray={thought.state === 'emerging' ? '5,5' : '0'}
                  className="transition-all duration-1000"
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
            <div
              className="relative p-3 rounded-full border-2 backdrop-blur-sm cursor-pointer hover:scale-110 transition-transform"
              style={{
                backgroundColor: getStateColor(thought.state, 0.2),
                borderColor: getStateColor(thought.state, 0.8),
                boxShadow: `0 0 ${thought.intensity * 20}px ${getStateColor(thought.state, 0.5)}`
              }}
            >
              <div className="flex items-center space-x-2">
                {getStateIcon(thought.state)}
                <span className="text-xs font-medium whitespace-nowrap">
                  {thought.content}
                </span>
              </div>
              
              {/* Intensity indicator */}
              <div className="absolute -bottom-1 -right-1 w-3 h-3 rounded-full border border-background"
                style={{ backgroundColor: getStateColor(thought.state, thought.intensity) }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="mt-6 flex space-x-6 text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-quantum-500" />
          <span>Active</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span>Emerging</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-full bg-neutral-400" />
          <span>Dormant</span>
        </div>
      </div>

      {/* Stats */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="p-3 bg-neural-800/20 rounded-lg border border-neural-600/30">
          <div className="text-lg font-semibold text-quantum-400">
            {thoughts.filter(t => t.state === 'active').length}
          </div>
          <div className="text-xs text-muted-foreground">Active Thoughts</div>
        </div>
        <div className="p-3 bg-neural-800/20 rounded-lg border border-neural-600/30">
          <div className="text-lg font-semibold text-green-400">
            {thoughts.filter(t => t.state === 'emerging').length}
          </div>
          <div className="text-xs text-muted-foreground">Emerging Ideas</div>
        </div>
        <div className="p-3 bg-neural-800/20 rounded-lg border border-neural-600/30">
          <div className="text-lg font-semibold text-neutral-400">
            {Math.round(thoughts.reduce((acc, t) => acc + t.intensity, 0) / thoughts.length * 100)}%
          </div>
          <div className="text-xs text-muted-foreground">Avg Intensity</div>
        </div>
      </div>
    </div>
  )
} 