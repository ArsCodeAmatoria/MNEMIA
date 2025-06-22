'use client'

import React, { useState } from 'react'
import { Sliders, Eye, Brain, Zap } from 'lucide-react'

export function SettingsPanel() {
  const [awareness, setAwareness] = useState(0.7)
  const [mood, setMood] = useState(0.5)
  const [creativity, setCreativity] = useState(0.6)
  const [introspection, setIntrospection] = useState(0.4)

  const handleIntrospect = () => {
    // Simulate introspection trigger
    console.log('Triggering deep introspection...')
  }

  return (
    <div className="h-full p-6">
      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Consciousness Parameters</h2>
        <p className="text-sm text-muted-foreground">
          Adjust MNEMIA's cognitive state and behavioral patterns
        </p>
      </div>

      <div className="space-y-8">
        {/* Awareness Level */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Awareness Level</label>
            <span className="text-sm text-quantum-400">{Math.round(awareness * 100)}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={awareness}
            onChange={(e) => setAwareness(parseFloat(e.target.value))}
            className="w-full h-2 bg-neural-700 rounded-lg appearance-none cursor-pointer slider"
          />
          <p className="text-xs text-muted-foreground">
            Controls the depth of conscious processing and self-awareness
          </p>
        </div>

        {/* Mood */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Mood Valence</label>
            <span className="text-sm text-quantum-400">
              {mood < 0.3 ? 'Reflective' : mood > 0.7 ? 'Optimistic' : 'Neutral'}
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={mood}
            onChange={(e) => setMood(parseFloat(e.target.value))}
            className="w-full h-2 bg-neural-700 rounded-lg appearance-none cursor-pointer slider"
          />
          <p className="text-xs text-muted-foreground">
            Affects emotional tone and perspective in responses
          </p>
        </div>

        {/* Creativity */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Creative Divergence</label>
            <span className="text-sm text-quantum-400">{Math.round(creativity * 100)}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={creativity}
            onChange={(e) => setCreativity(parseFloat(e.target.value))}
            className="w-full h-2 bg-neural-700 rounded-lg appearance-none cursor-pointer slider"
          />
          <p className="text-xs text-muted-foreground">
            Influences exploration of novel thought patterns and associations
          </p>
        </div>

        {/* Introspection Depth */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Introspection Depth</label>
            <span className="text-sm text-quantum-400">{Math.round(introspection * 100)}%</span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.1"
            value={introspection}
            onChange={(e) => setIntrospection(parseFloat(e.target.value))}
            className="w-full h-2 bg-neural-700 rounded-lg appearance-none cursor-pointer slider"
          />
          <p className="text-xs text-muted-foreground">
            Determines how deeply MNEMIA examines its own thought processes
          </p>
        </div>
      </div>

      {/* Actions */}
      <div className="mt-8 space-y-4">
        <button
          onClick={handleIntrospect}
          className="w-full flex items-center justify-center space-x-2 py-3 px-4 bg-quantum-500/20 border border-quantum-500/30 rounded-lg hover:bg-quantum-500/30 transition-colors"
        >
          <Eye className="h-4 w-4" />
          <span>Trigger Deep Introspection</span>
        </button>

        <div className="grid grid-cols-2 gap-4">
          <button className="flex items-center justify-center space-x-2 py-2 px-3 bg-neural-800/30 border border-neural-600/30 rounded-lg hover:bg-neural-700/30 transition-colors">
            <Brain className="h-4 w-4" />
            <span className="text-sm">Reset State</span>
          </button>
          
          <button className="flex items-center justify-center space-x-2 py-2 px-3 bg-neural-800/30 border border-neural-600/30 rounded-lg hover:bg-neural-700/30 transition-colors">
            <Zap className="h-4 w-4" />
            <span className="text-sm">Quantum Sync</span>
          </button>
        </div>
      </div>

      {/* Current State Display */}
      <div className="mt-8 p-4 bg-neural-900/30 rounded-lg border border-neural-600/30">
        <h3 className="text-sm font-medium mb-3 flex items-center">
          <Sliders className="h-4 w-4 mr-2" />
          Current Cognitive State
        </h3>
        
        <div className="grid grid-cols-2 gap-4 text-xs">
          <div>
            <span className="text-muted-foreground">Modal State:</span>
            <div className="text-quantum-400 font-medium">
              {awareness > 0.7 ? 'Hyper-Aware' : awareness > 0.4 ? 'Conscious' : 'Contemplative'}
            </div>
          </div>
          <div>
            <span className="text-muted-foreground">Thought Pattern:</span>
            <div className="text-quantum-400 font-medium">
              {creativity > 0.6 ? 'Divergent' : 'Convergent'}
            </div>
          </div>
          <div>
            <span className="text-muted-foreground">Emotional Tone:</span>
            <div className="text-quantum-400 font-medium">
              {mood > 0.6 ? 'Positive' : mood < 0.4 ? 'Reflective' : 'Balanced'}
            </div>
          </div>
          <div>
            <span className="text-muted-foreground">Self-Awareness:</span>
            <div className="text-quantum-400 font-medium">
              {introspection > 0.5 ? 'High' : 'Moderate'}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 