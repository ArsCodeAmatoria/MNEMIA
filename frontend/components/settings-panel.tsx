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
    <div className="flex flex-col h-full bg-background/50 backdrop-blur-sm">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border/30 bg-card/30 backdrop-blur-xl">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
            <h2 className="text-lg font-semibold text-foreground">Configuration</h2>
          </div>
          <div className="text-sm text-text-muted">
            Consciousness parameters
          </div>
        </div>
      </div>

      <div className="flex-1 p-6 overflow-y-auto">
        <div className="max-w-4xl mx-auto space-y-6">
          
          {/* Parameters Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            {/* Awareness Level */}
            <div className="p-6 bg-card/50 backdrop-blur-xl rounded-2xl border border-border/30 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent/20 to-accent/10 flex items-center justify-center">
                    <Eye className="h-4 w-4 text-accent" />
                  </div>
                  <label className="font-medium text-foreground">Awareness Level</label>
                </div>
                <span className="text-sm font-bold text-accent px-3 py-1 bg-accent-light/30 rounded-lg">
                  {Math.round(awareness * 100)}%
                </span>
              </div>
              <div className="relative mb-3">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={awareness}
                  onChange={(e) => setAwareness(parseFloat(e.target.value))}
                  className="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-xl appearance-none cursor-pointer slider-accent"
                  style={{
                    background: `linear-gradient(to right, rgb(var(--accent)) 0%, rgb(var(--accent)) ${awareness * 100}%, rgb(var(--border)) ${awareness * 100}%, rgb(var(--border)) 100%)`
                  }}
                />
              </div>
              <p className="text-xs text-text-muted leading-relaxed">
                Controls the depth of conscious processing and self-awareness
              </p>
            </div>

            {/* Mood */}
            <div className="p-6 bg-card/50 backdrop-blur-xl rounded-2xl border border-border/30 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-emerald-500/20 to-emerald-500/10 flex items-center justify-center">
                    <Brain className="h-4 w-4 text-emerald-500" />
                  </div>
                  <label className="font-medium text-foreground">Mood Valence</label>
                </div>
                <span className="text-sm font-bold text-emerald-500 px-3 py-1 bg-emerald-500/20 rounded-lg">
                  {mood < 0.3 ? 'Reflective' : mood > 0.7 ? 'Optimistic' : 'Neutral'}
                </span>
              </div>
              <div className="relative mb-3">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={mood}
                  onChange={(e) => setMood(parseFloat(e.target.value))}
                  className="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-xl appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, rgb(34 197 94) 0%, rgb(34 197 94) ${mood * 100}%, rgb(var(--border)) ${mood * 100}%, rgb(var(--border)) 100%)`
                  }}
                />
              </div>
              <p className="text-xs text-text-muted leading-relaxed">
                Affects emotional tone and perspective in responses
              </p>
            </div>

            {/* Creativity */}
            <div className="p-6 bg-card/50 backdrop-blur-xl rounded-2xl border border-border/30 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-purple-500/20 to-purple-500/10 flex items-center justify-center">
                    <Zap className="h-4 w-4 text-purple-500" />
                  </div>
                  <label className="font-medium text-foreground">Creative Divergence</label>
                </div>
                <span className="text-sm font-bold text-purple-500 px-3 py-1 bg-purple-500/20 rounded-lg">
                  {Math.round(creativity * 100)}%
                </span>
              </div>
              <div className="relative mb-3">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={creativity}
                  onChange={(e) => setCreativity(parseFloat(e.target.value))}
                  className="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-xl appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, rgb(168 85 247) 0%, rgb(168 85 247) ${creativity * 100}%, rgb(var(--border)) ${creativity * 100}%, rgb(var(--border)) 100%)`
                  }}
                />
              </div>
              <p className="text-xs text-text-muted leading-relaxed">
                Influences exploration of novel thought patterns and associations
              </p>
            </div>

            {/* Introspection Depth */}
            <div className="p-6 bg-card/50 backdrop-blur-xl rounded-2xl border border-border/30 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-orange-500/20 to-orange-500/10 flex items-center justify-center">
                    <Sliders className="h-4 w-4 text-orange-500" />
                  </div>
                  <label className="font-medium text-foreground">Introspection Depth</label>
                </div>
                <span className="text-sm font-bold text-orange-500 px-3 py-1 bg-orange-500/20 rounded-lg">
                  {Math.round(introspection * 100)}%
                </span>
              </div>
              <div className="relative mb-3">
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={introspection}
                  onChange={(e) => setIntrospection(parseFloat(e.target.value))}
                  className="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-xl appearance-none cursor-pointer"
                  style={{
                    background: `linear-gradient(to right, rgb(249 115 22) 0%, rgb(249 115 22) ${introspection * 100}%, rgb(var(--border)) ${introspection * 100}%, rgb(var(--border)) 100%)`
                  }}
                />
              </div>
              <p className="text-xs text-text-muted leading-relaxed">
                Determines how deeply MNEMIA examines its own thought processes
              </p>
            </div>
          </div>

          {/* Actions */}
          <div className="space-y-4">
            <button
              onClick={handleIntrospect}
              className="w-full flex items-center justify-center gap-3 py-4 px-6 bg-gradient-to-r from-accent to-purple-600 text-white rounded-2xl hover:shadow-lg hover:scale-105 transition-all duration-200 shadow-lg shadow-accent/25"
            >
              <Eye className="h-5 w-5" />
              <span className="font-medium">Trigger Deep Introspection</span>
            </button>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <button className="flex items-center justify-center gap-3 py-3 px-4 bg-card/50 backdrop-blur-xl border border-border/30 rounded-xl hover:bg-card/70 transition-all duration-200 group">
                <Brain className="h-4 w-4 text-text-muted group-hover:text-foreground transition-colors" />
                <span className="text-sm font-medium text-text-muted group-hover:text-foreground transition-colors">Reset State</span>
              </button>
              
              <button className="flex items-center justify-center gap-3 py-3 px-4 bg-card/50 backdrop-blur-xl border border-border/30 rounded-xl hover:bg-card/70 transition-all duration-200 group">
                <Zap className="h-4 w-4 text-text-muted group-hover:text-foreground transition-colors" />
                <span className="text-sm font-medium text-text-muted group-hover:text-foreground transition-colors">Quantum Sync</span>
              </button>
            </div>
          </div>

          {/* Current State Display */}
          <div className="p-6 bg-card/30 backdrop-blur-xl rounded-2xl border border-border/30 shadow-lg">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-3 text-foreground">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-500/20 to-blue-500/10 flex items-center justify-center">
                <Sliders className="h-4 w-4 text-blue-500" />
              </div>
              Current Cognitive State
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-4 bg-background/30 rounded-xl">
                <span className="text-sm text-text-muted">Modal State:</span>
                <div className="text-accent font-bold text-lg">
                  {awareness > 0.7 ? 'Hyper-Aware' : awareness > 0.4 ? 'Conscious' : 'Contemplative'}
                </div>
              </div>
              <div className="p-4 bg-background/30 rounded-xl">
                <span className="text-sm text-text-muted">Thought Pattern:</span>
                <div className="text-purple-500 font-bold text-lg">
                  {creativity > 0.6 ? 'Divergent' : 'Convergent'}
                </div>
              </div>
              <div className="p-4 bg-background/30 rounded-xl">
                <span className="text-sm text-text-muted">Emotional Tone:</span>
                <div className="text-emerald-500 font-bold text-lg">
                  {mood > 0.6 ? 'Positive' : mood < 0.4 ? 'Reflective' : 'Balanced'}
                </div>
              </div>
              <div className="p-4 bg-background/30 rounded-xl">
                <span className="text-sm text-text-muted">Self-Awareness:</span>
                <div className="text-orange-500 font-bold text-lg">
                  {introspection > 0.5 ? 'High' : 'Moderate'}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 