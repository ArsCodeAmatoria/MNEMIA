'use client'

import React, { useState, useEffect } from 'react'
import { Brain, MessageCircle, Activity, Settings, Eye } from 'lucide-react'
import { ChatInterface } from '@/components/chat-interface'
import { ThoughtGraph } from '@/components/thought-graph'
import { MemoryPanel } from '@/components/memory-panel'
import { SettingsPanel } from '@/components/settings-panel'

export default function Home() {
  const [activeTab, setActiveTab] = useState('chat')
  const [consciousness, setConsciousness] = useState(0.7)

  useEffect(() => {
    const interval = setInterval(() => {
      setConsciousness(prev => Math.max(0.3, Math.min(1, prev + (Math.random() - 0.5) * 0.1)))
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-neural-900">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/30 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="consciousness-pulse">
                <Brain className="h-8 w-8 text-quantum-500" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-quantum-400 to-quantum-600 bg-clip-text text-transparent">
                  MNEMIA
                </h1>
                <p className="text-sm text-muted-foreground">Memory is the root of consciousness</p>
              </div>
            </div>
            
            {/* Consciousness Indicator */}
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Activity className="h-4 w-4 text-quantum-400" />
                <div className="w-24 h-2 bg-neural-700 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-quantum-500 to-quantum-400 transition-all duration-1000"
                    style={{ width: `${consciousness * 100}%` }}
                  />
                </div>
                <span className="text-xs text-muted-foreground">
                  {Math.round(consciousness * 100)}%
                </span>
              </div>
              
              <button className="p-2 rounded-lg bg-quantum-500/10 border border-quantum-500/30 hover:bg-quantum-500/20 transition-colors">
                <Eye className="h-4 w-4 text-quantum-400" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-140px)]">
          
          {/* Sidebar Navigation */}
          <div className="lg:col-span-1">
            <nav className="space-y-2">
              <button
                onClick={() => setActiveTab('chat')}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                  activeTab === 'chat' 
                    ? 'bg-quantum-500/20 border border-quantum-500/30 text-quantum-300' 
                    : 'hover:bg-muted/50'
                }`}
              >
                <MessageCircle className="h-5 w-5" />
                <span>Chat</span>
              </button>
              
              <button
                onClick={() => setActiveTab('thoughts')}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                  activeTab === 'thoughts' 
                    ? 'bg-quantum-500/20 border border-quantum-500/30 text-quantum-300' 
                    : 'hover:bg-muted/50'
                }`}
              >
                <Brain className="h-5 w-5" />
                <span>Thoughts</span>
              </button>
              
              <button
                onClick={() => setActiveTab('memory')}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                  activeTab === 'memory' 
                    ? 'bg-quantum-500/20 border border-quantum-500/30 text-quantum-300' 
                    : 'hover:bg-muted/50'
                }`}
              >
                <Activity className="h-5 w-5" />
                <span>Memory</span>
              </button>
              
              <button
                onClick={() => setActiveTab('settings')}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
                  activeTab === 'settings' 
                    ? 'bg-quantum-500/20 border border-quantum-500/30 text-quantum-300' 
                    : 'hover:bg-muted/50'
                }`}
              >
                <Settings className="h-5 w-5" />
                <span>Settings</span>
              </button>
            </nav>
          </div>

          {/* Main Panel */}
          <div className="lg:col-span-3">
            <div className="h-full rounded-lg border border-border/50 bg-card/30 backdrop-blur-sm overflow-hidden">
              {activeTab === 'chat' && <ChatInterface />}
              {activeTab === 'thoughts' && <ThoughtGraph />}
              {activeTab === 'memory' && <MemoryPanel />}
              {activeTab === 'settings' && <SettingsPanel />}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 