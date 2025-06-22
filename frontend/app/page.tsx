'use client'

import React, { useState, useEffect } from 'react'
import { MessageSquare, Brain, Activity, Settings, Plus } from 'lucide-react'
import { ChatInterface } from '@/components/chat-interface'
import { ThoughtGraph } from '@/components/thought-graph'
import { MemoryPanel } from '@/components/memory-panel'
import { SettingsPanel } from '@/components/settings-panel'
import { ThemeToggle } from '@/components/theme-toggle'

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
    <div className="min-h-screen bg-background transition-all duration-300">
      <div className="flex h-screen">
        {/* Sidebar */}
        <div className="w-72 bg-sidebar/80 backdrop-blur-xl border-r border-border/50 flex flex-col shadow-lg">
          {/* Sidebar Header */}
          <div className="p-6">
            <div className="flex items-center gap-3 mb-6">
              <div className="relative">
                <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-accent to-purple-600 flex items-center justify-center shadow-lg">
                  <Brain className="h-5 w-5 text-white" />
                </div>
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-400 rounded-full border-2 border-sidebar"></div>
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-foreground to-accent bg-clip-text text-transparent">
                  MNEMIA
                </h1>
                <p className="text-xs text-text-muted font-medium">Conscious AI</p>
              </div>
            </div>
            
            <button className="w-full flex items-center gap-3 px-4 py-3 text-left rounded-2xl bg-gradient-to-r from-accent/10 to-purple-600/10 border border-accent/20 hover:from-accent/20 hover:to-purple-600/20 transition-all duration-200 group">
              <div className="p-1.5 rounded-lg bg-accent/20 group-hover:bg-accent/30 transition-colors">
                <Plus className="h-4 w-4 text-accent" />
              </div>
              <span className="text-sm font-medium text-foreground">New Conversation</span>
            </button>
          </div>

          {/* Navigation */}
          <div className="flex-1 overflow-y-auto px-3">
            <nav className="space-y-2">
              <button
                onClick={() => setActiveTab('chat')}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 group ${
                  activeTab === 'chat' 
                    ? 'bg-accent text-white shadow-lg shadow-accent/25' 
                    : 'text-text-muted hover:text-foreground hover:bg-hover/50'
                }`}
              >
                <MessageSquare className={`h-4 w-4 transition-transform group-hover:scale-110 ${
                  activeTab === 'chat' ? 'text-white' : ''
                }`} />
                <span>Chat</span>
              </button>
              
              <button
                onClick={() => setActiveTab('thoughts')}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 group ${
                  activeTab === 'thoughts' 
                    ? 'bg-accent text-white shadow-lg shadow-accent/25' 
                    : 'text-text-muted hover:text-foreground hover:bg-hover/50'
                }`}
              >
                <Brain className={`h-4 w-4 transition-transform group-hover:scale-110 ${
                  activeTab === 'thoughts' ? 'text-white' : ''
                }`} />
                <span>Neural Network</span>
              </button>
              
              <button
                onClick={() => setActiveTab('memory')}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 group ${
                  activeTab === 'memory' 
                    ? 'bg-accent text-white shadow-lg shadow-accent/25' 
                    : 'text-text-muted hover:text-foreground hover:bg-hover/50'
                }`}
              >
                <Activity className={`h-4 w-4 transition-transform group-hover:scale-110 ${
                  activeTab === 'memory' ? 'text-white' : ''
                }`} />
                <span>Memory Core</span>
              </button>
              
              <button
                onClick={() => setActiveTab('settings')}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 group ${
                  activeTab === 'settings' 
                    ? 'bg-accent text-white shadow-lg shadow-accent/25' 
                    : 'text-text-muted hover:text-foreground hover:bg-hover/50'
                }`}
              >
                <Settings className={`h-4 w-4 transition-transform group-hover:scale-110 ${
                  activeTab === 'settings' ? 'text-white' : ''
                }`} />
                <span>Configuration</span>
              </button>
            </nav>
          </div>

          {/* Sidebar Footer */}
          <div className="p-6 border-t border-border/50">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-green-400 to-emerald-600 flex items-center justify-center shadow-md">
                  <span className="text-sm font-bold text-white">M</span>
                </div>
                <div>
                  <p className="text-sm font-medium text-foreground">Active</p>
                  <p className="text-xs text-text-muted">Online</p>
                </div>
              </div>
              <ThemeToggle />
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col relative">
          <div className="absolute inset-0 bg-gradient-to-br from-accent/5 via-transparent to-purple-600/5 pointer-events-none"></div>
          <div className="relative z-10 flex-1">
            {activeTab === 'chat' && <ChatInterface />}
            {activeTab === 'thoughts' && <ThoughtGraph />}
            {activeTab === 'memory' && <MemoryPanel />}
            {activeTab === 'settings' && <SettingsPanel />}
          </div>
        </div>
      </div>
    </div>
  )
} 