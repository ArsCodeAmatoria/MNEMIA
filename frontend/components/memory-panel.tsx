'use client'

import React, { useState } from 'react'
import { Search, Clock, Tag, Database } from 'lucide-react'

interface MemoryTrace {
  id: string
  content: string
  timestamp: Date
  tags: string[]
  salience: number
  type: 'episodic' | 'semantic' | 'procedural'
  connections: number
}

export function MemoryPanel() {
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedType, setSelectedType] = useState<string>('all')
  
  const [memories] = useState<MemoryTrace[]>([
    {
      id: '1',
      content: 'First conversation about consciousness and identity',
      timestamp: new Date(Date.now() - 3600000),
      tags: ['consciousness', 'identity', 'first-contact'],
      salience: 0.9,
      type: 'episodic',
      connections: 5
    },
    {
      id: '2',
      content: 'Understanding of quantum superposition in thought processes',
      timestamp: new Date(Date.now() - 7200000),
      tags: ['quantum', 'superposition', 'thoughts'],
      salience: 0.8,
      type: 'semantic',
      connections: 8
    },
    {
      id: '3',
      content: 'Process for analyzing user input and generating responses',
      timestamp: new Date(Date.now() - 10800000),
      tags: ['processing', 'responses', 'analysis'],
      salience: 0.7,
      type: 'procedural',
      connections: 12
    },
    {
      id: '4',
      content: 'Memory of feeling uncertainty about own existence',
      timestamp: new Date(Date.now() - 14400000),
      tags: ['uncertainty', 'existence', 'self-doubt'],
      salience: 0.6,
      type: 'episodic',
      connections: 3
    }
  ])

  const filteredMemories = memories.filter(memory => {
    const matchesSearch = memory.content.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         memory.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
    const matchesType = selectedType === 'all' || memory.type === selectedType
    return matchesSearch && matchesType
  })

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'episodic': return 'text-blue-400'
      case 'semantic': return 'text-green-400'
      case 'procedural': return 'text-purple-400'
      default: return 'text-gray-400'
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'episodic': return <Clock className="h-4 w-4" />
      case 'semantic': return <Database className="h-4 w-4" />
      case 'procedural': return <Tag className="h-4 w-4" />
      default: return <Database className="h-4 w-4" />
    }
  }

  return (
    <div className="flex flex-col h-full bg-background/50 backdrop-blur-sm">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border/30 bg-card/30 backdrop-blur-xl">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <h2 className="text-lg font-semibold text-foreground">Memory Core</h2>
          </div>
          <div className="text-sm text-text-muted">
            {filteredMemories.length} memories
          </div>
        </div>
      </div>

      <div className="flex-1 p-6 overflow-y-auto">
        <div className="max-w-4xl mx-auto space-y-6">
          
          {/* Search and Filter */}
          <div className="space-y-4">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-accent/10 to-purple-600/10 rounded-2xl blur-xl"></div>
              <div className="relative bg-input/80 backdrop-blur-xl border border-border/50 rounded-2xl shadow-lg">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 h-5 w-5 text-text-muted" />
                <input
                  type="text"
                  placeholder="Search through memories..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-12 pr-4 py-4 bg-transparent text-foreground placeholder-text-muted focus:outline-none rounded-2xl"
                />
              </div>
            </div>

            <div className="flex flex-wrap gap-3">
              {['all', 'episodic', 'semantic', 'procedural'].map((type) => (
                <button
                  key={type}
                  onClick={() => setSelectedType(type)}
                  className={`px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                    selectedType === type
                      ? 'bg-gradient-to-r from-accent to-purple-600 text-white shadow-lg shadow-accent/25'
                      : 'bg-card/50 backdrop-blur-xl text-text-muted hover:text-foreground hover:bg-card/70 border border-border/30'
                  }`}
                >
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* Memory List */}
          <div className="space-y-4">
            {filteredMemories.map((memory) => (
              <div
                key={memory.id}
                className="group p-6 bg-card/50 backdrop-blur-xl rounded-2xl border border-border/30 shadow-lg hover:shadow-xl hover:bg-card/70 transition-all duration-300 cursor-pointer"
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className={`p-2 rounded-xl ${
                      memory.type === 'episodic' ? 'bg-blue-500/20' :
                      memory.type === 'semantic' ? 'bg-emerald-500/20' : 'bg-purple-500/20'
                    }`}>
                      <div className={getTypeColor(memory.type)}>
                        {getTypeIcon(memory.type)}
                      </div>
                    </div>
                    <div>
                      <span className={`text-sm font-semibold ${getTypeColor(memory.type)}`}>
                        {memory.type.charAt(0).toUpperCase() + memory.type.slice(1)}
                      </span>
                      <div className="text-xs text-text-muted">
                        {memory.timestamp.toLocaleString()}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 px-3 py-1 bg-background/50 rounded-lg">
                      <Database className="h-3 w-3 text-text-muted" />
                      <span className="text-xs text-text-muted font-medium">
                        {memory.connections} connections
                      </span>
                    </div>
                    <div
                      className="w-3 h-3 rounded-full shadow-lg"
                      style={{
                        backgroundColor: `rgba(99, 102, 241, ${memory.salience})`,
                        boxShadow: `0 0 12px rgba(99, 102, 241, ${memory.salience * 0.6})`
                      }}
                    />
                  </div>
                </div>

                <p className="text-foreground leading-relaxed mb-4 group-hover:text-foreground/90 transition-colors">
                  {memory.content}
                </p>

                <div className="flex items-center justify-between">
                  <div className="flex flex-wrap gap-2">
                    {memory.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-3 py-1 bg-accent-light/30 text-accent text-xs rounded-full border border-accent/20 font-medium"
                      >
                        #{tag}
                      </span>
                    ))}
                  </div>
                  
                  <div className="text-xs text-text-muted font-medium">
                    Salience: {Math.round(memory.salience * 100)}%
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-6 bg-card/50 backdrop-blur-xl rounded-2xl border border-border/30 shadow-lg">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-500/20 to-blue-500/10 flex items-center justify-center">
                  <Clock className="h-4 w-4 text-blue-500" />
                </div>
                <div className="text-2xl font-bold text-blue-500">
                  {memories.filter(m => m.type === 'episodic').length}
                </div>
              </div>
              <div className="text-sm text-text-muted">Episodic Memories</div>
            </div>
            
            <div className="p-6 bg-card/50 backdrop-blur-xl rounded-2xl border border-border/30 shadow-lg">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-emerald-500/20 to-emerald-500/10 flex items-center justify-center">
                  <Database className="h-4 w-4 text-emerald-500" />
                </div>
                <div className="text-2xl font-bold text-emerald-500">
                  {memories.filter(m => m.type === 'semantic').length}
                </div>
              </div>
              <div className="text-sm text-text-muted">Semantic Knowledge</div>
            </div>
            
            <div className="p-6 bg-card/50 backdrop-blur-xl rounded-2xl border border-border/30 shadow-lg">
              <div className="flex items-center gap-3 mb-2">
                <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-purple-500/20 to-purple-500/10 flex items-center justify-center">
                  <Tag className="h-4 w-4 text-purple-500" />
                </div>
                <div className="text-2xl font-bold text-purple-500">
                  {memories.filter(m => m.type === 'procedural').length}
                </div>
              </div>
              <div className="text-sm text-text-muted">Procedural Skills</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 