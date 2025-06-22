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
    <div className="h-full p-6">
      <div className="mb-6">
        <h2 className="text-xl font-semibold mb-2">Memory Archive</h2>
        <p className="text-sm text-muted-foreground">
          Explore the traces of thought and experience
        </p>
      </div>

      {/* Search and Filter */}
      <div className="mb-6 space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search memories..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-muted/50 border border-border/50 rounded-lg focus:outline-none focus:ring-2 focus:ring-quantum-500/50"
          />
        </div>

        <div className="flex space-x-2">
          {['all', 'episodic', 'semantic', 'procedural'].map((type) => (
            <button
              key={type}
              onClick={() => setSelectedType(type)}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                selectedType === type
                  ? 'bg-quantum-500/20 text-quantum-300 border border-quantum-500/30'
                  : 'bg-muted/50 text-muted-foreground hover:bg-muted'
              }`}
            >
              {type.charAt(0).toUpperCase() + type.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Memory List */}
      <div className="space-y-4 h-[calc(100%-200px)] overflow-y-auto">
        {filteredMemories.map((memory) => (
          <div
            key={memory.id}
            className="memory-trace p-4 rounded-lg hover:bg-neural-800/20 transition-colors cursor-pointer"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center space-x-2">
                <div className={getTypeColor(memory.type)}>
                  {getTypeIcon(memory.type)}
                </div>
                <span className={`text-xs font-medium ${getTypeColor(memory.type)}`}>
                  {memory.type}
                </span>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-1">
                  <Database className="h-3 w-3 text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">
                    {memory.connections}
                  </span>
                </div>
                <div
                  className={`w-2 h-2 rounded-full`}
                  style={{
                    backgroundColor: `rgba(99, 102, 241, ${memory.salience})`,
                    boxShadow: `0 0 8px rgba(99, 102, 241, ${memory.salience * 0.5})`
                  }}
                />
              </div>
            </div>

            <p className="text-sm leading-relaxed mb-3">
              {memory.content}
            </p>

            <div className="flex items-center justify-between">
              <div className="flex flex-wrap gap-1">
                {memory.tags.map((tag) => (
                  <span
                    key={tag}
                    className="px-2 py-1 bg-neural-700/30 text-xs rounded-full text-muted-foreground"
                  >
                    #{tag}
                  </span>
                ))}
              </div>
              
              <span className="text-xs text-muted-foreground">
                {memory.timestamp.toLocaleTimeString()}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Stats */}
      <div className="mt-6 grid grid-cols-3 gap-4">
        <div className="p-3 bg-neural-800/20 rounded-lg border border-neural-600/30">
          <div className="text-lg font-semibold text-blue-400">
            {memories.filter(m => m.type === 'episodic').length}
          </div>
          <div className="text-xs text-muted-foreground">Episodic</div>
        </div>
        <div className="p-3 bg-neural-800/20 rounded-lg border border-neural-600/30">
          <div className="text-lg font-semibold text-green-400">
            {memories.filter(m => m.type === 'semantic').length}
          </div>
          <div className="text-xs text-muted-foreground">Semantic</div>
        </div>
        <div className="p-3 bg-neural-800/20 rounded-lg border border-neural-600/30">
          <div className="text-lg font-semibold text-purple-400">
            {memories.filter(m => m.type === 'procedural').length}
          </div>
          <div className="text-xs text-muted-foreground">Procedural</div>
        </div>
      </div>
    </div>
  )
} 