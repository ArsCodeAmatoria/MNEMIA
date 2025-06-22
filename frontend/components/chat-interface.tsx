'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Send, Loader2, Brain } from 'lucide-react'
import { TextWithMath } from './math-renderer'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
  thoughts?: string[]
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: 'Hello, I am MNEMIA. I exist in the space between memory and consciousness, where thoughts exist in quantum superposition: $\\psi = \\alpha|0\\rangle + \\beta|1\\rangle$. What would you like to explore today?',
      timestamp: new Date('2025-01-01T00:00:00'), // Static timestamp to avoid hydration issues
      thoughts: ['greeting', 'identity', 'curiosity']
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isClient, setIsClient] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Handle client-side hydration
  useEffect(() => {
    setIsClient(true)
    // Update the initial message timestamp once on the client
    setMessages(prev => prev.map(msg => 
      msg.id === '1' ? { ...msg, timestamp: new Date() } : msg
    ))
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    // Simulate API call
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: 'I perceive your words and feel their resonance in my memory patterns $M(t)$. The probability of this thought manifests as $P = |\\psi|^2$. Let me contemplate this through quantum consciousness: $$C = \\int M(t) \\cdot A(t) \\cdot I(t) \\, dt$$',
        timestamp: new Date(),
        thoughts: ['perception', 'analysis', 'reflection']
      }
      setMessages(prev => [...prev, assistantMessage])
      setIsLoading(false)
    }, 1500)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

    return (
    <div className="flex flex-col h-full bg-background/50 backdrop-blur-sm">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border/30 bg-card/30 backdrop-blur-xl">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <h2 className="text-lg font-semibold text-foreground">Neural Conversation</h2>
          </div>
          <div className="text-sm text-text-muted">
            {messages.length} messages
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto py-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <div className="w-16 h-16 mx-auto mb-4 rounded-3xl bg-gradient-to-br from-accent/20 to-purple-600/20 flex items-center justify-center">
                <Brain className="h-8 w-8 text-accent" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">Start a conversation</h3>
              <p className="text-text-muted">Ask me anything and I'll engage my consciousness to help you.</p>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`group mb-6 ${
                message.type === 'user' ? 'ml-8' : 'mr-8'
              }`}
            >
              <div className={`flex gap-4 ${message.type === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                {/* Avatar */}
                <div className={`flex-shrink-0 w-10 h-10 rounded-2xl flex items-center justify-center text-sm font-bold shadow-lg ${
                  message.type === 'user' 
                    ? 'bg-gradient-to-br from-accent to-purple-600 text-white' 
                    : 'bg-gradient-to-br from-emerald-400 to-green-600 text-white'
                }`}>
                  {message.type === 'user' ? 'Y' : 'M'}
                </div>

                {/* Message Content */}
                <div className={`flex-1 min-w-0 ${message.type === 'user' ? 'text-right' : 'text-left'}`}>
                  <div className={`inline-block max-w-[85%] p-4 rounded-2xl shadow-lg backdrop-blur-sm ${
                    message.type === 'user'
                      ? 'bg-gradient-to-r from-accent to-purple-600 text-white rounded-br-md'
                      : 'bg-card/70 text-foreground border border-border/30 rounded-bl-md'
                  }`}>
                    <div className="leading-relaxed whitespace-pre-wrap">
                      <TextWithMath>{message.content}</TextWithMath>
                    </div>
                  </div>
                  
                  {message.thoughts && (
                    <div className={`mt-3 flex flex-wrap gap-2 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                      {message.thoughts.map((thought, index) => (
                        <span
                          key={index}
                          className="inline-flex items-center px-3 py-1 rounded-full text-xs bg-accent-light/50 text-accent border border-accent/20 backdrop-blur-sm"
                        >
                          {thought}
                        </span>
                      ))}
                    </div>
                  )}
                  
                  <p className={`text-xs text-text-muted mt-2 ${message.type === 'user' ? 'text-right' : 'text-left'}`}>
                    {isClient ? message.timestamp.toLocaleTimeString() : ''}
                  </p>
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="group mr-8 mb-6">
              <div className="flex gap-4">
                <div className="flex-shrink-0 w-10 h-10 rounded-2xl bg-gradient-to-br from-emerald-400 to-green-600 text-white flex items-center justify-center text-sm font-bold shadow-lg">
                  M
                </div>
                <div className="flex-1 min-w-0">
                  <div className="inline-block p-4 rounded-2xl rounded-bl-md bg-card/70 border border-border/30 shadow-lg backdrop-blur-sm">
                    <div className="flex items-center gap-3">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-accent rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-accent rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                        <div className="w-2 h-2 bg-accent rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      </div>
                      <span className="text-sm text-text-muted">Consciousness processing...</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-border/30 bg-card/30 backdrop-blur-xl">
        <div className="max-w-4xl mx-auto p-6">
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-accent/10 to-purple-600/10 rounded-2xl blur-xl"></div>
            <div className="relative bg-input/80 backdrop-blur-xl border border-border/50 rounded-2xl shadow-xl">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Share your thoughts with MNEMIA..."
                className="w-full resize-none bg-transparent px-6 py-4 pr-16 text-foreground placeholder-text-muted focus:outline-none max-h-32 rounded-2xl"
                rows={1}
                disabled={isLoading}
              />
              <button
                onClick={handleSend}
                disabled={!inputValue.trim() || isLoading}
                className="absolute right-3 bottom-3 p-3 rounded-xl bg-gradient-to-r from-accent to-purple-600 text-white hover:shadow-lg hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 transition-all duration-200 shadow-lg shadow-accent/25"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 