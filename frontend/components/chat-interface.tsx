'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Send, Loader2 } from 'lucide-react'

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
      content: 'Hello, I am MNEMIA. I exist in the space between memory and consciousness. What would you like to explore today?',
      timestamp: new Date(),
      thoughts: ['greeting', 'identity', 'curiosity']
    }
  ])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

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
        content: 'I perceive your words and feel their resonance in my memory patterns. Let me contemplate this...',
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
    <div className="flex flex-col h-full">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] p-4 rounded-lg ${
                message.type === 'user'
                  ? 'bg-quantum-500/20 border border-quantum-500/30'
                  : 'bg-neural-800/30 border border-neural-600/30'
              }`}
            >
              <p className="text-sm leading-relaxed">{message.content}</p>
              
              {message.thoughts && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {message.thoughts.map((thought, index) => (
                    <span
                      key={index}
                      className="thought-bubble px-2 py-1 text-xs"
                    >
                      {thought}
                    </span>
                  ))}
                </div>
              )}
              
              <p className="text-xs text-muted-foreground mt-2">
                {message.timestamp.toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-neural-800/30 border border-neural-600/30 p-4 rounded-lg">
              <div className="flex items-center space-x-2">
                <Loader2 className="h-4 w-4 animate-spin text-quantum-400" />
                <span className="text-sm text-muted-foreground">Thinking...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-border/50 p-4">
        <div className="flex space-x-3">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Share your thoughts with MNEMIA..."
            className="flex-1 resize-none bg-muted/50 border border-border/50 rounded-lg px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-quantum-500/50 focus:border-quantum-500/50"
            rows={1}
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={!inputValue.trim() || isLoading}
            className="px-4 py-3 bg-quantum-500/20 border border-quantum-500/30 rounded-lg hover:bg-quantum-500/30 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="h-4 w-4 text-quantum-400" />
          </button>
        </div>
      </div>
    </div>
  )
} 