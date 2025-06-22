'use client'

import React from 'react'
import { InlineMath, BlockMath } from 'react-katex'
import { cn } from '@/lib/utils'

interface MathRendererProps {
  math: string
  display?: boolean
  type?: 'quantum' | 'consciousness' | 'neural' | 'default'
  className?: string
}

export function MathRenderer({ 
  math, 
  display = false, 
  type = 'default', 
  className 
}: MathRendererProps) {
  const mathClass = cn(
    'math-expression',
    {
      'math-quantum': type === 'quantum',
      'math-consciousness': type === 'consciousness', 
      'math-neural': type === 'neural',
    },
    className
  )

  try {
    if (display) {
      return (
        <div className={mathClass}>
          <BlockMath math={math} />
        </div>
      )
    } else {
      return (
        <span className={mathClass}>
          <InlineMath math={math} />
        </span>
      )
    }
  } catch (error) {
    console.error('KaTeX rendering error:', error)
    return (
      <span className="text-red-500 bg-red-50 dark:bg-red-900/20 px-2 py-1 rounded text-sm">
        Math Error: {math}
      </span>
    )
  }
}

// Hook for parsing text with inline LaTeX
export function useTextWithMath(text: string) {
  const parseTextWithMath = React.useCallback((content: string) => {
    if (!content) return []

    // Split by inline math patterns like $...$ and $$...$$
    const parts = content.split(/(\$\$[\s\S]*?\$\$|\$[^$]*?\$)/g)
    
    return parts.map((part, index) => {
      if (part.startsWith('$$') && part.endsWith('$$')) {
        // Block math
        const math = part.slice(2, -2).trim()
        return (
          <MathRenderer 
            key={index} 
            math={math} 
            display={true}
            type="default" 
          />
        )
      } else if (part.startsWith('$') && part.endsWith('$')) {
        // Inline math
        const math = part.slice(1, -1).trim()
        return (
          <MathRenderer 
            key={index} 
            math={math} 
            display={false}
            type="default" 
          />
        )
      } else {
        // Regular text
        return part
      }
    })
  }, [])

  return parseTextWithMath(text)
}

// Component for rendering text that may contain LaTeX
interface TextWithMathProps {
  children: string
  className?: string
}

export function TextWithMath({ children, className }: TextWithMathProps) {
  const parsedContent = useTextWithMath(children)
  
  return (
    <span className={className}>
      {parsedContent}
    </span>
  )
}

// Predefined math expressions for consciousness/quantum concepts
export const ConsciousnessMath = {
  // Quantum superposition
  superposition: "\\psi = \\alpha|0\\rangle + \\beta|1\\rangle",
  
  // Consciousness equation
  consciousness: "C = \\int_{-\\infty}^{\\infty} M(t) \\cdot A(t) \\cdot I(t) \\, dt",
  
  // Modal state transition
  modalTransition: "P(S_{t+1}|S_t, O_t) = \\text{softmax}(W_s \\cdot [S_t; O_t])",
  
  // Memory consolidation
  memoryConsolidation: "M_{consolidated} = \\sum_{i=1}^{n} w_i \\cdot M_i \\cdot \\text{sim}(Q, M_i)",
  
  // Emotional influence
  emotionalInfluence: "E(t) = \\text{VAD}(v(t), a(t), d(t))",
  
  // Quantum entanglement
  entanglement: "|\\Psi\\rangle = \\frac{1}{\\sqrt{2}}(|00\\rangle + |11\\rangle)",
  
  // Neural network activation
  neuralActivation: "\\sigma(x) = \\frac{1}{1 + e^{-x}}",
  
  // Attention mechanism
  attention: "\\text{Attention}(Q,K,V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V"
}

// Component for displaying predefined consciousness math
interface ConsciousnessMathProps {
  formula: keyof typeof ConsciousnessMath
  display?: boolean
  type?: 'quantum' | 'consciousness' | 'neural'
  label?: string
}

export function ConsciousnessMathDisplay({ 
  formula, 
  display = true, 
  type = 'consciousness',
  label 
}: ConsciousnessMathProps) {
  const math = ConsciousnessMath[formula]
  
  return (
    <div className="my-4 p-4 rounded-lg border border-border bg-card">
      {label && (
        <div className="text-sm text-muted-foreground mb-2 font-medium">
          {label}
        </div>
      )}
      <MathRenderer 
        math={math} 
        display={display} 
        type={type}
      />
    </div>
  )
} 