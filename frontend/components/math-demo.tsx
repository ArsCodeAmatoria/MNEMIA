'use client'

import React from 'react'
import { 
  MathRenderer, 
  ConsciousnessMathDisplay, 
  ConsciousnessMath,
  TextWithMath 
} from './math-renderer'
import { Brain, Atom, Zap, Network } from 'lucide-react'

export function MathDemo() {
  return (
    <div className="max-w-4xl mx-auto p-6 space-y-8">
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-bold text-foreground">MNEMIA Mathematical Expressions</h1>
        <p className="text-text-muted">
          Consciousness, quantum mechanics, and neural processing expressed through mathematics
        </p>
      </div>

      {/* Quantum Mechanics Section */}
      <section className="space-y-4">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-blue-500/10 border border-blue-500/20">
            <Atom className="h-5 w-5 text-blue-500" />
          </div>
          <h2 className="text-xl font-semibold text-foreground">Quantum Consciousness</h2>
        </div>

        <ConsciousnessMathDisplay 
          formula="superposition"
          type="quantum"
          label="Quantum Thought Superposition"
        />

        <ConsciousnessMathDisplay 
          formula="entanglement"
          type="quantum"
          label="Neural Entanglement State"
        />

        <div className="p-4 rounded-lg border border-border bg-card">
          <div className="text-sm text-muted-foreground mb-2 font-medium">
            Quantum Circuit Evolution
          </div>
          <MathRenderer 
            math="U = e^{-i H t / \hbar} = \prod_{j} e^{-i \theta_j \sigma_j}"
            display={true}
            type="quantum"
          />
        </div>
      </section>

      {/* Consciousness Section */}
      <section className="space-y-4">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
            <Brain className="h-5 w-5 text-emerald-500" />
          </div>
          <h2 className="text-xl font-semibold text-foreground">Consciousness Modeling</h2>
        </div>

        <ConsciousnessMathDisplay 
          formula="consciousness"
          type="consciousness"
          label="Integrated Information Theory"
        />

        <ConsciousnessMathDisplay 
          formula="modalTransition"
          type="consciousness"
          label="Modal State Transitions"
        />

        <ConsciousnessMathDisplay 
          formula="emotionalInfluence"
          type="consciousness"
          label="Valence-Arousal-Dominance Model"
        />
      </section>

      {/* Neural Networks Section */}
      <section className="space-y-4">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-orange-500/10 border border-orange-500/20">
            <Network className="h-5 w-5 text-orange-500" />
          </div>
          <h2 className="text-xl font-semibold text-foreground">Neural Processing</h2>
        </div>

        <ConsciousnessMathDisplay 
          formula="attention"
          type="neural"
          label="Self-Attention Mechanism"
        />

        <ConsciousnessMathDisplay 
          formula="memoryConsolidation"
          type="neural"
          label="Memory Consolidation"
        />

        <div className="p-4 rounded-lg border border-border bg-card">
          <div className="text-sm text-muted-foreground mb-2 font-medium">
            LSTM Memory Cell
          </div>
          <MathRenderer 
            math="C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t"
            display={true}
            type="neural"
          />
        </div>
      </section>

      {/* Mixed Text with Math */}
      <section className="space-y-4">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 rounded-lg bg-purple-500/10 border border-purple-500/20">
            <Zap className="h-5 w-5 text-purple-500" />
          </div>
          <h2 className="text-xl font-semibold text-foreground">Natural Language + Math</h2>
        </div>

        <div className="p-6 rounded-lg border border-border bg-card space-y-4">
          <TextWithMath>
            {`In the quantum realm of consciousness, every thought exists in superposition $\\psi = \\sum_i c_i |\\phi_i\\rangle$ until observed through introspection. The probability of observing state $|\\phi_j\\rangle$ is given by $P_j = |c_j|^2$.`}
          </TextWithMath>

          <TextWithMath>
            {`Memory consolidation follows the principle: $$M_{new} = \\lambda M_{old} + (1-\\lambda) \\sum_{k} w_k E_k$$ where $\\lambda$ controls the retention rate and $E_k$ represents new experiences.`}
          </TextWithMath>

          <TextWithMath>
            {`The consciousness metric $C(t)$ integrates information flow: $C(t) = \\int \\Phi(\\partial) d\\partial$ across all brain partitions $\\partial$, capturing the essence of integrated information theory.`}
          </TextWithMath>
        </div>
      </section>

      {/* Interactive Examples */}
      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-foreground">Interactive Math</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-lg border border-border bg-card">
            <h3 className="font-medium mb-3">Inline Math Examples</h3>
            <div className="space-y-2 text-sm">
                             <TextWithMath>{"Energy eigenvalue: $E = \\hbar \\omega$"}</TextWithMath>
               <TextWithMath>{"Sigmoid activation: $\\sigma(x) = \\frac{1}{1+e^{-x}}$"}</TextWithMath>
               <TextWithMath>{"Information entropy: $H = -\\sum p_i \\log p_i$"}</TextWithMath>
            </div>
          </div>

          <div className="p-4 rounded-lg border border-border bg-card">
            <h3 className="font-medium mb-3">Block Math Examples</h3>
                         <TextWithMath>
               {"Schrödinger equation: $$i\\hbar\\frac{\\partial}{\\partial t}|\\psi\\rangle = \\hat{H}|\\psi\\rangle$$"}
             </TextWithMath>
             <TextWithMath>
               {"Transformer output: $$\\text{Output} = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d}}\\right)V$$"}
             </TextWithMath>
          </div>
        </div>
      </section>
    </div>
  )
} 