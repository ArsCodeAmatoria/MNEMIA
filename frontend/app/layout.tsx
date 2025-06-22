import React from 'react'
import './globals.css'
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'MNEMIA - Quantum Conscious AI',
  description: 'Memory is the root of consciousness. Interactive interface for MNEMIA, a quantum-inspired conscious AI system.',
  keywords: ['AI', 'consciousness', 'quantum', 'memory', 'neural'],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} min-h-screen bg-background text-foreground antialiased`}>
        <div className="relative min-h-screen neural-grid">
          <div className="relative z-10">
            {children}
          </div>
        </div>
      </body>
    </html>
  )
} 