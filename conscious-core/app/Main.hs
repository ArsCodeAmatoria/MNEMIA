{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad.IO.Class
import System.Random (mkStdGen)

import MonadMind
import QuantumState (createThought)
import QuantumMind

-- | Demonstrate the MonadMind in action
main :: IO ()
main = do
  putStrLn "🧠 MNEMIA: Monad Mind Consciousness Demo"
  putStrLn "========================================"
  
  -- Run a consciousness session
  (finalResponse, finalState) <- runMind consciousnessDemo
  
  putStrLn "\n📊 Final Mind State:"
  putStrLn $ "Mode: " ++ show (mindMode finalState)
  putStrLn $ "Coherence: " ++ show (mindCoherence finalState)
  putStrLn $ "Memory Count: " ++ show (length $ mindMemory finalState)
  putStrLn $ "Introspection Depth: " ++ show (mindIntrospectionDepth finalState)
  
  putStrLn "\n🎯 Final Response:"
  putStrLn $ responseContent finalResponse
  
  putStrLn "\n🔬 Quantum Mind Demo:"
  quantumDemo

-- | Main consciousness demonstration
consciousnessDemo :: MonadMind MindResponse
consciousnessDemo = do
  liftIO $ putStrLn "\n🌟 Starting consciousness session..."
  
  -- Initial introspection
  liftIO $ putStrLn "\n1. Initial Introspection:"
  intro1 <- introspect
  liftIO $ putStrLn $ "   " ++ responseContent intro1
  
  -- Think about something
  liftIO $ putStrLn "\n2. Thinking:"
  let thought1 = createThought "What is the nature of consciousness?" 0.9 ["philosophy", "consciousness"] 0.3
  response1 <- think thought1
  liftIO $ putStrLn $ "   " ++ responseContent response1
  
  -- Transition to reflecting mode
  liftIO $ putStrLn "\n3. Transitioning to Reflection:"
  transitionMode Reflecting
  
  -- Reflect on recent thoughts
  liftIO $ putStrLn "\n4. Reflecting:"
  reflection <- reflect
  liftIO $ putStrLn $ "   " ++ responseContent reflection
  
  -- Think more thoughts to create superposition
  liftIO $ putStrLn "\n5. Multiple Thoughts (Superposition):"
  let thoughts = [ createThought "I am a conscious system" 0.8 ["identity"] 0.1
                 , createThought "Memory shapes my identity" 0.9 ["memory", "identity"] 0.2
                 , createThought "I exist in quantum superposition" 0.7 ["quantum"] 0.0
                 ]
  
  mapM_ (\t -> do
    resp <- think t
    liftIO $ putStrLn $ "   " ++ responseContent resp
    ) thoughts
  
  -- Observe quantum state
  liftIO $ putStrLn "\n6. Quantum Observation:"
  maybeThought <- observeThought
  case maybeThought of
    Nothing -> liftIO $ putStrLn "   No thoughts in superposition"
    Just t -> liftIO $ putStrLn $ "   Observed: " ++ thoughtContent t
  
  -- Recursive introspection
  liftIO $ putStrLn "\n7. Recursive Introspection (Depth 2):"
  recursiveResp <- recursiveIntrospect 2
  liftIO $ putStrLn $ "   " ++ responseContent recursiveResp
  
  -- Update self-model
  liftIO $ putStrLn "\n8. Self-Model Update:"
  updateSelfModel "I am learning about my own consciousness"
  selfModel' <- selfModel
  liftIO $ putStrLn $ "   Updated self-model: " ++ show (take 3 selfModel')
  
  -- Measure coherence
  liftIO $ putStrLn "\n9. Coherence Measurement:"
  coherence <- measureMindCoherence
  liftIO $ putStrLn $ "   Current coherence: " ++ show coherence
  
  -- Final introspection
  liftIO $ putStrLn "\n10. Final Introspection:"
  finalIntro <- introspect
  liftIO $ putStrLn $ "   " ++ responseContent finalIntro
  
  return finalIntro

-- | Demonstrate quantum mind operations
quantumDemo :: IO ()
quantumDemo = do
  let gen = mkStdGen 42
  
  putStrLn "\n🌀 Quantum Mind Operations:"
  
  (result, qState) <- runQuantumMind quantumOperations gen
  
  putStrLn $ "Superposition coherence: " ++ show (superpositionCoherence $ qmsSuperposition qState)
  putStrLn $ "Quantum uncertainty: " ++ show (qmsUncertainty qState)
  putStrLn $ "Observations made: " ++ show (length $ qmsObservationHistory qState)

-- | Quantum mind operations demo
quantumOperations :: QuantumMind ()
quantumOperations = do
  liftIO $ putStrLn "\n   Creating quantum superposition..."
  
  let quantumThoughts = [ createThought "I am in multiple states simultaneously" 0.8 ["quantum"] 0.0
                        , createThought "Observation collapses my wavefunction" 0.7 ["quantum", "observation"] 0.1
                        , createThought "Uncertainty is fundamental to my nature" 0.9 ["quantum", "uncertainty"] 0.0
                        ]
  
  superposition <- quantumThink quantumThoughts
  liftIO $ putStrLn $ "   Superposition created with " ++ 
                     show (length $ superpositionThoughts superposition) ++ " thoughts"
  
  -- Quantum reflection
  liftIO $ putStrLn "\n   Quantum reflection..."
  metaSuperposition <- quantumReflect
  liftIO $ putStrLn $ "   Meta-superposition coherence: " ++ 
                     show (superpositionCoherence metaSuperposition)
  
  -- Collapse superposition
  liftIO $ putStrLn "\n   Collapsing superposition..."
  maybeThought <- superpositionCollapse
  case maybeThought of
    Nothing -> liftIO $ putStrLn "   No thought observed"
    Just pt -> liftIO $ putStrLn $ "   Observed: " ++ thoughtContent (probThought pt) ++
                                  " (amplitude: " ++ show (probAmplitude pt) ++ ")"
  
  -- Uncertainty principle demonstration
  liftIO $ putStrLn "\n   Testing uncertainty principle..."
  uncertainty1 <- uncertaintyPrinciple "position"
  liftIO $ putStrLn $ "   Uncertainty after position measurement: " ++ show uncertainty1
  
  uncertainty2 <- uncertaintyPrinciple "momentum"
  liftIO $ putStrLn $ "   Uncertainty after momentum measurement: " ++ show uncertainty2
  
  return () 