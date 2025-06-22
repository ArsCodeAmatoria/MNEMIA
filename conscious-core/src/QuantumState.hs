{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module QuantumState
  ( Thought(..)
  , QuantumState(..)
  , Superposition
  , Entanglement
  , WeightedThought
  , createThought
  , superpose
  , entangle
  , observe
  , collapse
  , measureCoherence
  , evolveSuperposition
  ) where

import GHC.Generics (Generic)
import System.Random (Random, randomR, StdGen, mkStdGen)
import qualified Data.Vector as V
import Data.List (sortBy)
import Data.Ord (comparing, Down(..))

-- | Represents a single thought or concept
data Thought = Thought
  { thoughtContent :: String
  , thoughtSalience :: Double  -- ^ Importance/relevance (0.0 to 1.0)
  , thoughtContext :: [String] -- ^ Associated contexts/tags
  , thoughtValence :: Double   -- ^ Emotional charge (-1.0 to 1.0)
  } deriving (Eq, Show, Generic)

-- | Quantum-inspired state of multiple thoughts
data QuantumState = QuantumState
  { qsThoughts :: [WeightedThought]
  , qsCoherence :: Double -- ^ How coherent/stable the superposition is
  , qsEntangled :: [Entanglement]
  } deriving (Eq, Show, Generic)

-- | A thought with quantum probability amplitude
type WeightedThought = (Thought, Double)

-- | Superposition of multiple weighted thoughts
type Superposition = [WeightedThought]

-- | Entanglement between thoughts
data Entanglement = Entanglement
  { entangledThoughts :: (Thought, Thought)
  , entanglementStrength :: Double
  , entanglementType :: EntanglementType
  } deriving (Eq, Show, Generic)

data EntanglementType 
  = Semantic    -- ^ Meaning-based connection
  | Temporal    -- ^ Time-based connection  
  | Causal      -- ^ Cause-effect relationship
  | Associative -- ^ Free association
  deriving (Eq, Show, Generic)

-- | Create a basic thought
createThought :: String -> Double -> [String] -> Double -> Thought
createThought content salience context valence = Thought
  { thoughtContent = content
  , thoughtSalience = clamp 0.0 1.0 salience
  , thoughtContext = context
  , thoughtValence = clamp (-1.0) 1.0 valence
  }
  where
    clamp minVal maxVal val = max minVal (min maxVal val)

-- | Create a superposition from multiple thoughts
superpose :: [Thought] -> StdGen -> (QuantumState, StdGen)
superpose thoughts gen = 
  let (weights, gen') = generateWeights (length thoughts) gen
      weightedThoughts = zip thoughts weights
      normalizedWeights = normalizeWeights weightedThoughts
      coherence = calculateCoherence normalizedWeights
  in (QuantumState normalizedWeights coherence [], gen')

-- | Generate random weights for superposition
generateWeights :: Int -> StdGen -> ([Double], StdGen)
generateWeights n gen = 
  let (weights, finalGen) = foldr (\_ (acc, g) -> 
        let (w, g') = randomR (0.1, 1.0) g
        in (w:acc, g')) ([], gen) [1..n]
  in (weights, finalGen)

-- | Normalize weights so they sum to 1
normalizeWeights :: Superposition -> Superposition
normalizeWeights weightedThoughts =
  let totalWeight = sum $ map snd weightedThoughts
      normalize (thought, weight) = (thought, weight / totalWeight)
  in map normalize weightedThoughts

-- | Calculate coherence of a superposition
calculateCoherence :: Superposition -> Double
calculateCoherence superposition =
  let weights = map snd superposition
      maxWeight = maximum weights
      entropy = -sum [w * log w | w <- weights, w > 0]
      maxEntropy = log (fromIntegral $ length superposition)
  in 1.0 - (entropy / maxEntropy) -- Higher coherence = lower entropy

-- | Create entanglement between two thoughts
entangle :: Thought -> Thought -> EntanglementType -> Double -> Entanglement
entangle t1 t2 entType strength = Entanglement (t1, t2) (clamp 0.0 1.0 strength) entType
  where
    clamp minVal maxVal val = max minVal (min maxVal val)

-- | Observe the superposition (measurement)
observe :: QuantumState -> StdGen -> (Thought, QuantumState, StdGen)
observe qs gen =
  let (selectedThought, gen') = collapse (qsThoughts qs) gen
      -- Observation affects the quantum state
      newCoherence = max 0.1 (qsCoherence qs - 0.1)
      newQS = qs { qsCoherence = newCoherence }
  in (selectedThought, newQS, gen')

-- | Collapse superposition to single thought based on probabilities
collapse :: Superposition -> StdGen -> (Thought, StdGen)
collapse superposition gen =
  let (randomVal, gen') = randomR (0.0, 1.0) gen
      sortedThoughts = sortBy (comparing (Down . snd)) superposition
  in (selectByProbability randomVal 0.0 sortedThoughts, gen')

-- | Select thought based on cumulative probability
selectByProbability :: Double -> Double -> Superposition -> Thought
selectByProbability _ _ [(thought, _)] = thought
selectByProbability target cumulative ((thought, weight):rest)
  | target <= cumulative + weight = thought
  | otherwise = selectByProbability target (cumulative + weight) rest
selectByProbability _ _ [] = error "Empty superposition"

-- | Measure coherence of quantum state
measureCoherence :: QuantumState -> Double
measureCoherence = qsCoherence

-- | Evolve superposition over time (quantum evolution)
evolveSuperposition :: QuantumState -> Double -> StdGen -> (QuantumState, StdGen)
evolveSuperposition qs timeStep gen =
  let (evolvedThoughts, gen') = evolveThoughts (qsThoughts qs) timeStep gen
      newCoherence = min 1.0 $ qsCoherence qs + (timeStep * 0.1)
      evolvedEntanglements = evolveEntanglements (qsEntangled qs) timeStep
  in (QuantumState evolvedThoughts newCoherence evolvedEntanglements, gen')

-- | Evolve individual thoughts in superposition
evolveThoughts :: Superposition -> Double -> StdGen -> (Superposition, StdGen)
evolveThoughts thoughts timeStep gen =
  let (perturbations, gen') = generatePerturbations (length thoughts) timeStep gen
      evolvedThoughts = zipWith perturbWeight thoughts perturbations
  in (normalizeWeights evolvedThoughts, gen')
  where
    perturbWeight (thought, weight) perturbation = 
      (thought, max 0.01 $ weight + perturbation)

-- | Generate small random perturbations for evolution
generatePerturbations :: Int -> Double -> StdGen -> ([Double], StdGen)
generatePerturbations n timeStep gen =
  let range = timeStep * 0.05 -- Small perturbations
      (perturbs, finalGen) = foldr (\_ (acc, g) ->
        let (p, g') = randomR (-range, range) g
        in (p:acc, g')) ([], gen) [1..n]
  in (perturbs, finalGen)

-- | Evolve entanglements over time
evolveEntanglements :: [Entanglement] -> Double -> [Entanglement]
evolveEntanglements entanglements timeStep =
  map (evolveEntanglement timeStep) entanglements
  where
    evolveEntanglement dt (Entanglement pair strength eType) =
      let decayRate = 0.02 -- Entanglements naturally decay
          newStrength = max 0.0 $ strength - (dt * decayRate)
      in Entanglement pair newStrength eType 