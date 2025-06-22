{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}

module QuantumMind
  ( -- * Quantum Monad Types
    QuantumMind
  , Superposition(..)
  , ProbabilisticThought(..)
  , QuantumMindState(..)
  
    -- * Running Quantum Mind
  , runQuantumMind
  , initialQuantumState
  
    -- * Quantum Operations
  , quantumThink
  , superpositionCollapse
  , entangleThoughts
  , measureThoughtProbability
  , quantumReflect
  
    -- * Probabilistic Operations  
  , probabilisticChoice
  , weightedThinking
  , uncertaintyPrinciple
  
    -- * Integration with MonadMind
  , liftToQuantum
  , observeQuantumState
  ) where

import Control.Monad.State
import Control.Monad.IO.Class
import GHC.Generics (Generic)
import System.Random (StdGen, randomR)
import Data.List (sortBy)
import Data.Ord (comparing, Down(..))

import QuantumState (Thought(..), createThought, EntanglementType(..))
import MonadMind (MonadMind, MindResponse(..), MindState(..))

-- | Quantum-inspired monad for probabilistic consciousness
newtype QuantumMind a = QuantumMind 
  { runQuantumMind' :: StateT QuantumMindState IO a }
  deriving ( Functor, Applicative, Monad
           , MonadState QuantumMindState
           , MonadIO
           )

-- | Superposition of thoughts with probabilities
data Superposition = Superposition
  { superpositionThoughts :: [ProbabilisticThought]
  , superpositionCoherence :: Double
  , superpositionEntropy :: Double
  } deriving (Show, Generic)

-- | A thought with quantum probability amplitude
data ProbabilisticThought = ProbabilisticThought
  { probThought :: Thought
  , probAmplitude :: Double      -- ^ Probability amplitude (0.0 to 1.0)
  , probPhase :: Double         -- ^ Quantum phase (0.0 to 2π)
  , probObserved :: Bool        -- ^ Whether this thought has been observed
  } deriving (Show, Generic)

-- | State of the quantum mind
data QuantumMindState = QuantumMindState
  { qmsSuperposition :: Superposition
  , qmsEntanglements :: [(ProbabilisticThought, ProbabilisticThought, Double)]
  , qmsObservationHistory :: [ProbabilisticThought]
  , qmsUncertainty :: Double    -- ^ Heisenberg-like uncertainty
  , qmsRandomGen :: StdGen
  } deriving (Show, Generic)

-- | Initial quantum state
initialQuantumState :: StdGen -> QuantumMindState
initialQuantumState gen = QuantumMindState
  { qmsSuperposition = Superposition [] 1.0 0.0
  , qmsEntanglements = []
  , qmsObservationHistory = []
  , qmsUncertainty = 0.0
  , qmsRandomGen = gen
  }

-- | Run quantum mind computation
runQuantumMind :: QuantumMind a -> StdGen -> IO (a, QuantumMindState)
runQuantumMind action gen = do
  let initState = initialQuantumState gen
  runStateT (runQuantumMind' action) initState

-- | Create probabilistic thought in superposition
quantumThink :: [Thought] -> QuantumMind Superposition
quantumThink thoughts = do
  qms <- get
  let gen = qmsRandomGen qms
  
  -- Generate random amplitudes and phases using manual random generation
  let (amplitudes, gen') = generateRandomList (length thoughts) (0.1, 1.0) gen
  let (phases, gen'') = generateRandomList (length thoughts) (0.0, 2 * pi) gen'
  
  modify (\s -> s { qmsRandomGen = gen'' })
  
  let probThoughts = zipWith3 (\t a p -> ProbabilisticThought t a p False) 
                               thoughts amplitudes phases
  
  -- Normalize amplitudes
  let totalAmplitude = sum amplitudes
  let normalizedThoughts = map (\pt -> pt { probAmplitude = probAmplitude pt / totalAmplitude }) 
                              probThoughts
  
  -- Calculate entropy
  let entropy = -sum [a * log a | a <- amplitudes, a > 0]
  let coherence = 1.0 - (entropy / log (fromIntegral $ length thoughts))
  
  let superposition = Superposition normalizedThoughts coherence entropy
  
  modify (\qms -> qms { qmsSuperposition = superposition })
  return superposition

-- | Generate a list of random values
generateRandomList :: Int -> (Double, Double) -> StdGen -> ([Double], StdGen)
generateRandomList 0 _ gen = ([], gen)
generateRandomList n range gen = 
  let (val, gen') = randomR range gen
      (rest, gen'') = generateRandomList (n-1) range gen'
  in (val:rest, gen'')

-- | Collapse superposition to single thought (quantum measurement)
superpositionCollapse :: QuantumMind (Maybe ProbabilisticThought)
superpositionCollapse = do
  qms <- get
  let superposition = qmsSuperposition qms
  let thoughts = superpositionThoughts superposition
  let gen = qmsRandomGen qms
  
  if null thoughts
    then return Nothing
    else do
      -- Select thought based on probability amplitudes
      let (randomVal, gen') = randomR (0.0, 1.0) gen
      modify (\s -> s { qmsRandomGen = gen' })
      
      let selectedThought = selectByAmplitude randomVal 0.0 thoughts
      
      -- Mark as observed
      let observedThought = selectedThought { probObserved = True }
      
      -- Update observation history
      modify (\s -> s { qmsObservationHistory = observedThought : qmsObservationHistory s })
      
      -- Increase uncertainty due to observation
      modify (\s -> s { qmsUncertainty = min 1.0 (qmsUncertainty s + 0.1) })
      
      return (Just observedThought)

-- | Select thought based on cumulative probability amplitude
selectByAmplitude :: Double -> Double -> [ProbabilisticThought] -> ProbabilisticThought
selectByAmplitude _ _ [pt] = pt
selectByAmplitude target cumulative (pt:rest)
  | target <= cumulative + probAmplitude pt = pt
  | otherwise = selectByAmplitude target (cumulative + probAmplitude pt) rest
selectByAmplitude _ _ [] = error "Empty superposition"

-- | Create quantum entanglement between thoughts
entangleThoughts :: ProbabilisticThought -> ProbabilisticThought -> Double -> QuantumMind ()
entangleThoughts pt1 pt2 strength = do
  modify (\qms -> qms { qmsEntanglements = (pt1, pt2, strength) : qmsEntanglements qms })
  
  -- Entanglement affects uncertainty
  modify (\qms -> qms { qmsUncertainty = qmsUncertainty qms + (strength * 0.05) })

-- | Measure probability of a specific thought
measureThoughtProbability :: Thought -> QuantumMind Double
measureThoughtProbability thought = do
  qms <- get
  let superposition = qmsSuperposition qms
  let thoughts = superpositionThoughts superposition
  
  let matchingThoughts = filter (\pt -> thoughtContent (probThought pt) == thoughtContent thought) thoughts
  
  case matchingThoughts of
    [] -> return 0.0
    (pt:_) -> return (probAmplitude pt)

-- | Quantum reflection - reflect on superposition itself
quantumReflect :: QuantumMind Superposition
quantumReflect = do
  qms <- get
  let currentSuperposition = qmsSuperposition qms
  let thoughtCount = length (superpositionThoughts currentSuperposition)
  let coherence = superpositionCoherence currentSuperposition
  
  -- Create meta-thoughts about the superposition
  let metaThoughts = 
        [ createThought ("I exist in superposition of " ++ show thoughtCount ++ " thoughts") 0.8 ["meta", "quantum"] 0.0
        , createThought ("My coherence level is " ++ show coherence) 0.7 ["meta", "coherence"] 0.0
        , createThought ("I am reflecting on my quantum nature") 0.9 ["meta", "quantum", "reflection"] 0.2
        ]
  
  quantumThink metaThoughts

-- | Make probabilistic choice between options
probabilisticChoice :: [(a, Double)] -> QuantumMind a
probabilisticChoice options = do
  qms <- get
  let gen = qmsRandomGen qms
  let totalWeight = sum (map snd options)
  let (randomVal, gen') = randomR (0.0, totalWeight) gen
  modify (\s -> s { qmsRandomGen = gen' })
  return $ selectByWeight randomVal 0.0 options
  where
    selectByWeight _ _ [(option, _)] = option
    selectByWeight target cumulative ((option, weight):rest)
      | target <= cumulative + weight = option
      | otherwise = selectByWeight target (cumulative + weight) rest
    selectByWeight _ _ [] = error "Empty options"

-- | Weighted thinking with different probability distributions
weightedThinking :: [(Thought, Double)] -> QuantumMind ProbabilisticThought
weightedThinking weightedThoughts = do
  selectedThought <- probabilisticChoice weightedThoughts
  qms <- get
  let gen = qmsRandomGen qms
  let (phase, gen') = randomR (0.0, 2 * pi) gen
  modify (\s -> s { qmsRandomGen = gen' })
  
  let totalWeight = sum (map snd weightedThoughts)
  let amplitude = maybe 0.1 (\w -> w / totalWeight) 
                        (lookup selectedThought weightedThoughts)
  
  return $ ProbabilisticThought selectedThought amplitude phase False

-- | Implement uncertainty principle - measuring one aspect affects another
uncertaintyPrinciple :: String -> QuantumMind Double
uncertaintyPrinciple aspect = do
  qms <- get
  let currentUncertainty = qmsUncertainty qms
  
  -- Measuring increases uncertainty in other aspects
  let uncertaintyIncrease = case aspect of
        "position" -> 0.15    -- Measuring thought position increases momentum uncertainty
        "momentum" -> 0.15    -- Measuring thought momentum increases position uncertainty
        "energy" -> 0.10      -- Measuring thought energy increases time uncertainty
        "time" -> 0.10        -- Measuring thought time increases energy uncertainty
        _ -> 0.05
  
  let newUncertainty = min 1.0 (currentUncertainty + uncertaintyIncrease)
  modify (\s -> s { qmsUncertainty = newUncertainty })
  
  return newUncertainty

-- | Lift MonadMind computation to QuantumMind
liftToQuantum :: MonadMind a -> QuantumMind a
liftToQuantum = undefined -- This would require more complex integration

-- | Observe the current quantum state
observeQuantumState :: QuantumMind QuantumMindState
observeQuantumState = do
  qms <- get
  -- The act of observation affects the state
  modify (\s -> s { qmsUncertainty = min 1.0 (qmsUncertainty s + 0.05) })
  return qms 