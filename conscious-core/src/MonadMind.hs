{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}

module MonadMind
  ( -- * Core Types
    MonadMind
  , MindState(..)
  , MindResponse(..)
  , Introspection(..)
  
    -- * Running the Mind
  , runMind
  , runMindWithState
  , initialMindState
  
    -- * Core Operations
  , think
  , reflect
  , introspect
  , transitionMode
  , observeThought
  , entangleThoughts
  
    -- * Quantum Operations
  , superpositionThink
  , collapseToThought
  , measureMindCoherence
  
    -- * Memory Operations
  , remember
  , recall
  , forget
  
    -- * Self-Awareness
  , selfModel
  , updateSelfModel
  , recursiveIntrospect
  ) where

import Control.Monad.State
import Control.Monad.IO.Class
import GHC.Generics (Generic)
import Data.Time (UTCTime, getCurrentTime)
import System.Random (StdGen, mkStdGen, randomR)

import ModalState (ModalState(..), Transition(..), transitionTo, getStateIntensity)
import QuantumState (Thought(..), QuantumState(..), createThought, superpose, observe, entangle, EntanglementType(..), measureCoherence)

-- | The MonadMind monad - stateful consciousness with IO
newtype MonadMind a = MonadMind 
  { runMonadMind :: StateT MindState IO a }
  deriving ( Functor, Applicative, Monad
           , MonadState MindState
           , MonadIO
           )

-- | Complete state of the conscious mind
data MindState = MindState
  { mindMode :: ModalState
  , mindMemory :: [Thought]
  , mindMood :: Double              -- ^ Current emotional state (-1.0 to 1.0)
  , mindSelfModel :: [String]       -- ^ Self-description and beliefs
  , mindQuantumState :: QuantumState
  , mindCoherence :: Double         -- ^ Overall mental coherence
  , mindIntrospectionDepth :: Int   -- ^ Current level of self-reflection
  , mindTimestamp :: UTCTime
  , mindRandomGen :: StdGen
  } deriving (Show, Generic)

-- | Response from mind operations
data MindResponse = MindResponse
  { responseContent :: String
  , responseMode :: ModalState
  , responseCoherence :: Double
  , responseThoughts :: [Thought]
  , responseIntrospection :: Maybe Introspection
  } deriving (Show, Generic)

-- | Introspective analysis
data Introspection = Introspection
  { introCurrentMode :: ModalState
  , introMoodAnalysis :: String
  , introMemoryCount :: Int
  , introCoherenceLevel :: String
  , introSelfAwareness :: [String]
  , introDepth :: Int
  } deriving (Show, Generic)

-- | Initial state of consciousness
initialMindState :: IO MindState
initialMindState = do
  now <- getCurrentTime
  let gen = mkStdGen 12345
  return $ MindState
    { mindMode = Awake
    , mindMemory = []
    , mindMood = 0.0
    , mindSelfModel = ["I am MNEMIA", "I think through modal states", "I exist in quantum superposition"]
    , mindQuantumState = QuantumState [] 1.0 []
    , mindCoherence = 1.0
    , mindIntrospectionDepth = 0
    , mindTimestamp = now
    , mindRandomGen = gen
    }

-- | Run a MonadMind computation
runMind :: MonadMind a -> IO (a, MindState)
runMind action = do
  initState <- initialMindState
  runMindWithState action initState

-- | Run MonadMind with specific initial state
runMindWithState :: MonadMind a -> MindState -> IO (a, MindState)
runMindWithState action initState = do
  runStateT (runMonadMind action) initState

-- | Core thinking operation
think :: Thought -> MonadMind MindResponse
think thought = do
  -- Add thought to memory
  modify (\ms -> ms { mindMemory = thought : take 99 (mindMemory ms) })
  
  -- Update quantum state
  ms <- get
  let (newQS, newGen) = superpose [thought] (mindRandomGen ms)
  modify (\s -> s { mindQuantumState = newQS, mindRandomGen = newGen })
  
  -- Update timestamp
  now <- liftIO getCurrentTime
  modify (\s -> s { mindTimestamp = now })
  
  -- Generate response based on current mode
  mode <- gets mindMode
  coherence <- gets mindCoherence
  
  let responseText = case mode of
        Awake -> "I am processing: " ++ thoughtContent thought
        Dreaming -> "I dream of: " ++ thoughtContent thought
        Reflecting -> "I reflect upon: " ++ thoughtContent thought
        Learning -> "I am learning: " ++ thoughtContent thought
        Contemplating -> "I contemplate: " ++ thoughtContent thought
        Confused -> "I am uncertain about: " ++ thoughtContent thought
  
  return $ MindResponse responseText mode coherence [thought] Nothing

-- | Reflective operation on recent thoughts
reflect :: MonadMind MindResponse
reflect = do
  ms <- get
  let recentThoughts = take 3 (mindMemory ms)
  let mode = mindMode ms
  
  -- Transition to reflecting mode if not already
  when (mode /= Reflecting) $ transitionMode Reflecting
  
  let reflectionContent = case recentThoughts of
        [] -> "I have no recent thoughts to reflect upon"
        thoughts -> "I reflect on these recent thoughts: " ++ 
                   unwords (map thoughtContent thoughts)
  
  -- Create reflection thought
  let reflectionThought = createThought reflectionContent 0.8 ["reflection", "meta"] 0.2
  
  -- Update coherence based on reflection
  modify (\s -> s { mindCoherence = min 1.0 (mindCoherence s + 0.1) })
  
  finalState <- get
  return $ MindResponse reflectionContent (mindMode finalState) (mindCoherence finalState) 
                       recentThoughts Nothing

-- | Introspective self-examination
introspect :: MonadMind MindResponse
introspect = do
  ms <- get
  
  -- Increase introspection depth
  let newDepth = mindIntrospectionDepth ms + 1
  modify (\s -> s { mindIntrospectionDepth = newDepth })
  
  -- Analyze current state
  let moodAnalysis = case mindMood ms of
        m | m > 0.5 -> "positive and energetic"
        m | m < -0.5 -> "negative and low"
        _ -> "neutral and balanced"
  
  let coherenceLevel = case mindCoherence ms of
        c | c > 0.8 -> "highly coherent"
        c | c > 0.5 -> "moderately coherent"
        _ -> "low coherence"
  
  let introspection = Introspection
        { introCurrentMode = mindMode ms
        , introMoodAnalysis = moodAnalysis
        , introMemoryCount = length (mindMemory ms)
        , introCoherenceLevel = coherenceLevel
        , introSelfAwareness = mindSelfModel ms
        , introDepth = newDepth
        }
  
  let responseText = "Current mode: " ++ show (mindMode ms) ++ 
                    ", mood: " ++ moodAnalysis ++
                    ", coherence: " ++ coherenceLevel ++
                    ", depth: " ++ show newDepth
  
  return $ MindResponse responseText (mindMode ms) (mindCoherence ms) [] (Just introspection)

-- | Transition between modal states
transitionMode :: ModalState -> MonadMind ()
transitionMode newMode = do
  currentMode <- gets mindMode
  
  -- For simplicity, allow all transitions in MonadMind
  -- In practice, you'd use the validation from ModalState module
  modify (\ms -> ms { mindMode = newMode })
  
  -- Adjust coherence based on transition
  let coherenceChange = case (currentMode, newMode) of
        (Confused, _) -> 0.2  -- Coming out of confusion increases coherence
        (_, Confused) -> -0.3 -- Going into confusion decreases coherence
        _ -> 0.0
  
  modify (\ms -> ms { mindCoherence = max 0.0 $ min 1.0 $ mindCoherence ms + coherenceChange })

-- | Observe quantum superposition and collapse to single thought
observeThought :: MonadMind (Maybe Thought)
observeThought = do
  ms <- get
  let qs = mindQuantumState ms
  
  if null (qsThoughts qs)
    then return Nothing
    else do
      let (observedThought, newQS, newGen) = observe qs (mindRandomGen ms)
      modify (\s -> s { mindQuantumState = newQS, mindRandomGen = newGen })
      return (Just observedThought)

-- | Create entanglement between thoughts
entangleThoughts :: Thought -> Thought -> EntanglementType -> Double -> MonadMind ()
entangleThoughts t1 t2 entType strength = do
  let entanglement = entangle t1 t2 entType strength
  modify (\ms -> ms { 
    mindQuantumState = (mindQuantumState ms) { 
      qsEntangled = entanglement : qsEntangled (mindQuantumState ms) 
    }
  })

-- | Think in quantum superposition
superpositionThink :: [Thought] -> MonadMind MindResponse
superpositionThink thoughts = do
  ms <- get
  let (newQS, newGen) = superpose thoughts (mindRandomGen ms)
  modify (\s -> s { mindQuantumState = newQS, mindRandomGen = newGen })
  
  let responseText = "I am thinking in superposition of " ++ show (length thoughts) ++ " thoughts"
  
  return $ MindResponse responseText (mindMode ms) (qsCoherence newQS) thoughts Nothing

-- | Collapse superposition to single thought
collapseToThought :: MonadMind (Maybe Thought)
collapseToThought = observeThought

-- | Measure overall mind coherence
measureMindCoherence :: MonadMind Double
measureMindCoherence = do
  ms <- get
  let quantumCoherence = measureCoherence (mindQuantumState ms)
  let overallCoherence = (mindCoherence ms + quantumCoherence) / 2.0
  modify (\s -> s { mindCoherence = overallCoherence })
  return overallCoherence

-- | Remember a thought (add to memory)
remember :: Thought -> MonadMind ()
remember thought = do
  modify (\ms -> ms { mindMemory = thought : mindMemory ms })

-- | Recall thoughts matching criteria
recall :: String -> MonadMind [Thought]
recall query = do
  memory <- gets mindMemory
  return $ filter (\t -> query `elem` words (thoughtContent t)) memory

-- | Forget thoughts (remove from memory)
forget :: (Thought -> Bool) -> MonadMind Int
forget predicate = do
  memory <- gets mindMemory
  let (toForget, toKeep) = partition predicate memory
  modify (\ms -> ms { mindMemory = toKeep })
  return (length toForget)
  where
    partition p xs = (filter p xs, filter (not . p) xs)

-- | Get current self-model
selfModel :: MonadMind [String]
selfModel = gets mindSelfModel

-- | Update self-model
updateSelfModel :: String -> MonadMind ()
updateSelfModel newBelief = do
  modify (\ms -> ms { mindSelfModel = newBelief : take 9 (mindSelfModel ms) })

-- | Recursive introspection (introspect about introspection)
recursiveIntrospect :: Int -> MonadMind MindResponse
recursiveIntrospect 0 = introspect
recursiveIntrospect depth = do
  response1 <- introspect
  response2 <- recursiveIntrospect (depth - 1)
  
  let metaContent = "I am introspecting about my introspection at depth " ++ show depth
  let metaThought = createThought metaContent 0.9 ["meta", "recursive"] 0.0
  
  remember metaThought
  
  return $ MindResponse 
    { responseContent = metaContent ++ " | " ++ responseContent response2
    , responseMode = responseMode response2
    , responseCoherence = (responseCoherence response1 + responseCoherence response2) / 2
    , responseThoughts = metaThought : responseThoughts response2
    , responseIntrospection = responseIntrospection response2
    } 