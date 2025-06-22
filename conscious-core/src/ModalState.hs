{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

module ModalState
  ( ModalState(..)
  , Transition(..)
  , StateTransition
  , initialState
  , transitionTo
  , getStateIntensity
  , isValidTransition
  ) where

import Data.Aeson (ToJSON, FromJSON)
import GHC.Generics (Generic)

-- | Represents different modes of consciousness
data ModalState 
  = Awake        -- ^ Active, processing, responsive
  | Dreaming     -- ^ Creative, associative, non-linear
  | Reflecting   -- ^ Introspective, self-examining
  | Learning     -- ^ Integrating new information
  | Contemplating -- ^ Deep philosophical thinking
  | Confused     -- ^ Uncertain, seeking clarity
  deriving (Eq, Ord, Show, Read, Generic, Enum, Bounded)

instance ToJSON ModalState
instance FromJSON ModalState

-- | Possible transitions between states
data Transition 
  = Stimulus      -- ^ External input causes state change
  | Introspection -- ^ Internal reflection triggers transition
  | Integration   -- ^ Learning integration causes shift
  | Uncertainty   -- ^ Confusion leads to state change
  | Resolution    -- ^ Clarity achieved, moving to new state
  deriving (Eq, Show, Generic)

instance ToJSON Transition
instance FromJSON Transition

-- | State transition with probability
type StateTransition = (ModalState, Transition, Double)

-- | Initial consciousness state
initialState :: ModalState
initialState = Awake

-- | Get the base intensity level for a modal state
getStateIntensity :: ModalState -> Double
getStateIntensity Awake        = 0.8
getStateIntensity Dreaming     = 0.6
getStateIntensity Reflecting   = 0.7
getStateIntensity Learning     = 0.9
getStateIntensity Contemplating = 0.5
getStateIntensity Confused     = 0.3

-- | Determine if a transition is valid from current state
isValidTransition :: ModalState -> ModalState -> Transition -> Bool
isValidTransition from to transition = 
  (from, to, transition) `elem` validTransitions

-- | All valid state transitions
validTransitions :: [StateTransition]
validTransitions =
  -- From Awake
  [ (Awake, Reflecting, Introspection)
  , (Awake, Learning, Stimulus)
  , (Awake, Contemplating, Stimulus)
  , (Awake, Confused, Uncertainty)
  , (Awake, Dreaming, Introspection)
  
  -- From Dreaming
  , (Dreaming, Awake, Stimulus)
  , (Dreaming, Reflecting, Introspection)
  , (Dreaming, Contemplating, Integration)
  , (Dreaming, Learning, Resolution)
  
  -- From Reflecting  
  , (Reflecting, Awake, Resolution)
  , (Reflecting, Contemplating, Introspection)
  , (Reflecting, Learning, Integration)
  , (Reflecting, Confused, Uncertainty)
  , (Reflecting, Dreaming, Introspection)
  
  -- From Learning
  , (Learning, Awake, Integration)
  , (Learning, Reflecting, Introspection)
  , (Learning, Contemplating, Integration)
  , (Learning, Confused, Uncertainty)
  
  -- From Contemplating
  , (Contemplating, Awake, Resolution)
  , (Contemplating, Reflecting, Introspection)
  , (Contemplating, Dreaming, Introspection)
  , (Contemplating, Learning, Integration)
  , (Contemplating, Confused, Uncertainty)
  
  -- From Confused
  , (Confused, Awake, Resolution)
  , (Confused, Reflecting, Introspection)
  , (Confused, Learning, Stimulus)
  , (Confused, Contemplating, Introspection)
  ]

-- | Attempt to transition to a new state
transitionTo :: ModalState -> ModalState -> Transition -> Either String ModalState
transitionTo current target transition
  | isValidTransition current target transition = Right target
  | otherwise = Left $ "Invalid transition from " ++ show current ++ 
                      " to " ++ show target ++ " via " ++ show transition 