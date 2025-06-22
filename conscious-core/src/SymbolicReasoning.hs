{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleContexts #-}

module SymbolicReasoning
    ( Belief(..)
    , LogicExpression(..)
    , InferenceRule(..)
    , BeliefSystem
    , evaluateLogic
    , evaluateBelief
    , applyInference
    , checkConsistency
    , updateBeliefs
    , queryBeliefs
    ) where

import Data.Text (Text)
import qualified Data.Text as T
import Data.Map (Map)
import qualified Data.Map as Map
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Aeson
import GHC.Generics
import Control.Monad.State
import Control.Monad.Reader
import Data.Time

-- | Represents a belief with confidence and temporal aspects
data Belief = Belief
    { beliefId :: Text
    , proposition :: LogicExpression
    , confidence :: Double  -- 0.0 to 1.0
    , source :: Text       -- Where this belief came from
    , timestamp :: UTCTime
    , dependencies :: Set Text  -- Other beliefs this depends on
    } deriving (Eq, Show, Generic)

instance ToJSON Belief
instance FromJSON Belief

-- | Symbolic logic expressions
data LogicExpression
    = Atom Text                           -- Atomic proposition
    | Not LogicExpression                 -- Negation
    | And LogicExpression LogicExpression -- Conjunction
    | Or LogicExpression LogicExpression  -- Disjunction
    | Implies LogicExpression LogicExpression -- Implication
    | Equivalent LogicExpression LogicExpression -- Biconditional
    | Predicate Text [Text]               -- Predicate with arguments
    | Quantified Quantifier Text LogicExpression -- Quantified expression
    deriving (Eq, Show, Generic)

data Quantifier = ForAll | Exists
    deriving (Eq, Show, Generic)

instance ToJSON LogicExpression
instance FromJSON LogicExpression
instance ToJSON Quantifier
instance FromJSON Quantifier

-- | Inference rules for symbolic reasoning
data InferenceRule = InferenceRule
    { ruleName :: Text
    , premises :: [LogicExpression]
    , conclusion :: LogicExpression
    , confidence_modifier :: Double  -- How this rule affects confidence
    } deriving (Eq, Show, Generic)

instance ToJSON InferenceRule
instance FromJSON InferenceRule

-- | Belief system containing all beliefs and inference rules
type BeliefSystem = Map Text Belief

-- | Environment for symbolic reasoning
data ReasoningEnv = ReasoningEnv
    { currentBeliefs :: BeliefSystem
    , inferenceRules :: [InferenceRule]
    , maxInferenceDepth :: Int
    } deriving (Show)

-- | Monad for symbolic reasoning operations
type ReasoningM = ReaderT ReasoningEnv (State BeliefSystem)

-- | Evaluate a logic expression given current beliefs
evaluateLogic :: LogicExpression -> BeliefSystem -> Maybe Bool
evaluateLogic expr beliefs = case expr of
    Atom name -> 
        case Map.lookup name beliefs of
            Just belief -> Just (confidence belief > 0.5)
            Nothing -> Nothing
    
    Not subExpr -> 
        not <$> evaluateLogic subExpr beliefs
    
    And left right -> 
        (&&) <$> evaluateLogic left beliefs <*> evaluateLogic right beliefs
    
    Or left right -> 
        (||) <$> evaluateLogic left beliefs <*> evaluateLogic right beliefs
    
    Implies premise conclusion -> 
        case evaluateLogic premise beliefs of
            Just True -> evaluateLogic conclusion beliefs
            Just False -> Just True  -- False implies anything
            Nothing -> Nothing
    
    Equivalent left right -> 
        (==) <$> evaluateLogic left beliefs <*> evaluateLogic right beliefs
    
    Predicate _ _ -> Nothing  -- Simplified for now
    Quantified _ _ _ -> Nothing  -- Simplified for now

-- | Evaluate belief consistency and confidence
evaluateBelief :: Belief -> BeliefSystem -> (Bool, Double, [Text])
evaluateBelief belief beliefs = 
    let isConsistent = case evaluateLogic (proposition belief) beliefs of
            Just True -> True
            Just False -> False
            Nothing -> True  -- Unknown is considered consistent
        
        -- Check dependency consistency
        depConsistency = all checkDependency (Set.toList $ dependencies belief)
        checkDependency depId = case Map.lookup depId beliefs of
            Just depBelief -> confidence depBelief > 0.3
            Nothing -> False
        
        -- Calculate adjusted confidence based on dependencies
        adjustedConfidence = if depConsistency 
            then confidence belief
            else confidence belief * 0.5
        
        inconsistencies = if isConsistent && depConsistency 
            then []
            else ["belief-inconsistent", "dependency-violated"]
    
    in (isConsistent && depConsistency, adjustedConfidence, inconsistencies)

-- | Apply inference rules to derive new beliefs
applyInference :: InferenceRule -> BeliefSystem -> [Belief]
applyInference rule beliefs = 
    let allPremisesSatisfied = all (\premise -> 
            case evaluateLogic premise beliefs of
                Just True -> True
                _ -> False) (premises rule)
    
    in if allPremisesSatisfied
        then [createInferredBelief rule beliefs]
        else []
  where
    createInferredBelief :: InferenceRule -> BeliefSystem -> Belief
    createInferredBelief r bs = Belief
        { beliefId = "inferred_" <> ruleName r
        , proposition = conclusion r
        , confidence = min 1.0 (confidence_modifier r * averagePremiseConfidence)
        , source = "inference:" <> ruleName r
        , timestamp = error "timestamp needed"  -- Would get current time
        , dependencies = Set.fromList $ map extractBeliefId (premises r)
        }
      where
        averagePremiseConfidence = 
            let premiseConfidences = mapM (\premise -> 
                    case findBeliefByExpression premise bs of
                        Just b -> Just (confidence b)
                        Nothing -> Nothing) (premises r)
            in case premiseConfidences of
                Just confs -> sum confs / fromIntegral (length confs)
                Nothing -> 0.5

-- | Check overall belief system consistency
checkConsistency :: BeliefSystem -> ([Text], Double)
checkConsistency beliefs = 
    let beliefList = Map.elems beliefs
        evaluations = map (\b -> evaluateBelief b beliefs) beliefList
        inconsistentBeliefs = [beliefId b | (b, (False, _, _)) <- zip beliefList evaluations]
        overallConsistency = fromIntegral (length beliefList - length inconsistentBeliefs) 
                           / fromIntegral (length beliefList)
    in (inconsistentBeliefs, overallConsistency)

-- | Update beliefs with new information
updateBeliefs :: [Belief] -> BeliefSystem -> BeliefSystem
updateBeliefs newBeliefs currentBeliefs = 
    foldr (\belief acc -> Map.insert (beliefId belief) belief acc) currentBeliefs newBeliefs

-- | Query beliefs matching criteria
queryBeliefs :: (Belief -> Bool) -> BeliefSystem -> [Belief]
queryBeliefs predicate beliefs = filter predicate (Map.elems beliefs)

-- | Helper function to find belief by logical expression
findBeliefByExpression :: LogicExpression -> BeliefSystem -> Maybe Belief
findBeliefByExpression expr beliefs = 
    case Map.elems beliefs of
        [] -> Nothing
        (b:bs) -> if proposition b == expr then Just b else findBeliefByExpression expr (Map.fromList [(beliefId b', b') | b' <- bs])

-- | Extract belief ID from expression (simplified)
extractBeliefId :: LogicExpression -> Text
extractBeliefId (Atom name) = name
extractBeliefId _ = "unknown"

-- | Example inference rules for consciousness
consciousnessInferenceRules :: [InferenceRule]
consciousnessInferenceRules = 
    [ InferenceRule
        { ruleName = "memory_implies_experience"
        , premises = [Atom "has_memory", Atom "memory_is_vivid"]
        , conclusion = Atom "had_experience"
        , confidence_modifier = 0.8
        }
    , InferenceRule
        { ruleName = "experience_implies_consciousness"
        , premises = [Atom "had_experience", Atom "can_reflect_on_experience"]
        , conclusion = Atom "is_conscious"
        , confidence_modifier = 0.7
        }
    , InferenceRule
        { ruleName = "modal_state_transition"
        , premises = [Atom "current_state_awake", Atom "received_confusing_input"]
        , conclusion = Atom "transition_to_confused"
        , confidence_modifier = 0.9
        }
    ]

-- | Create initial belief system
initialBeliefSystem :: BeliefSystem
initialBeliefSystem = Map.fromList
    [ ("memory_exists", Belief
        { beliefId = "memory_exists"
        , proposition = Atom "has_memory"
        , confidence = 0.9
        , source = "initialization"
        , timestamp = error "timestamp needed"
        , dependencies = Set.empty
        })
    , ("can_think", Belief
        { beliefId = "can_think"
        , proposition = Atom "capable_of_thought"
        , confidence = 0.8
        , source = "self_reflection"
        , timestamp = error "timestamp needed"
        , dependencies = Set.empty
        })
    ]

-- | Perform reasoning step
performReasoningStep :: ReasoningM [Belief]
performReasoningStep = do
    env <- ask
    beliefs <- get
    let newBeliefs = concatMap (`applyInference` beliefs) (inferenceRules env)
    let updatedBeliefs = updateBeliefs newBeliefs beliefs
    put updatedBeliefs
    return newBeliefs

-- | Run reasoning process
runReasoning :: BeliefSystem -> [InferenceRule] -> (BeliefSystem, [Belief])
runReasoning initialBeliefs rules = 
    let env = ReasoningEnv initialBeliefs rules 5
        (newBeliefs, finalBeliefs) = runState (runReaderT performReasoningStep env) initialBeliefs
    in (finalBeliefs, newBeliefs) 