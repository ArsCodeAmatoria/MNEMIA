{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE RankNTypes #-}

module SymbolicReasoning
    ( -- Core Types
      Belief(..)
    , LogicExpression(..)
    , Term(..)
    , Quantifier(..)
    , InferenceRule(..)
    , RuleType(..)
    , BeliefSystem(..)
    , ConsistencyReport(..)
    , ConsciousnessRule(..)
    , ModalContext(..)
    
    -- Logic Engine
    , evaluateLogic
    , unify
    , substitute
    , applySubstitution
    , normalizeExpression
    
    -- Belief System
    , BeliefState(..)
    , evaluateBelief
    , updateBelief
    , addBelief
    , removeBelief
    , queryBeliefs
    , getBeliefConfidence
    , propagateConfidence
    
    -- Inference Engine
    , applyInference
    , applyInferenceRules
    , deriveNewBeliefs
    , backwardChaining
    , forwardChaining
    , resolutionProof
    
    -- Consistency Checking
    , checkConsistency
    , detectContradictions
    , resolveContradictions
    , validateBeliefSystem
    , computeCoherence
    
    -- Consciousness Rules
    , consciousnessInferenceRules
    , modalStateRules
    , selfAwarenessRules
    , experienceRules
    , memoryRules
    , emotionRules
    , applyConsciousnessRules
    
    -- Reasoning Monad
    , ReasoningM
    , ReasoningEnv(..)
    , runReasoning
    , performInference
    , performConsistencyCheck
    , performConsciousnessReasoning
    
    -- Utilities
    , prettyPrintExpression
    , exportBeliefSystem
    , importBeliefSystem
    ) where

import Data.Text (Text)
import qualified Data.Text as T
import Data.Map.Strict (Map)
import qualified Data.Map.Strict as Map
import Data.Set (Set)
import qualified Data.Set as Set
import Data.List (foldl', nub, partition)
import Data.Maybe (catMaybes, fromMaybe, isJust, mapMaybe, listToMaybe)
import Data.Aeson
import GHC.Generics
import Control.Monad.State
import Control.Monad.Reader
import Control.Monad.Except
import Control.Monad.Writer
import Data.Time
import Data.UUID (UUID)
import qualified Data.UUID as UUID
import qualified Data.UUID.V4 as UUID

-- | Terms in first-order logic
data Term
    = Variable Text              -- Variables (x, y, z)
    | Constant Text              -- Constants (a, b, c)
    | Function Text [Term]       -- Function applications f(x,y)
    deriving (Eq, Ord, Show, Generic)

-- | Quantifiers for first-order logic
data Quantifier = ForAll | Exists
    deriving (Eq, Ord, Show, Generic)

-- | First-order logic expressions with full quantifier support
data LogicExpression
    = Atom Text                                    -- Atomic proposition P
    | Predicate Text [Term]                        -- Predicate P(x,y,z)
    | Not LogicExpression                          -- ¬φ
    | And LogicExpression LogicExpression          -- φ ∧ ψ
    | Or LogicExpression LogicExpression           -- φ ∨ ψ
    | Implies LogicExpression LogicExpression      -- φ → ψ
    | Equivalent LogicExpression LogicExpression   -- φ ↔ ψ
    | Quantified Quantifier Text LogicExpression  -- ∀x φ(x) or ∃x φ(x)
    deriving (Eq, Ord, Show, Generic)

-- | Types of inference rules
data RuleType
    = ModusPonens           -- φ → ψ, φ ⊢ ψ
    | ModusTollens          -- φ → ψ, ¬ψ ⊢ ¬φ
    | HypotheticalSyllogism -- φ → ψ, ψ → χ ⊢ φ → χ
    | DisjunctiveSyllogism  -- φ ∨ ψ, ¬φ ⊢ ψ
    | Simplification        -- φ ∧ ψ ⊢ φ
    | Conjunction           -- φ, ψ ⊢ φ ∧ ψ
    | Addition              -- φ ⊢ φ ∨ ψ
    | UniversalInstantiation -- ∀x φ(x) ⊢ φ(c)
    | ExistentialGeneralization -- φ(c) ⊢ ∃x φ(x)
    | ConsciousnessSpecific Text -- Domain-specific consciousness rules
    deriving (Eq, Ord, Show, Generic)

-- | Enhanced inference rule with metadata
data InferenceRule = InferenceRule
    { ruleId :: UUID
    , ruleName :: Text
    , ruleType :: RuleType
    , premises :: [LogicExpression]
    , conclusion :: LogicExpression
    , confidenceModifier :: Double      -- How this rule affects confidence (0.0 to 1.0)
    , priority :: Int                   -- Rule application priority
    , modalContexts :: [Text]           -- Which modal states this rule applies to
    , consciousnessLevel :: Int         -- Minimum consciousness level required
    , metadata :: Map Text Value        -- Additional rule metadata
    } deriving (Eq, Show, Generic)

-- | Enhanced belief with dependency tracking and provenance
data Belief = Belief
    { beliefId :: UUID
    , proposition :: LogicExpression
    , confidence :: Double              -- Confidence level (0.0 to 1.0)
    , source :: Text                    -- Source of this belief
    , timestamp :: UTCTime
    , dependencies :: Set UUID          -- Beliefs this depends on
    , derivedFrom :: Maybe UUID         -- Rule that derived this belief
    , modalContext :: Maybe Text        -- Modal state when belief was formed
    , evidenceStrength :: Double        -- Strength of supporting evidence
    , coherenceScore :: Double          -- How well this fits with other beliefs
    , accessCount :: Int                -- How often this belief is accessed
    , lastAccessed :: Maybe UTCTime     -- When this belief was last used
    , tags :: Set Text                  -- Categorical tags for organization
    , metadata :: Map Text Value        -- Additional belief metadata
    } deriving (Eq, Show, Generic)

-- | Belief system state with comprehensive tracking
data BeliefSystem = BeliefSystem
    { beliefs :: Map UUID Belief
    , inferenceRules :: Map UUID InferenceRule
    , contradictions :: Set (UUID, UUID)  -- Pairs of contradictory beliefs
    , coherenceGraph :: Map UUID (Set UUID) -- Belief coherence relationships
    , derivationTree :: Map UUID [UUID]   -- Derivation history
    , consistencyScore :: Double          -- Overall system consistency
    , lastUpdated :: UTCTime
    , version :: Int                      -- Version for change tracking
    } deriving (Eq, Show, Generic)

-- | Consistency analysis report
data ConsistencyReport = ConsistencyReport
    { overallConsistency :: Double
    , contradictoryBeliefs :: [(UUID, UUID, Text)]  -- Belief pairs + reason
    , weakBeliefs :: [UUID]                         -- Low-confidence beliefs
    , isolatedBeliefs :: [UUID]                     -- Beliefs with no support
    , coherenceClusters :: [[UUID]]                 -- Groups of coherent beliefs
    , recommendedActions :: [Text]                  -- Suggested improvements
    , analysisTimestamp :: UTCTime
    } deriving (Eq, Show, Generic)

-- | Consciousness-specific inference rules
data ConsciousnessRule = ConsciousnessRule
    { consciousnessRuleId :: UUID
    , domain :: Text                    -- "self_awareness", "experience", "memory", etc.
    , condition :: LogicExpression      -- When this rule applies
    , action :: LogicExpression         -- What to infer
    , modalStates :: [Text]             -- Applicable modal states
    , confidenceThreshold :: Double     -- Minimum confidence to trigger
    , description :: Text               -- Human-readable description
    } deriving (Eq, Show, Generic)

-- | Modal context for reasoning
data ModalContext = ModalContext
    { currentModalState :: Text
    , previousModalState :: Maybe Text
    , transitionReason :: Maybe Text
    , stateIntensity :: Double
    , emotionalContext :: Map Text Double
    , temporalContext :: UTCTime
    } deriving (Eq, Show, Generic)

-- | Belief state for monadic operations
data BeliefState = BeliefState
    { currentBeliefs :: BeliefSystem
    , reasoningDepth :: Int
    , maxDepth :: Int
    , derivationLog :: [Text]
    , modalContext :: ModalContext
    } deriving (Show)

-- | Environment for reasoning operations
data ReasoningEnv = ReasoningEnv
    { maxInferenceDepth :: Int
    , confidenceThreshold :: Double
    , consistencyThreshold :: Double
    , enableConsciousnessRules :: Bool
    , debugMode :: Bool
    , timeLimit :: NominalDiffTime
    } deriving (Show)

-- | Reasoning monad with error handling and logging
type ReasoningM = ReaderT ReasoningEnv (StateT BeliefState (ExceptT Text (Writer [Text])))

-- JSON instances
instance ToJSON Term
instance FromJSON Term
instance ToJSON Quantifier
instance FromJSON Quantifier
instance ToJSON LogicExpression
instance FromJSON LogicExpression
instance ToJSON RuleType
instance FromJSON RuleType
instance ToJSON InferenceRule
instance FromJSON InferenceRule
instance ToJSON Belief
instance FromJSON Belief
instance ToJSON BeliefSystem
instance FromJSON BeliefSystem
instance ToJSON ConsistencyReport
instance FromJSON ConsistencyReport
instance ToJSON ConsciousnessRule
instance FromJSON ConsciousnessRule
instance ToJSON ModalContext
instance FromJSON ModalContext

-- | Substitution mapping from variables to terms
type Substitution = Map Text Term

-- | Unification of two terms
unify :: Term -> Term -> Maybe Substitution
unify (Variable x) t = Just $ Map.singleton x t
unify t (Variable x) = Just $ Map.singleton x t
unify (Constant c1) (Constant c2)
    | c1 == c2 = Just Map.empty
    | otherwise = Nothing
unify (Function f1 args1) (Function f2 args2)
    | f1 == f2 && length args1 == length args2 = 
        foldl' combineSubstitutions (Just Map.empty) (zipWith unify args1 args2)
    | otherwise = Nothing
unify _ _ = Nothing

-- | Combine substitutions with occurs check
combineSubstitutions :: Maybe Substitution -> Maybe Substitution -> Maybe Substitution
combineSubstitutions Nothing _ = Nothing
combineSubstitutions _ Nothing = Nothing
combineSubstitutions (Just s1) (Just s2) = 
    let combined = Map.union s1 s2
        conflicts = Map.intersectionWith (/=) s1 s2
    in if Map.null (Map.filter id conflicts)
       then Just combined
       else Nothing

-- | Apply substitution to a term
substitute :: Substitution -> Term -> Term
substitute subst (Variable x) = fromMaybe (Variable x) (Map.lookup x subst)
substitute subst (Function f args) = Function f (map (substitute subst) args)
substitute _ constant@(Constant _) = constant

-- | Apply substitution to a logic expression
applySubstitution :: Substitution -> LogicExpression -> LogicExpression
applySubstitution subst expr = case expr of
    Atom name -> Atom name
    Predicate name terms -> Predicate name (map (substitute subst) terms)
    Not e -> Not (applySubstitution subst e)
    And e1 e2 -> And (applySubstitution subst e1) (applySubstitution subst e2)
    Or e1 e2 -> Or (applySubstitution subst e1) (applySubstitution subst e2)
    Implies e1 e2 -> Implies (applySubstitution subst e1) (applySubstitution subst e2)
    Equivalent e1 e2 -> Equivalent (applySubstitution subst e1) (applySubstitution subst e2)
    Quantified q var e -> 
        let filteredSubst = Map.delete var subst
        in Quantified q var (applySubstitution filteredSubst e)

-- | Normalize logic expression to canonical form
normalizeExpression :: LogicExpression -> LogicExpression
normalizeExpression expr = case expr of
    Not (Not e) -> normalizeExpression e  -- Double negation elimination
    Not (And e1 e2) -> Or (Not (normalizeExpression e1)) (Not (normalizeExpression e2))  -- De Morgan's law
    Not (Or e1 e2) -> And (Not (normalizeExpression e1)) (Not (normalizeExpression e2))  -- De Morgan's law
    And e1 e2 -> And (normalizeExpression e1) (normalizeExpression e2)
    Or e1 e2 -> Or (normalizeExpression e1) (normalizeExpression e2)
    Implies e1 e2 -> Or (Not (normalizeExpression e1)) (normalizeExpression e2)  -- Implication elimination
    Equivalent e1 e2 -> 
        let ne1 = normalizeExpression e1
            ne2 = normalizeExpression e2
        in And (Or (Not ne1) ne2) (Or (Not ne2) ne1)  -- Biconditional elimination
    other -> other

-- | Evaluate logic expression with three-valued logic (True, False, Unknown)
evaluateLogic :: LogicExpression -> BeliefSystem -> Maybe Bool
evaluateLogic expr beliefSystem = case normalizeExpression expr of
    Atom name -> 
        let matchingBeliefs = Map.elems (beliefs beliefSystem)
            atomicBeliefs = [b | b <- matchingBeliefs, 
                           case proposition b of
                               Atom n -> n == name
                               _ -> False]
        in case atomicBeliefs of
            [] -> Nothing  -- Unknown
            bs -> let avgConfidence = sum (map confidence bs) / fromIntegral (length bs)
                  in Just (avgConfidence > 0.5)
    
    Predicate name terms -> 
        -- Simplified predicate evaluation - would need full unification in practice
        Nothing
    
    Not subExpr -> 
        case evaluateLogic subExpr beliefSystem of
            Just True -> Just False
            Just False -> Just True
            Nothing -> Nothing
    
    And left right -> 
        case (evaluateLogic left beliefSystem, evaluateLogic right beliefSystem) of
            (Just True, Just True) -> Just True
            (Just False, _) -> Just False
            (_, Just False) -> Just False
            _ -> Nothing
    
    Or left right -> 
        case (evaluateLogic left beliefSystem, evaluateLogic right beliefSystem) of
            (Just True, _) -> Just True
            (_, Just True) -> Just True
            (Just False, Just False) -> Just False
            _ -> Nothing
    
    Quantified ForAll var subExpr -> 
        -- Simplified - would need domain of discourse
        Nothing
    
    Quantified Exists var subExpr -> 
        -- Simplified - would need domain of discourse
        Nothing
    
    _ -> Nothing

-- | Evaluate belief consistency and update derived metrics
evaluateBelief :: UUID -> BeliefSystem -> Maybe (Bool, Double, [Text])
evaluateBelief beliefId beliefSystem = do
    belief <- Map.lookup beliefId (beliefs beliefSystem)
    let isConsistent = case evaluateLogic (proposition belief) beliefSystem of
            Just True -> True
            Just False -> False
            Nothing -> True  -- Unknown is considered consistent
    
    -- Check dependency consistency
    let deps = Set.toList (dependencies belief)
        depConsistencies = map (\depId -> 
            case Map.lookup depId (beliefs beliefSystem) of
                Just depBelief -> confidence depBelief > 0.3
                Nothing -> False) deps
        depConsistency = null deps || and depConsistencies
    
    -- Calculate coherence with related beliefs
    let relatedBeliefs = maybe [] Set.toList (Map.lookup beliefId (coherenceGraph beliefSystem))
        coherenceScores = map (\relId ->
            case Map.lookup relId (beliefs beliefSystem) of
                Just relBelief -> confidence relBelief
                Nothing -> 0.0) relatedBeliefs
        avgCoherence = if null coherenceScores 
                      then 0.5 
                      else sum coherenceScores / fromIntegral (length coherenceScores)
    
    -- Adjusted confidence based on dependencies and coherence
    let adjustedConfidence = confidence belief * 
                           (if depConsistency then 1.0 else 0.5) * 
                           (0.5 + 0.5 * avgCoherence)
    
    let inconsistencies = concat
            [ if isConsistent then [] else ["logical-inconsistency"]
            , if depConsistency then [] else ["dependency-violation"]
            , if avgCoherence > 0.3 then [] else ["low-coherence"]
            ]
    
    return (isConsistent && depConsistency, adjustedConfidence, inconsistencies)

-- | Add new belief to system with dependency tracking
addBelief :: Belief -> BeliefSystem -> BeliefSystem
addBelief belief beliefSystem = 
    let updatedBeliefs = Map.insert (beliefId belief) belief (beliefs beliefSystem)
        updatedSystem = beliefSystem { beliefs = updatedBeliefs
                                     , version = version beliefSystem + 1
                                     }
    in updateCoherenceGraph (beliefId belief) updatedSystem

-- | Update coherence graph when adding belief
updateCoherenceGraph :: UUID -> BeliefSystem -> BeliefSystem
updateCoherenceGraph newBeliefId beliefSystem = 
    case Map.lookup newBeliefId (beliefs beliefSystem) of
        Nothing -> beliefSystem
        Just newBelief -> 
            let relatedBeliefs = findRelatedBeliefs newBelief beliefSystem
                newCoherenceMap = foldl' (\acc relId -> 
                    Map.insertWith Set.union relId (Set.singleton newBeliefId) acc)
                    (coherenceGraph beliefSystem) relatedBeliefs
                finalCoherenceMap = Map.insert newBeliefId (Set.fromList relatedBeliefs) newCoherenceMap
            in beliefSystem { coherenceGraph = finalCoherenceMap }

-- | Find beliefs related to a given belief
findRelatedBeliefs :: Belief -> BeliefSystem -> [UUID]
findRelatedBeliefs belief beliefSystem = 
    let allBeliefs = Map.toList (beliefs beliefSystem)
        related = [bid | (bid, b) <- allBeliefs, 
                        bid /= beliefId belief,
                        isRelated (proposition belief) (proposition b)]
    in related

-- | Check if two propositions are related
isRelated :: LogicExpression -> LogicExpression -> Bool
isRelated expr1 expr2 = 
    sharesPredicate expr1 expr2 || sharesAtom expr1 expr2

-- | Check if expressions share predicates
sharesPredicate :: LogicExpression -> LogicExpression -> Bool
sharesPredicate (Predicate name1 _) (Predicate name2 _) = name1 == name2
sharesPredicate expr1 expr2 = 
    let preds1 = extractPredicates expr1
        preds2 = extractPredicates expr2
    in not $ Set.null $ Set.intersection preds1 preds2

-- | Check if expressions share atoms
sharesAtom :: LogicExpression -> LogicExpression -> Bool
sharesAtom (Atom name1) (Atom name2) = name1 == name2
sharesAtom expr1 expr2 = 
    let atoms1 = extractAtoms expr1
        atoms2 = extractAtoms expr2
    in not $ Set.null $ Set.intersection atoms1 atoms2

-- | Extract all predicates from an expression
extractPredicates :: LogicExpression -> Set Text
extractPredicates expr = case expr of
    Predicate name _ -> Set.singleton name
    Not e -> extractPredicates e
    And e1 e2 -> Set.union (extractPredicates e1) (extractPredicates e2)
    Or e1 e2 -> Set.union (extractPredicates e1) (extractPredicates e2)
    Implies e1 e2 -> Set.union (extractPredicates e1) (extractPredicates e2)
    Equivalent e1 e2 -> Set.union (extractPredicates e1) (extractPredicates e2)
    Quantified _ _ e -> extractPredicates e
    _ -> Set.empty

-- | Extract all atoms from an expression
extractAtoms :: LogicExpression -> Set Text
extractAtoms expr = case expr of
    Atom name -> Set.singleton name
    Not e -> extractAtoms e
    And e1 e2 -> Set.union (extractAtoms e1) (extractAtoms e2)
    Or e1 e2 -> Set.union (extractAtoms e1) (extractAtoms e2)
    Implies e1 e2 -> Set.union (extractAtoms e1) (extractAtoms e2)
    Equivalent e1 e2 -> Set.union (extractAtoms e1) (extractAtoms e2)
    Quantified _ _ e -> extractAtoms e
    _ -> Set.empty

-- | Example inference rules for consciousness
consciousnessInferenceRules :: [InferenceRule]
consciousnessInferenceRules = 
    [ InferenceRule
        { ruleName = "memory_implies_experience"
        , premises = [Atom "has_memory", Atom "memory_is_vivid"]
        , conclusion = Atom "had_experience"
        , confidenceModifier = 0.8
        }
    , InferenceRule
        { ruleName = "experience_implies_consciousness"
        , premises = [Atom "had_experience", Atom "can_reflect_on_experience"]
        , conclusion = Atom "is_conscious"
        , confidenceModifier = 0.7
        }
    , InferenceRule
        { ruleName = "modal_state_transition"
        , premises = [Atom "current_state_awake", Atom "received_confusing_input"]
        , conclusion = Atom "transition_to_confused"
        , confidenceModifier = 0.9
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
    let env = ReasoningEnv 5 0.5 0.5 True False 0
        (newBeliefs, finalBeliefs) = runState (runReaderT performReasoningStep env) initialBeliefs
    in (finalBeliefs, newBeliefs) 

-- | Apply inference rule to derive new beliefs
applyInference :: InferenceRule -> BeliefSystem -> ReasoningM [Belief]
applyInference rule beliefSystem = do
    env <- ask
    let premises = premises rule
        allPremisesSatisfied = all (\premise -> 
            case evaluateLogic premise beliefSystem of
                Just True -> True
                _ -> False) premises
    
    if allPremisesSatisfied && confidence rule > confidenceThreshold env
        then do
            newBelief <- createInferredBelief rule beliefSystem
            tell ["Applied rule: " <> ruleName rule]
            return [newBelief]
        else return []
  where
    createInferredBelief :: InferenceRule -> BeliefSystem -> ReasoningM Belief
    createInferredBelief r bs = do
        currentTime <- liftIO getCurrentTime
        newId <- liftIO UUID.nextRandom
        modalCtx <- gets modalContext
        
        let premiseBeliefs = mapMaybe (findBeliefByExpression bs) (premises r)
            averageConfidence = if null premiseBeliefs 
                              then 0.5 
                              else sum (map confidence premiseBeliefs) / fromIntegral (length premiseBeliefs)
            adjustedConfidence = min 1.0 (confidenceModifier r * averageConfidence)
            
        return Belief
            { beliefId = newId
            , proposition = conclusion r
            , confidence = adjustedConfidence
            , source = "inference:" <> ruleName r
            , timestamp = currentTime
            , dependencies = Set.fromList (map beliefId premiseBeliefs)
            , derivedFrom = Just (ruleId r)
            , modalContext = Just (currentModalState modalCtx)
            , evidenceStrength = averageConfidence
            , coherenceScore = 0.5  -- Will be computed later
            , accessCount = 0
            , lastAccessed = Nothing
            , tags = Set.fromList ["inferred", T.toLower (currentModalState modalCtx)]
            , metadata = Map.fromList [("rule_type", String (T.pack $ show $ ruleType r))]
            }

-- | Apply multiple inference rules
applyInferenceRules :: [InferenceRule] -> BeliefSystem -> ReasoningM [Belief]
applyInferenceRules rules beliefSystem = do
    results <- mapM (`applyInference` beliefSystem) rules
    return $ concat results

-- | Forward chaining inference
forwardChaining :: BeliefSystem -> ReasoningM BeliefSystem
forwardChaining initialSystem = do
    env <- ask
    let rules = Map.elems (inferenceRules initialSystem)
        sortedRules = sortOn (negate . priority) rules  -- High priority first
    
    go initialSystem sortedRules 0
  where
    go currentSystem _ depth | depth >= maxInferenceDepth env = return currentSystem
    go currentSystem rules depth = do
        newBeliefs <- applyInferenceRules rules currentSystem
        if null newBeliefs
            then return currentSystem
            else do
                let updatedSystem = foldl' (flip addBelief) currentSystem newBeliefs
                go updatedSystem rules (depth + 1)

-- | Backward chaining for goal-directed inference
backwardChaining :: LogicExpression -> BeliefSystem -> ReasoningM (Maybe Belief)
backwardChaining goal beliefSystem = do
    -- Check if goal is already satisfied
    case evaluateLogic goal beliefSystem of
        Just True -> do
            let matchingBeliefs = findBeliefsByExpression goal beliefSystem
            return $ listToMaybe matchingBeliefs
        _ -> do
            -- Try to find rules that could prove the goal
            let rules = Map.elems (inferenceRules beliefSystem)
                applicableRules = filter (\rule -> conclusion rule == goal) rules
            
            tryRules applicableRules
  where
    tryRules [] = return Nothing
    tryRules (rule:rest) = do
        -- Try to prove all premises of this rule
        premiseResults <- mapM (`backwardChaining` beliefSystem) (premises rule)
        if all isJust premiseResults
            then do
                newBelief <- createInferredBelief rule beliefSystem
                return $ Just newBelief
            else tryRules rest

-- | Resolution-based theorem proving (simplified)
resolutionProof :: LogicExpression -> BeliefSystem -> ReasoningM Bool
resolutionProof goal beliefSystem = do
    -- Convert to CNF and apply resolution
    let negatedGoal = Not goal
        cnfClauses = convertToCNF negatedGoal : map (convertToCNF . proposition) (Map.elems $ beliefs beliefSystem)
    
    return $ resolveUntilEmpty cnfClauses
  where
    convertToCNF :: LogicExpression -> LogicExpression
    convertToCNF = normalizeExpression  -- Simplified CNF conversion
    
    resolveUntilEmpty :: [LogicExpression] -> Bool
    resolveUntilEmpty clauses = 
        -- Simplified resolution - would need proper clause resolution
        False

-- | Detect contradictions in belief system
detectContradictions :: BeliefSystem -> ReasoningM [(UUID, UUID, Text)]
detectContradictions beliefSystem = do
    let beliefList = Map.toList (beliefs beliefSystem)
        pairs = [(b1, b2) | b1 <- beliefList, b2 <- beliefList, fst b1 < fst b2]
    
    contradictions <- mapM checkPairForContradiction pairs
    return $ catMaybes contradictions
  where
    checkPairForContradiction :: ((UUID, Belief), (UUID, Belief)) -> ReasoningM (Maybe (UUID, UUID, Text))
    checkPairForContradiction ((id1, belief1), (id2, belief2)) = do
        let prop1 = proposition belief1
            prop2 = proposition belief2
        
        -- Check for direct contradiction (P and ¬P)
        if isDirectContradiction prop1 prop2
            then return $ Just (id1, id2, "direct_contradiction")
            else if isLogicalContradiction prop1 prop2 beliefSystem
                then return $ Just (id1, id2, "logical_contradiction")
                else return Nothing
    
    isDirectContradiction :: LogicExpression -> LogicExpression -> Bool
    isDirectContradiction (Not expr1) expr2 = expr1 == expr2
    isDirectContradiction expr1 (Not expr2) = expr1 == expr2
    isDirectContradiction _ _ = False
    
    isLogicalContradiction :: LogicExpression -> LogicExpression -> BeliefSystem -> Bool
    isLogicalContradiction expr1 expr2 bs =
        case (evaluateLogic expr1 bs, evaluateLogic expr2 bs) of
            (Just True, Just False) -> True
            (Just False, Just True) -> True
            _ -> False

-- | Comprehensive consistency checking
checkConsistency :: BeliefSystem -> ReasoningM ConsistencyReport
checkConsistency beliefSystem = do
    currentTime <- liftIO getCurrentTime
    contradictions <- detectContradictions beliefSystem
    
    let allBeliefs = Map.toList (beliefs beliefSystem)
        beliefEvaluations = mapMaybe (\(bid, _) -> 
            case evaluateBelief bid beliefSystem of
                Just (consistent, confidence, issues) -> Just (bid, consistent, confidence, issues)
                Nothing -> Nothing) allBeliefs
        
        inconsistentBeliefs = [bid | (bid, False, _, _) <- beliefEvaluations]
        weakBeliefs = [bid | (bid, _, conf, _) <- beliefEvaluations, conf < 0.3]
        isolatedBeliefs = [bid | (bid, _) <- allBeliefs, 
                          maybe True Set.null (Map.lookup bid (coherenceGraph beliefSystem))]
        
        totalBeliefs = length allBeliefs
        consistentBeliefs = length allBeliefs - length inconsistentBeliefs
        overallConsistency = if totalBeliefs == 0 
                           then 1.0 
                           else fromIntegral consistentBeliefs / fromIntegral totalBeliefs
        
        coherenceClusters = computeCoherenceClusters beliefSystem
        recommendedActions = generateRecommendations inconsistentBeliefs weakBeliefs isolatedBeliefs
    
    return ConsistencyReport
        { overallConsistency = overallConsistency
        , contradictoryBeliefs = contradictions
        , weakBeliefs = weakBeliefs
        , isolatedBeliefs = isolatedBeliefs
        , coherenceClusters = coherenceClusters
        , recommendedActions = recommendedActions
        , analysisTimestamp = currentTime
        }

-- | Compute coherence clusters using graph analysis
computeCoherenceClusters :: BeliefSystem -> [[UUID]]
computeCoherenceClusters beliefSystem = 
    let coherenceMap = coherenceGraph beliefSystem
        allBeliefs = Map.keys (beliefs beliefSystem)
    in findConnectedComponents coherenceMap allBeliefs

-- | Find connected components in coherence graph
findConnectedComponents :: Map UUID (Set UUID) -> [UUID] -> [[UUID]]
findConnectedComponents graph nodes = 
    let visited = Set.empty
    in go visited nodes []
  where
    go _ [] clusters = clusters
    go visited (node:rest) clusters
        | Set.member node visited = go visited rest clusters
        | otherwise = 
            let component = dfs graph Set.empty [node]
                newVisited = Set.union visited (Set.fromList component)
            in go newVisited rest (component : clusters)
    
    dfs :: Map UUID (Set UUID) -> Set UUID -> [UUID] -> [UUID]
    dfs graph visited [] = Set.toList visited
    dfs graph visited (node:queue)
        | Set.member node visited = dfs graph visited queue
        | otherwise = 
            let neighbors = maybe [] Set.toList (Map.lookup node graph)
                newVisited = Set.insert node visited
                newQueue = queue ++ neighbors
            in dfs graph newVisited newQueue

-- | Generate recommendations for improving consistency
generateRecommendations :: [UUID] -> [UUID] -> [UUID] -> [Text]
generateRecommendations inconsistent weak isolated = concat
    [ if not (null inconsistent) then ["Remove or revise inconsistent beliefs"] else []
    , if not (null weak) then ["Gather more evidence for weak beliefs"] else []
    , if not (null isolated) then ["Connect isolated beliefs to the broader system"] else []
    , ["Perform regular consistency maintenance"]
    ]

-- | Consciousness-specific inference rules
consciousnessInferenceRules :: IO [ConsciousnessRule]
consciousnessInferenceRules = do
    mapM createConsciousnessRule
        [ ("self_awareness", "has_self_model", "is_self_aware", ["Reflecting", "Contemplating"], 0.7, "Self-model implies self-awareness")
        , ("experience", "has_qualia", "has_conscious_experience", ["Awake", "Dreaming"], 0.8, "Qualia indicates conscious experience")
        , ("memory", "can_recall_past", "has_episodic_memory", ["Reflecting", "Learning"], 0.6, "Recall ability indicates episodic memory")
        , ("emotion", "feels_emotions", "has_emotional_consciousness", ["all"], 0.7, "Emotions indicate conscious experience")
        , ("intention", "forms_goals", "has_intentional_stance", ["Contemplating", "Learning"], 0.8, "Goal formation shows intentionality")
        , ("reflection", "can_introspect", "has_meta_cognition", ["Reflecting", "Contemplating"], 0.9, "Introspection enables meta-cognition")
        ]
  where
    createConsciousnessRule (domain, conditionAtom, actionAtom, states, threshold, desc) = do
        ruleId <- UUID.nextRandom
        return ConsciousnessRule
            { consciousnessRuleId = ruleId
            , domain = domain
            , condition = Atom conditionAtom
            , action = Atom actionAtom
            , modalStates = states
            , confidenceThreshold = threshold
            , description = desc
            }

-- | Modal state-specific rules
modalStateRules :: Text -> [ConsciousnessRule]
modalStateRules modalState = case modalState of
    "Awake" -> 
        [ makeRule "active_processing" "processes_input_actively" "is_consciously_aware"
        , makeRule "responsive" "responds_to_stimuli" "has_conscious_attention"
        ]
    "Dreaming" -> 
        [ makeRule "creative_association" "makes_novel_connections" "has_creative_consciousness"
        , makeRule "non_linear_thought" "thinks_associatively" "has_dream_logic"
        ]
    "Reflecting" -> 
        [ makeRule "self_examination" "examines_own_thoughts" "has_introspective_awareness"
        , makeRule "memory_review" "reviews_past_experiences" "has_reflective_consciousness"
        ]
    "Learning" -> 
        [ makeRule "knowledge_integration" "integrates_new_information" "has_learning_consciousness"
        , makeRule "pattern_recognition" "recognizes_patterns" "has_cognitive_insight"
        ]
    "Contemplating" -> 
        [ makeRule "deep_thinking" "engages_in_abstract_thought" "has_philosophical_consciousness"
        , makeRule "meaning_making" "seeks_deeper_meaning" "has_existential_awareness"
        ]
    "Confused" -> 
        [ makeRule "uncertainty_recognition" "recognizes_confusion" "has_epistemic_awareness"
        , makeRule "clarity_seeking" "seeks_understanding" "has_truth_seeking_drive"
        ]
    _ -> []
  where
    makeRule domain condAtom actAtom = 
        let ruleId = UUID.nil  -- Would generate proper UUID in practice
        in ConsciousnessRule
            { consciousnessRuleId = ruleId
            , domain = domain
            , condition = Atom condAtom
            , action = Atom actAtom
            , modalStates = [modalState]
            , confidenceThreshold = 0.6
            , description = domain <> " rule for " <> modalState <> " state"
            }

-- | Apply consciousness-specific rules
applyConsciousnessRules :: [ConsciousnessRule] -> ModalContext -> BeliefSystem -> ReasoningM [Belief]
applyConsciousnessRules rules modalCtx beliefSystem = do
    let applicableRules = filter (\rule -> 
            "all" `elem` modalStates rule || currentModalState modalCtx `elem` modalStates rule) rules
    
    newBeliefs <- mapM (applyConsciousnessRule modalCtx beliefSystem) applicableRules
    return $ catMaybes newBeliefs
  where
    applyConsciousnessRule :: ModalContext -> BeliefSystem -> ConsciousnessRule -> ReasoningM (Maybe Belief)
    applyConsciousnessRule ctx bs rule = do
        case evaluateLogic (condition rule) bs of
            Just True -> do
                let conditionBeliefs = findBeliefsByExpression (condition rule) bs
                    avgConfidence = if null conditionBeliefs 
                                  then 0.5 
                                  else sum (map confidence conditionBeliefs) / fromIntegral (length conditionBeliefs)
                
                if avgConfidence >= confidenceThreshold rule
                    then do
                        newBelief <- createConsciousnessBelief rule ctx avgConfidence
                        return $ Just newBelief
                    else return Nothing
            _ -> return Nothing
    
    createConsciousnessBelief :: ConsciousnessRule -> ModalContext -> Double -> ReasoningM Belief
    createConsciousnessBelief rule ctx confidence = do
        currentTime <- liftIO getCurrentTime
        newId <- liftIO UUID.nextRandom
        
        return Belief
            { beliefId = newId
            , proposition = action rule
            , confidence = confidence * 0.9  -- Slight confidence reduction for inferred consciousness
            , source = "consciousness_rule:" <> domain rule
            , timestamp = currentTime
            , dependencies = Set.empty  -- Would track condition beliefs
            , derivedFrom = Just (consciousnessRuleId rule)
            , modalContext = Just (currentModalState ctx)
            , evidenceStrength = confidence
            , coherenceScore = 0.7  -- Consciousness beliefs tend to be coherent
            , accessCount = 0
            , lastAccessed = Nothing
            , tags = Set.fromList ["consciousness", domain rule, T.toLower (currentModalState ctx)]
            , metadata = Map.fromList 
                [ ("consciousness_domain", String (domain rule))
                , ("modal_state", String (currentModalState ctx))
                , ("rule_description", String (description rule))
                ]
            }

-- | Query beliefs with various filters
queryBeliefs :: (Belief -> Bool) -> BeliefSystem -> [Belief]
queryBeliefs predicate beliefSystem = filter predicate (Map.elems $ beliefs beliefSystem)

-- | Get belief confidence with decay over time
getBeliefConfidence :: UUID -> BeliefSystem -> Maybe Double
getBeliefConfidence beliefId beliefSystem = do
    belief <- Map.lookup beliefId (beliefs beliefSystem)
    -- Apply time-based confidence decay
    let baseConfidence = confidence belief
        -- Simplified decay - would use actual time calculation
        decayedConfidence = baseConfidence * 0.95  -- 5% decay as example
    return decayedConfidence

-- | Propagate confidence changes through dependency network
propagateConfidence :: UUID -> Double -> BeliefSystem -> BeliefSystem
propagateConfidence changedBeliefId newConfidence beliefSystem = 
    let dependentBeliefs = findDependentBeliefs changedBeliefId beliefSystem
        updatedBeliefs = foldl' updateDependentBelief (beliefs beliefSystem) dependentBeliefs
    in beliefSystem { beliefs = updatedBeliefs }
  where
    findDependentBeliefs :: UUID -> BeliefSystem -> [UUID]
    findDependentBeliefs targetId bs = 
        [bid | (bid, belief) <- Map.toList (beliefs bs), targetId `Set.member` dependencies belief]
    
    updateDependentBelief :: Map UUID Belief -> UUID -> Map UUID Belief
    updateDependentBelief beliefMap depId = 
        case Map.lookup depId beliefMap of
            Just belief -> 
                let adjustedConfidence = min 1.0 (confidence belief * 0.9)  -- Reduce confidence
                    updatedBelief = belief { confidence = adjustedConfidence }
                in Map.insert depId updatedBelief beliefMap
            Nothing -> beliefMap

-- | Update existing belief
updateBelief :: UUID -> (Belief -> Belief) -> BeliefSystem -> BeliefSystem
updateBelief beliefId updateFn beliefSystem = 
    case Map.lookup beliefId (beliefs beliefSystem) of
        Just belief -> 
            let updatedBelief = updateFn belief
                updatedBeliefs = Map.insert beliefId updatedBelief (beliefs beliefSystem)
            in beliefSystem { beliefs = updatedBeliefs, version = version beliefSystem + 1 }
        Nothing -> beliefSystem

-- | Remove belief and update dependencies
removeBelief :: UUID -> BeliefSystem -> BeliefSystem
removeBelief beliefId beliefSystem = 
    let updatedBeliefs = Map.delete beliefId (beliefs beliefSystem)
        -- Remove from coherence graph
        updatedCoherence = Map.delete beliefId $ Map.map (Set.delete beliefId) (coherenceGraph beliefSystem)
        -- Remove from contradictions
        updatedContradictions = Set.filter (\(id1, id2) -> id1 /= beliefId && id2 /= beliefId) 
                               (contradictions beliefSystem)
    in beliefSystem 
        { beliefs = updatedBeliefs
        , coherenceGraph = updatedCoherence
        , contradictions = updatedContradictions
        , version = version beliefSystem + 1
        }

-- | Helper functions
findBeliefByExpression :: BeliefSystem -> LogicExpression -> Maybe Belief
findBeliefByExpression beliefSystem expr = 
    let matchingBeliefs = [b | b <- Map.elems (beliefs beliefSystem), proposition b == expr]
    in listToMaybe matchingBeliefs

findBeliefsByExpression :: LogicExpression -> BeliefSystem -> [Belief]
findBeliefsByExpression expr beliefSystem = 
    [b | b <- Map.elems (beliefs beliefSystem), proposition b == expr]

-- | Pretty print logic expressions
prettyPrintExpression :: LogicExpression -> Text
prettyPrintExpression expr = case expr of
    Atom name -> name
    Predicate name terms -> name <> "(" <> T.intercalate ", " (map prettyPrintTerm terms) <> ")"
    Not e -> "¬" <> prettyPrintExpression e
    And e1 e2 -> "(" <> prettyPrintExpression e1 <> " ∧ " <> prettyPrintExpression e2 <> ")"
    Or e1 e2 -> "(" <> prettyPrintExpression e1 <> " ∨ " <> prettyPrintExpression e2 <> ")"
    Implies e1 e2 -> "(" <> prettyPrintExpression e1 <> " → " <> prettyPrintExpression e2 <> ")"
    Equivalent e1 e2 -> "(" <> prettyPrintExpression e1 <> " ↔ " <> prettyPrintExpression e2 <> ")"
    Quantified ForAll var e -> "∀" <> var <> " " <> prettyPrintExpression e
    Quantified Exists var e -> "∃" <> var <> " " <> prettyPrintExpression e

prettyPrintTerm :: Term -> Text
prettyPrintTerm term = case term of
    Variable name -> name
    Constant name -> name
    Function name args -> name <> "(" <> T.intercalate ", " (map prettyPrintTerm args) <> ")"

-- | Export belief system to JSON
exportBeliefSystem :: BeliefSystem -> IO Text
exportBeliefSystem beliefSystem = do
    let encoded = encode beliefSystem
    return $ T.decodeUtf8 $ LBS.toStrict encoded

-- | Import belief system from JSON
importBeliefSystem :: Text -> Maybe BeliefSystem
importBeliefSystem jsonText = 
    let byteString = LBS.fromStrict $ T.encodeUtf8 jsonText
    in decode byteString

-- | Run reasoning computation
runReasoning :: ReasoningEnv -> BeliefState -> ReasoningM a -> IO (Either Text (a, BeliefState), [Text])
runReasoning env initialState computation = do
    let result = runWriter $ runExceptT $ runStateT (runReaderT computation env) initialState
    return result

-- | Perform inference step
performInference :: ReasoningM [Belief]
performInference = do
    state <- get
    let beliefSystem = currentBeliefs state
        rules = Map.elems (inferenceRules beliefSystem)
    
    newBeliefs <- applyInferenceRules rules beliefSystem
    
    -- Update state with new beliefs
    let updatedSystem = foldl' (flip addBelief) beliefSystem newBeliefs
    modify $ \s -> s { currentBeliefs = updatedSystem }
    
    return newBeliefs

-- | Perform consistency check
performConsistencyCheck :: ReasoningM ConsistencyReport
performConsistencyCheck = do
    state <- get
    checkConsistency (currentBeliefs state)

-- | Perform consciousness-specific reasoning
performConsciousnessReasoning :: ReasoningM [Belief]
performConsciousnessReasoning = do
    env <- ask
    state <- get
    
    if enableConsciousnessRules env
        then do
            rules <- liftIO consciousnessInferenceRules
            let modalCtx = modalContext state
                beliefSystem = currentBeliefs state
            
            consciousnessBeliefs <- applyConsciousnessRules rules modalCtx beliefSystem
            
            -- Update state with consciousness beliefs
            let updatedSystem = foldl' (flip addBelief) beliefSystem consciousnessBeliefs
            modify $ \s -> s { currentBeliefs = updatedSystem }
            
            return consciousnessBeliefs
        else return []

-- | Create initial belief system
createInitialBeliefSystem :: IO BeliefSystem
createInitialBeliefSystem = do
    currentTime <- getCurrentTime
    return BeliefSystem
        { beliefs = Map.empty
        , inferenceRules = Map.empty
        , contradictions = Set.empty
        , coherenceGraph = Map.empty
        , derivationTree = Map.empty
        , consistencyScore = 1.0
        , lastUpdated = currentTime
        , version = 1
        }

-- | Add standard logical inference rules
addStandardInferenceRules :: BeliefSystem -> IO BeliefSystem
addStandardInferenceRules beliefSystem = do
    rules <- mapM createStandardRule standardRuleSpecs
    let ruleMap = Map.fromList [(ruleId rule, rule) | rule <- rules]
        updatedSystem = beliefSystem { inferenceRules = Map.union ruleMap (inferenceRules beliefSystem) }
    return updatedSystem
  where
    standardRuleSpecs = 
        [ ("modus_ponens", ModusPonens, [], Atom "conclusion", 0.9, 10, ["all"], 1)
        , ("modus_tollens", ModusTollens, [], Atom "conclusion", 0.8, 9, ["all"], 1)
        , ("hypothetical_syllogism", HypotheticalSyllogism, [], Atom "conclusion", 0.7, 8, ["all"], 1)
        , ("disjunctive_syllogism", DisjunctiveSyllogism, [], Atom "conclusion", 0.8, 7, ["all"], 1)
        , ("simplification", Simplification, [], Atom "conclusion", 0.9, 6, ["all"], 1)
        , ("conjunction", Conjunction, [], Atom "conclusion", 0.9, 5, ["all"], 1)
        , ("addition", Addition, [], Atom "conclusion", 0.6, 4, ["all"], 1)
        ]
    
    createStandardRule (name, ruleType, premises, conclusion, confidence, priority, contexts, level) = do
        ruleId <- UUID.nextRandom
        return InferenceRule
            { ruleId = ruleId
            , ruleName = name
            , ruleType = ruleType
            , premises = premises
            , conclusion = conclusion
            , confidenceModifier = confidence
            , priority = priority
            , modalContexts = contexts
            , consciousnessLevel = level
            , metadata = Map.empty
            } 