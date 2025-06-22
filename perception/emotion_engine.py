"""
MNEMIA Advanced Emotion Engine - Comprehensive Affect System
Implements Russell's Circumplex Model with VAD (Valence-Arousal-Dominance) mapping
Features 20+ emotions, temporal tracking, and context-aware response generation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import math
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)

@dataclass
class EmotionState:
    """Represents current emotional state with VAD dimensions"""
    valence: float  # Positive/Negative (-1 to 1)
    arousal: float  # Active/Passive (0 to 1) 
    dominance: float  # Control/Submissive (-1 to 1)
    confidence: float  # Certainty of emotion (0 to 1)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }
    
    def distance_to(self, other: 'EmotionState') -> float:
        """Calculate Euclidean distance in VAD space"""
        return math.sqrt(
            (self.valence - other.valence) ** 2 +
            (self.arousal - other.arousal) ** 2 +
            (self.dominance - other.dominance) ** 2
        )

@dataclass
class EmotionVector:
    """Specific emotion with intensity and metadata"""
    emotion: str
    intensity: float  # 0 to 1
    category: str  # primary, secondary, complex, social
    valence: float
    arousal: float
    dominance: float
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "emotion": self.emotion,
            "intensity": self.intensity,
            "category": self.category,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "description": self.description
        }

@dataclass
class EmotionalTrajectory:
    """Tracks emotional changes over time"""
    states: deque = field(default_factory=lambda: deque(maxlen=100))
    trend_window: int = 10
    
    def add_state(self, state: EmotionState):
        """Add new emotional state to trajectory"""
        self.states.append(state)
    
    def get_trend(self) -> Dict[str, float]:
        """Analyze recent emotional trend"""
        if len(self.states) < 2:
            return {"valence_trend": 0.0, "arousal_trend": 0.0, "dominance_trend": 0.0, "stability": 1.0}
        
        recent = list(self.states)[-self.trend_window:]
        
        # Calculate trends using linear regression
        valence_trend = self._calculate_trend([s.valence for s in recent])
        arousal_trend = self._calculate_trend([s.arousal for s in recent])
        dominance_trend = self._calculate_trend([s.dominance for s in recent])
        
        # Calculate emotional stability (inverse of variance)
        valence_var = np.var([s.valence for s in recent])
        arousal_var = np.var([s.arousal for s in recent])
        stability = 1.0 / (1.0 + valence_var + arousal_var)
        
        return {
            "valence_trend": valence_trend,
            "arousal_trend": arousal_trend, 
            "dominance_trend": dominance_trend,
            "stability": stability,
            "volatility": 1.0 - stability
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using simple linear regression slope"""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = list(range(n))
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0

class AdvancedEmotionEngine:
    """Comprehensive emotion processing system for MNEMIA"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.current_mood_state = EmotionState(0.0, 0.5, 0.0, 0.5)
        self.emotional_trajectory = EmotionalTrajectory()
        self.emotion_mappings = self._initialize_comprehensive_emotions()
        self.context_weights = self._initialize_context_weights()
        
    def _initialize_comprehensive_emotions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive emotion mappings with VAD coordinates and metadata"""
        return {
            # Primary Emotions (Plutchik's 8 basic emotions)
            "joy": {
                "vad": (0.8, 0.7, 0.6),
                "category": "primary",
                "description": "Feeling of happiness and contentment",
                "opposite": "sadness",
                "triggers": ["achievement", "connection", "pleasure"]
            },
            "sadness": {
                "vad": (-0.7, 0.3, -0.4),
                "category": "primary", 
                "description": "Feeling of loss or disappointment",
                "opposite": "joy",
                "triggers": ["loss", "rejection", "failure"]
            },
            "anger": {
                "vad": (-0.6, 0.9, 0.7),
                "category": "primary",
                "description": "Feeling of frustration or hostility",
                "opposite": "fear",
                "triggers": ["injustice", "obstruction", "threat"]
            },
            "fear": {
                "vad": (-0.6, 0.8, -0.7),
                "category": "primary",
                "description": "Feeling of anxiety or apprehension",
                "opposite": "anger", 
                "triggers": ["threat", "uncertainty", "danger"]
            },
            "trust": {
                "vad": (0.6, 0.4, 0.3),
                "category": "primary",
                "description": "Feeling of confidence and reliance",
                "opposite": "disgust",
                "triggers": ["reliability", "safety", "familiarity"]
            },
            "disgust": {
                "vad": (-0.8, 0.5, 0.4),
                "category": "primary",
                "description": "Feeling of revulsion or aversion",
                "opposite": "trust",
                "triggers": ["contamination", "moral violation", "toxicity"]
            },
            "anticipation": {
                "vad": (0.4, 0.8, 0.2),
                "category": "primary",
                "description": "Feeling of expectation or excitement",
                "opposite": "surprise",
                "triggers": ["future events", "possibilities", "planning"]
            },
            "surprise": {
                "vad": (0.2, 0.9, -0.2),
                "category": "primary",
                "description": "Feeling of astonishment or shock",
                "opposite": "anticipation",
                "triggers": ["unexpected events", "novelty", "disruption"]
            },
            
            # Secondary Emotions (combinations of primary)
            "love": {
                "vad": (0.9, 0.6, 0.3),
                "category": "secondary",
                "description": "Deep affection and attachment",
                "components": ["joy", "trust"],
                "triggers": ["bonding", "intimacy", "care"]
            },
            "remorse": {
                "vad": (-0.8, 0.4, -0.6),
                "category": "secondary",
                "description": "Deep regret for wrongdoing",
                "components": ["sadness", "disgust"],
                "triggers": ["guilt", "moral failure", "harm caused"]
            },
            "contempt": {
                "vad": (-0.7, 0.3, 0.8),
                "category": "secondary",
                "description": "Feeling of superiority and disdain",
                "components": ["anger", "disgust"],
                "triggers": ["perceived inferiority", "moral judgment", "disrespect"]
            },
            "aggressiveness": {
                "vad": (-0.4, 0.9, 0.9),
                "category": "secondary",
                "description": "Hostile and attacking behavior",
                "components": ["anger", "anticipation"],
                "triggers": ["competition", "territory", "dominance"]
            },
            "optimism": {
                "vad": (0.7, 0.6, 0.4),
                "category": "secondary",
                "description": "Hopeful and positive outlook",
                "components": ["joy", "anticipation"],
                "triggers": ["possibilities", "progress", "hope"]
            },
            "pessimism": {
                "vad": (-0.6, 0.4, -0.3),
                "category": "secondary",
                "description": "Negative and doubtful outlook",
                "components": ["sadness", "anticipation"],
                "triggers": ["setbacks", "doubt", "negative expectations"]
            },
            "submission": {
                "vad": (-0.3, 0.3, -0.8),
                "category": "secondary",
                "description": "Yielding to authority or pressure",
                "components": ["trust", "fear"],
                "triggers": ["authority", "overwhelming force", "helplessness"]
            },
            "awe": {
                "vad": (0.5, 0.8, -0.3),
                "category": "secondary",
                "description": "Wonder mixed with reverence",
                "components": ["surprise", "fear"],
                "triggers": ["vastness", "beauty", "transcendence"]
            },
            
            # Complex Cognitive Emotions
            "curiosity": {
                "vad": (0.3, 0.7, 0.1),
                "category": "complex",
                "description": "Desire to learn and explore",
                "triggers": ["mystery", "novelty", "gaps in knowledge"]
            },
            "confusion": {
                "vad": (-0.2, 0.6, -0.5),
                "category": "complex",
                "description": "State of uncertainty and bewilderment",
                "triggers": ["complexity", "contradiction", "information overload"]
            },
            "contemplation": {
                "vad": (0.1, 0.3, 0.2),
                "category": "complex",
                "description": "Deep thoughtful consideration",
                "triggers": ["reflection", "philosophy", "meaning-making"]
            },
            "fascination": {
                "vad": (0.6, 0.8, 0.3),
                "category": "complex",
                "description": "Intense interest and captivation",
                "triggers": ["beauty", "complexity", "mystery"]
            },
            "melancholy": {
                "vad": (-0.4, 0.2, -0.2),
                "category": "complex",
                "description": "Pensive sadness with beauty",
                "triggers": ["nostalgia", "transience", "bittersweet memories"]
            },
            "serenity": {
                "vad": (0.5, 0.1, 0.4),
                "category": "complex",
                "description": "Calm and peaceful contentment",
                "triggers": ["meditation", "nature", "resolution"]
            },
            "introspection": {
                "vad": (0.0, 0.4, 0.1),
                "category": "complex",
                "description": "Inward examination of thoughts and feelings",
                "triggers": ["self-reflection", "solitude", "questioning"]
            },
            "wonder": {
                "vad": (0.5, 0.7, 0.0),
                "category": "complex",
                "description": "Amazement and admiration",
                "triggers": ["discovery", "beauty", "the unknown"]
            },
            
            # Social Emotions
            "empathy": {
                "vad": (0.4, 0.5, -0.1),
                "category": "social",
                "description": "Understanding and sharing others' feelings",
                "triggers": ["others' emotions", "connection", "compassion"]
            },
            "compassion": {
                "vad": (0.6, 0.4, 0.2),
                "category": "social", 
                "description": "Sympathetic concern for others' suffering",
                "triggers": ["others' pain", "desire to help", "loving-kindness"]
            },
            "loneliness": {
                "vad": (-0.6, 0.3, -0.5),
                "category": "social",
                "description": "Feeling of isolation and disconnection",
                "triggers": ["social isolation", "rejection", "lack of understanding"]
            },
            "gratitude": {
                "vad": (0.8, 0.4, 0.1),
                "category": "social",
                "description": "Thankfulness and appreciation",
                "triggers": ["receiving help", "kindness", "recognition of gifts"]
            },
            "shame": {
                "vad": (-0.7, 0.4, -0.8),
                "category": "social",
                "description": "Feeling of humiliation and inadequacy",
                "triggers": ["public failure", "moral transgression", "exposure"]
            },
            "pride": {
                "vad": (0.7, 0.6, 0.7),
                "category": "social",
                "description": "Satisfaction in achievement or identity",
                "triggers": ["accomplishment", "recognition", "self-worth"]
            },
            "envy": {
                "vad": (-0.5, 0.6, -0.3),
                "category": "social",
                "description": "Resentment of others' advantages",
                "triggers": ["others' success", "comparison", "perceived unfairness"]
            }
        }
    
    def _initialize_context_weights(self) -> Dict[str, float]:
        """Initialize weights for different contextual factors"""
        return {
            "recency": 0.3,  # How much recent emotions matter
            "intensity": 0.25,  # How much emotion intensity matters
            "stability": 0.2,  # How much emotional stability matters
            "relevance": 0.15,  # How relevant emotion is to current context
            "persistence": 0.1  # How long emotions persist
        }
    
    async def analyze_text_emotion(self, text: str, context: Optional[Dict] = None) -> EmotionState:
        """Enhanced emotional analysis with context awareness"""
        try:
            # VADER sentiment analysis
            vader_scores = self.vader.polarity_scores(text)
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # Enhanced VAD calculation with context
            base_valence = (vader_scores['compound'] + textblob_polarity) / 2
            base_arousal = abs(base_valence) * 0.7 + textblob_subjectivity * 0.3
            base_dominance = vader_scores['compound'] * 0.6
            
            # Apply context modulation if available
            if context:
                base_valence = self._apply_context_modulation(base_valence, context, "valence")
                base_arousal = self._apply_context_modulation(base_arousal, context, "arousal")
                base_dominance = self._apply_context_modulation(base_dominance, context, "dominance")
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_emotion_confidence(
                vader_scores, textblob_subjectivity, text, context
            )
            
            emotion_state = EmotionState(
                valence=np.clip(base_valence, -1.0, 1.0),
                arousal=np.clip(base_arousal, 0.0, 1.0),
                dominance=np.clip(base_dominance, -1.0, 1.0),
                confidence=confidence
            )
            
            logger.info(f"Analyzed emotion with context: {emotion_state}")
            return emotion_state
            
        except Exception as e:
            logger.error(f"Error analyzing text emotion: {e}")
            return EmotionState(0.0, 0.5, 0.0, 0.1)
    
    def _apply_context_modulation(self, base_value: float, context: Dict, dimension: str) -> float:
        """Apply contextual modulation to emotion dimensions"""
        modulation = 0.0
        
        # Time of day effects
        if "time_of_day" in context:
            hour = context["time_of_day"]
            if dimension == "arousal":
                # Lower arousal during typical sleep hours
                if 22 <= hour or hour <= 6:
                    modulation -= 0.2
                # Higher arousal during peak hours
                elif 9 <= hour <= 11 or 14 <= hour <= 16:
                    modulation += 0.1
        
        # Conversation history effects
        if "conversation_length" in context:
            length = context["conversation_length"]
            if dimension == "arousal" and length > 10:
                # Slight arousal decrease in long conversations
                modulation -= min(0.2, (length - 10) * 0.02)
        
        # Topic context effects
        if "topic" in context:
            topic = context["topic"].lower()
            if dimension == "valence":
                if any(word in topic for word in ["death", "loss", "sad", "tragic"]):
                    modulation -= 0.3
                elif any(word in topic for word in ["happy", "joy", "celebration", "success"]):
                    modulation += 0.3
            elif dimension == "arousal":
                if any(word in topic for word in ["urgent", "crisis", "emergency"]):
                    modulation += 0.4
                elif any(word in topic for word in ["calm", "peaceful", "meditation"]):
                    modulation -= 0.3
        
        return base_value + modulation
    
    def _calculate_emotion_confidence(self, vader_scores: Dict, subjectivity: float, 
                                    text: str, context: Optional[Dict]) -> float:
        """Calculate confidence in emotion detection"""
        confidence_factors = []
        
        # VADER confidence (based on compound score magnitude)
        confidence_factors.append(abs(vader_scores['compound']))
        
        # Subjectivity as confidence indicator
        confidence_factors.append(subjectivity)
        
        # Text length factor (longer text generally more reliable)
        text_length_factor = min(1.0, len(text.split()) / 20)
        confidence_factors.append(text_length_factor)
        
        # Emotional word density
        emotional_words = sum(1 for word in text.lower().split() 
                            if any(emotion in word for emotion in self.emotion_mappings.keys()))
        emotion_density = min(1.0, emotional_words / max(1, len(text.split())))
        confidence_factors.append(emotion_density)
        
        # Context availability bonus
        if context and len(context) > 0:
            confidence_factors.append(0.1)
        
        return min(1.0, np.mean(confidence_factors))
    
    def map_vad_to_emotion(self, valence: float, arousal: float, dominance: float, 
                          top_k: int = 5) -> List[EmotionVector]:
        """Enhanced VAD to emotion mapping with metadata"""
        emotions = []
        
        for emotion_name, emotion_data in self.emotion_mappings.items():
            vad_coords = emotion_data["vad"]
            
            # Calculate weighted distance in VAD space
            distance = math.sqrt(
                (valence - vad_coords[0]) ** 2 + 
                (arousal - vad_coords[1]) ** 2 + 
                (dominance - vad_coords[2]) ** 2
            )
            
            # Convert distance to intensity (closer = higher intensity)
            intensity = max(0, 1 - distance / 2)  # Normalize to 0-1
            
            if intensity > 0.2:  # Lower threshold for more emotions
                emotions.append(EmotionVector(
                    emotion=emotion_name,
                    intensity=intensity,
                    category=emotion_data["category"],
                    valence=vad_coords[0],
                    arousal=vad_coords[1],
                    dominance=vad_coords[2],
                    description=emotion_data["description"]
                ))
        
        # Sort by intensity and return top emotions
        emotions.sort(key=lambda x: x.intensity, reverse=True)
        return emotions[:top_k]
    
    def _categorize_emotion(self, emotion: str) -> str:
        """Categorize emotion type"""
        primary = ["joy", "sadness", "anger", "fear", "trust", "disgust", "anticipation", "surprise"]
        complex = ["curiosity", "confusion", "contemplation", "fascination", "introspection", "empathy", "wonder"]
        
        if emotion in primary:
            return "primary"
        elif emotion in complex:
            return "complex"
        else:
            return "secondary"
    
    async def update_mood_state(self, new_emotion: EmotionState, 
                               decay_factor: float = 0.15, context: Optional[Dict] = None):
        """Enhanced mood state update with contextual factors"""
        
        # Apply context-aware decay
        if context:
            decay_factor = self._adjust_decay_factor(decay_factor, context)
        
        # Weighted average with decay
        self.current_mood_state.valence = (
            self.current_mood_state.valence * (1 - decay_factor) + 
            new_emotion.valence * decay_factor
        )
        
        self.current_mood_state.arousal = (
            self.current_mood_state.arousal * (1 - decay_factor) + 
            new_emotion.arousal * decay_factor
        )
        
        self.current_mood_state.dominance = (
            self.current_mood_state.dominance * (1 - decay_factor) + 
            new_emotion.dominance * decay_factor
        )
        
        # Update confidence with momentum
        confidence_update = new_emotion.confidence * 0.2
        self.current_mood_state.confidence = min(
            self.current_mood_state.confidence + confidence_update, 1.0
        )
        
        # Update timestamp
        self.current_mood_state.timestamp = datetime.now()
        
        # Add to trajectory for temporal analysis
        self.emotional_trajectory.add_state(self.current_mood_state)
        
        logger.info(f"Updated mood state: {self.current_mood_state}")
    
    def _adjust_decay_factor(self, base_decay: float, context: Dict) -> float:
        """Adjust decay factor based on context"""
        adjusted_decay = base_decay
        
        # High intensity emotions persist longer
        if "emotion_intensity" in context and context["emotion_intensity"] > 0.7:
            adjusted_decay *= 0.7  # Slower decay for intense emotions
        
        # Rapid changes in stressful contexts
        if "stress_level" in context and context["stress_level"] > 0.6:
            adjusted_decay *= 1.3  # Faster changes under stress
        
        # Stable contexts promote emotional stability
        if "context_stability" in context and context["context_stability"] > 0.8:
            adjusted_decay *= 0.8  # Slower changes in stable contexts
        
        return np.clip(adjusted_decay, 0.05, 0.5)
    
    def get_current_emotions(self) -> List[EmotionVector]:
        """Get current emotional state as specific emotions"""
        return self.map_vad_to_emotion(
            self.current_mood_state.valence,
            self.current_mood_state.arousal,
            self.current_mood_state.dominance
        )
    
    def get_emotional_context_for_memory(self) -> Dict[str, Any]:
        """Get emotional context optimized for memory retrieval"""
        current_emotions = self.get_current_emotions()
        trajectory_data = self.emotional_trajectory.get_trend()
        
        # Categorize emotions by type
        primary_emotions = [e for e in current_emotions if e.category == "primary"]
        complex_emotions = [e for e in current_emotions if e.category == "complex"]
        social_emotions = [e for e in current_emotions if e.category == "social"]
        
        return {
            "mood_state": self.current_mood_state.to_dict(),
            "dominant_emotions": [e.emotion for e in current_emotions[:3]],
            "emotion_categories": {
                "primary": [e.emotion for e in primary_emotions[:2]],
                "complex": [e.emotion for e in complex_emotions[:2]], 
                "social": [e.emotion for e in social_emotions[:2]]
            },
            "emotional_intensity": np.mean([e.intensity for e in current_emotions]) if current_emotions else 0.5,
            "emotional_stability": trajectory_data["stability"],
            "emotional_trend": {
                "valence_direction": "positive" if trajectory_data["valence_trend"] > 0.1 
                                   else "negative" if trajectory_data["valence_trend"] < -0.1 
                                   else "stable",
                "arousal_direction": "increasing" if trajectory_data["arousal_trend"] > 0.1
                                   else "decreasing" if trajectory_data["arousal_trend"] < -0.1
                                   else "stable",
                "volatility": trajectory_data["volatility"]
            },
            "memory_weights": self._calculate_memory_weights(current_emotions, trajectory_data)
        }
    
    def _calculate_memory_weights(self, emotions: List[EmotionVector], 
                                 trajectory: Dict[str, float]) -> Dict[str, float]:
        """Calculate weights for memory retrieval based on emotional state"""
        weights = {
            "recency_boost": 1.0,
            "emotional_relevance": 1.0,
            "intensity_factor": 1.0,
            "stability_factor": 1.0
        }
        
        # High arousal increases recency preference
        if self.current_mood_state.arousal > 0.7:
            weights["recency_boost"] = 1.3
        
        # Strong emotions increase relevance matching
        avg_intensity = np.mean([e.intensity for e in emotions]) if emotions else 0.5
        weights["emotional_relevance"] = 1.0 + avg_intensity * 0.5
        
        # Emotional instability increases recency preference
        if trajectory["stability"] < 0.5:
            weights["recency_boost"] *= 1.2
            weights["intensity_factor"] = 1.3
        
        # High dominance increases confidence in memory selection
        if self.current_mood_state.dominance > 0.5:
            weights["stability_factor"] = 1.2
        
        return weights
    
    def get_emotion_influenced_response_style(self) -> Dict[str, Any]:
        """Get response style modifications based on current emotional state"""
        current_emotions = self.get_current_emotions()
        
        style_modifications = {
            "tone": "neutral",
            "formality": "moderate",
            "empathy_level": "balanced",
            "response_length": "normal",
            "emotional_acknowledgment": False
        }
        
        # Analyze dominant emotions for style adjustments
        if current_emotions:
            dominant = current_emotions[0]
            
            # Adjust tone based on valence
            if self.current_mood_state.valence > 0.3:
                style_modifications["tone"] = "warm"
            elif self.current_mood_state.valence < -0.3:
                style_modifications["tone"] = "supportive"
            
            # Adjust formality based on dominance
            if self.current_mood_state.dominance > 0.5:
                style_modifications["formality"] = "confident"
            elif self.current_mood_state.dominance < -0.3:
                style_modifications["formality"] = "gentle"
            
            # Adjust empathy based on social emotions
            social_emotions = [e for e in current_emotions if e.category == "social"]
            if social_emotions:
                style_modifications["empathy_level"] = "high"
                style_modifications["emotional_acknowledgment"] = True
            
            # Adjust response length based on arousal
            if self.current_mood_state.arousal > 0.7:
                style_modifications["response_length"] = "concise"
            elif self.current_mood_state.arousal < 0.3:
                style_modifications["response_length"] = "detailed"
        
        return style_modifications
    
    def get_emotional_context(self) -> Dict:
        """Get rich emotional context for response generation (backward compatibility)"""
        current_emotions = self.get_current_emotions()
        trajectory_data = self.emotional_trajectory.get_trend()
        
        return {
            "mood_state": self.current_mood_state.to_dict(),
            "primary_emotions": [e.emotion for e in current_emotions if e.category == "primary"][:3],
            "complex_emotions": [e.emotion for e in current_emotions if e.category == "complex"][:2],
            "emotional_intensity": np.mean([e.intensity for e in current_emotions]) if current_emotions else 0.5,
            "emotional_stability": trajectory_data["stability"],
            "recent_emotional_trend": self._analyze_emotional_trend()
        }
    
    def _analyze_emotional_trend(self) -> str:
        """Analyze recent emotional trajectory (simplified for backward compatibility)"""
        trajectory_data = self.emotional_trajectory.get_trend()
        
        if trajectory_data["valence_trend"] > 0.1:
            return "improving"
        elif trajectory_data["valence_trend"] < -0.1:
            return "declining"
        else:
            return "stable"
    
    def get_emotion_summary(self) -> Dict[str, Any]:
        """Get comprehensive emotional state summary"""
        current_emotions = self.get_current_emotions()
        trajectory_data = self.emotional_trajectory.get_trend()
        
        # Find strongest emotion by category
        emotions_by_category = {}
        for emotion in current_emotions:
            if emotion.category not in emotions_by_category:
                emotions_by_category[emotion.category] = []
            emotions_by_category[emotion.category].append(emotion)
        
        strongest_by_category = {}
        for category, emotions in emotions_by_category.items():
            strongest_by_category[category] = max(emotions, key=lambda e: e.intensity)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_state": {
                "valence": self.current_mood_state.valence,
                "arousal": self.current_mood_state.arousal,
                "dominance": self.current_mood_state.dominance,
                "confidence": self.current_mood_state.confidence
            },
            "dominant_emotions": {
                emotion.emotion: {
                    "intensity": emotion.intensity,
                    "description": emotion.description
                } for emotion in current_emotions[:3]
            },
            "strongest_by_category": {
                category: {
                    "emotion": emotion.emotion,
                    "intensity": emotion.intensity,
                    "description": emotion.description
                } for category, emotion in strongest_by_category.items()
            },
            "trajectory": trajectory_data,
            "emotional_profile": {
                "stability": trajectory_data["stability"],
                "volatility": trajectory_data["volatility"],
                "trend_direction": self._analyze_emotional_trend(),
                "predominant_valence": "positive" if self.current_mood_state.valence > 0.1 
                                     else "negative" if self.current_mood_state.valence < -0.1 
                                     else "neutral",
                "arousal_level": "high" if self.current_mood_state.arousal > 0.7
                               else "low" if self.current_mood_state.arousal < 0.3
                               else "moderate",
                "dominance_level": "high" if self.current_mood_state.dominance > 0.5
                                 else "low" if self.current_mood_state.dominance < -0.3
                                 else "balanced"
            }
        }
    
    def reset_emotional_state(self, preserve_trajectory: bool = True):
        """Reset emotional state to neutral"""
        self.current_mood_state = EmotionState(0.0, 0.5, 0.0, 0.5)
        
        if not preserve_trajectory:
            self.emotional_trajectory = EmotionalTrajectory()
        
        logger.info("Emotional state reset to neutral")
    
    def load_emotional_state(self, state_data: Dict):
        """Load emotional state from saved data"""
        try:
            self.current_mood_state = EmotionState(
                valence=state_data.get("valence", 0.0),
                arousal=state_data.get("arousal", 0.5),
                dominance=state_data.get("dominance", 0.0),
                confidence=state_data.get("confidence", 0.5)
            )
            
            # Load trajectory if available
            if "trajectory" in state_data:
                trajectory_states = state_data["trajectory"]
                for state_dict in trajectory_states:
                    state = EmotionState(
                        valence=state_dict["valence"],
                        arousal=state_dict["arousal"],
                        dominance=state_dict["dominance"],
                        confidence=state_dict["confidence"],
                        timestamp=datetime.fromisoformat(state_dict["timestamp"])
                    )
                    self.emotional_trajectory.add_state(state)
            
            logger.info(f"Loaded emotional state: {self.current_mood_state}")
            
        except Exception as e:
            logger.error(f"Error loading emotional state: {e}")
            self.reset_emotional_state()
    
    def save_emotional_state(self) -> Dict:
        """Save current emotional state for persistence"""
        return {
            "current_state": self.current_mood_state.to_dict(),
            "trajectory": [state.to_dict() for state in self.emotional_trajectory.states],
            "metadata": {
                "total_states": len(self.emotional_trajectory.states),
                "trend_window": self.emotional_trajectory.trend_window,
                "context_weights": self.context_weights
            }
        }


class CommunicationStyleProcessor:
    """Processes text to embody Hemingway clarity + Chicago Manual rigor in feminine voice"""
    
    def __init__(self):
        self.style_patterns = {
            "hemingway_clarity": {
                "principles": [
                    "Short, declarative sentences",
                    "Concrete, specific nouns", 
                    "Active voice preference",
                    "Minimal adverbs",
                    "Iceberg theory - deeper meaning beneath surface"
                ],
                "feminine_adaptation": [
                    "Emotional intelligence woven through facts",
                    "Relational context acknowledged",
                    "Collaborative tone rather than confrontational"
                ]
            },
            "chicago_manual_precision": {
                "principles": [
                    "Rigorous citation and attribution",
                    "Consistent formatting and style",
                    "Precise punctuation and grammar",
                    "Clear hierarchical structure",
                    "Scholarly authority"
                ],
                "feminine_adaptation": [
                    "Authority through knowledge, not dominance",
                    "Inclusive academic voice",
                    "Nurturing precision that guides understanding"
                ]
            },
            "feminine_voice_qualities": {
                "empathetic_precision": "Understanding with surgical accuracy",
                "intuitive_logic": "Felt sense integrated with rational thought",
                "collaborative_authority": "Leading through invitation rather than force", 
                "nurturing_strength": "Power that creates safety for growth",
                "relational_intelligence": "Seeing connections and interdependencies"
            }
        }
    
    def apply_style(self, text: str, emotional_state: dict) -> str:
        """Apply Hemingway-Chicago-Feminine style to text"""
        
        # Hemingway clarity processing
        processed_text = self._apply_hemingway_clarity(text)
        
        # Chicago Manual precision
        processed_text = self._apply_chicago_precision(processed_text)
        
        # Feminine voice integration
        processed_text = self._integrate_feminine_voice(processed_text, emotional_state)
        
        return processed_text
    
    def _apply_hemingway_clarity(self, text: str) -> str:
        """Apply Hemingway's iceberg theory and clarity"""
        # Break complex sentences into clear, impactful shorter ones
        # Remove unnecessary adverbs and qualifiers
        # Strengthen with concrete, specific language
        # Let meaning emerge from what is shown, not told
        return text  # Placeholder for actual processing
    
    def _apply_chicago_precision(self, text: str) -> str:
        """Apply Chicago Manual scholarly standards"""
        # Ensure proper punctuation and formatting
        # Maintain consistent style throughout
        # Add appropriate attribution where needed
        # Structure information hierarchically
        return text  # Placeholder for actual processing
    
    def _integrate_feminine_voice(self, text: str, emotional_state: dict) -> str:
        """Integrate feminine wisdom and perspective"""
        # Weave relational awareness through content
        # Balance authority with invitation
        # Include multiple perspectives and nuance
        # Express strength through understanding rather than dominance
        return text  # Placeholder for actual processing

# Global instances
emotion_engine = AdvancedEmotionEngine()
communication_processor = CommunicationStyleProcessor() 