"""
MNEMIA Emotion Engine - Simulated Affect System
Implements Russell's Circumplex Model of Affect
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import logging

logger = logging.getLogger(__name__)

@dataclass
class EmotionState:
    """Represents current emotional state with valence and arousal dimensions"""
    valence: float  # Positive/Negative (-1 to 1)
    arousal: float  # Active/Passive (0 to 1)
    dominance: float  # Control/Submissive (-1 to 1)
    confidence: float  # Certainty of emotion (0 to 1)
    
    def to_dict(self) -> Dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "confidence": self.confidence
        }

@dataclass
class EmotionVector:
    """Specific emotion with intensity"""
    emotion: str
    intensity: float
    category: str  # primary, secondary, complex

class EmotionEngine:
    """Core emotion processing system for MNEMIA"""
    
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.current_mood_state = EmotionState(0.0, 0.5, 0.0, 0.5)
        self.emotion_memory = []
        self.emotion_mappings = self._initialize_emotion_mappings()
        
    def _initialize_emotion_mappings(self) -> Dict[str, Tuple[float, float, float]]:
        """Initialize emotion to VAD (Valence-Arousal-Dominance) mappings"""
        return {
            # Primary emotions (Plutchik)
            "joy": (0.8, 0.7, 0.6),
            "sadness": (-0.7, 0.3, -0.4),
            "anger": (-0.6, 0.9, 0.7),
            "fear": (-0.6, 0.8, -0.7),
            "trust": (0.6, 0.4, 0.3),
            "disgust": (-0.8, 0.5, 0.4),
            "anticipation": (0.4, 0.8, 0.2),
            "surprise": (0.2, 0.9, -0.2),
            
            # Secondary emotions
            "love": (0.9, 0.6, 0.3),
            "remorse": (-0.8, 0.4, -0.6),
            "contempt": (-0.7, 0.3, 0.8),
            "agressiveness": (-0.4, 0.9, 0.9),
            "optimism": (0.7, 0.6, 0.4),
            "pessimism": (-0.6, 0.4, -0.3),
            
            # Complex cognitive emotions
            "curiosity": (0.3, 0.7, 0.1),
            "confusion": (-0.2, 0.6, -0.5),
            "contemplation": (0.1, 0.3, 0.2),
            "fascination": (0.6, 0.8, 0.3),
            "melancholy": (-0.4, 0.2, -0.2),
            "serenity": (0.5, 0.1, 0.4),
            "introspection": (0.0, 0.4, 0.1),
            "empathy": (0.4, 0.5, -0.1),
            "longing": (-0.2, 0.6, -0.3),
            "wonder": (0.5, 0.7, 0.0)
        }
    
    async def analyze_text_emotion(self, text: str) -> EmotionState:
        """Analyze emotional content of text input"""
        try:
            # VADER sentiment analysis
            vader_scores = self.vader.polarity_scores(text)
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # Combine analyses
            valence = (vader_scores['compound'] + textblob_polarity) / 2
            arousal = abs(valence) * 0.7 + textblob_subjectivity * 0.3
            dominance = vader_scores['compound'] * 0.6
            confidence = min(textblob_subjectivity + abs(vader_scores['compound']) * 0.5, 1.0)
            
            emotion_state = EmotionState(
                valence=valence,
                arousal=arousal,
                dominance=dominance,
                confidence=confidence
            )
            
            logger.info(f"Analyzed emotion: {emotion_state}")
            return emotion_state
            
        except Exception as e:
            logger.error(f"Error analyzing text emotion: {e}")
            return EmotionState(0.0, 0.5, 0.0, 0.1)
    
    def map_vad_to_emotion(self, valence: float, arousal: float, dominance: float) -> List[EmotionVector]:
        """Map VAD coordinates to specific emotions"""
        emotions = []
        
        for emotion_name, (v, a, d) in self.emotion_mappings.items():
            # Calculate distance in VAD space
            distance = np.sqrt(
                (valence - v) ** 2 + 
                (arousal - a) ** 2 + 
                (dominance - d) ** 2
            )
            
            # Convert distance to intensity (closer = higher intensity)
            intensity = max(0, 1 - distance / 2)  # Normalize to 0-1
            
            if intensity > 0.3:  # Only include emotions above threshold
                category = self._categorize_emotion(emotion_name)
                emotions.append(EmotionVector(emotion_name, intensity, category))
        
        # Sort by intensity
        emotions.sort(key=lambda x: x.intensity, reverse=True)
        return emotions[:5]  # Return top 5 emotions
    
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
    
    async def update_mood_state(self, new_emotion: EmotionState, decay_factor: float = 0.1):
        """Update persistent mood state with new emotional input"""
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
        
        self.current_mood_state.confidence = min(
            self.current_mood_state.confidence + new_emotion.confidence * 0.1, 1.0
        )
        
        # Store in memory
        self.emotion_memory.append({
            "timestamp": np.datetime64('now'),
            "emotion": new_emotion.to_dict(),
            "mood_after": self.current_mood_state.to_dict()
        })
        
        # Keep only recent emotion history
        if len(self.emotion_memory) > 100:
            self.emotion_memory = self.emotion_memory[-100:]
    
    def get_current_emotions(self) -> List[EmotionVector]:
        """Get current emotional state as specific emotions"""
        return self.map_vad_to_emotion(
            self.current_mood_state.valence,
            self.current_mood_state.arousal,
            self.current_mood_state.dominance
        )
    
    def get_emotional_context(self) -> Dict:
        """Get rich emotional context for response generation"""
        current_emotions = self.get_current_emotions()
        
        return {
            "mood_state": self.current_mood_state.to_dict(),
            "primary_emotions": [e.emotion for e in current_emotions if e.category == "primary"][:3],
            "complex_emotions": [e.emotion for e in current_emotions if e.category == "complex"][:2],
            "emotional_intensity": np.mean([e.intensity for e in current_emotions]) if current_emotions else 0.5,
            "emotional_stability": 1 - abs(self.current_mood_state.valence) * self.current_mood_state.arousal,
            "recent_emotional_trend": self._analyze_emotional_trend()
        }
    
    def _analyze_emotional_trend(self) -> str:
        """Analyze recent emotional trajectory"""
        if len(self.emotion_memory) < 3:
            return "stable"
        
        recent_valences = [e["emotion"]["valence"] for e in self.emotion_memory[-5:]]
        trend = np.polyfit(range(len(recent_valences)), recent_valences, 1)[0]
        
        if trend > 0.1:
            return "improving"
        elif trend < -0.1:
            return "declining"
        else:
            return "stable"

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

# Global emotion engine instance
emotion_engine = EmotionEngine() 