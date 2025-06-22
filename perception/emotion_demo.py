#!/usr/bin/env python3
"""
MNEMIA Emotional Intelligence Demo
Showcases comprehensive VAD modeling, 25+ emotions, temporal tracking, and context integration
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

try:
    from emotion_engine import AdvancedEmotionEngine, EmotionState, EmotionVector
    from memory_guided_response import MemoryGuidedResponseGenerator
except ImportError:
    print("Note: Running in demo mode without full MNEMIA integration")
    AdvancedEmotionEngine = None
    MemoryGuidedResponseGenerator = None

class EmotionalIntelligenceDemo:
    """Comprehensive demonstration of MNEMIA's emotional intelligence capabilities"""
    
    def __init__(self):
        if AdvancedEmotionEngine:
            self.emotion_engine = AdvancedEmotionEngine()
            self.memory_generator = MemoryGuidedResponseGenerator()
        else:
            self.emotion_engine = None
            self.memory_generator = None
            
        self.demo_scenarios = self._initialize_demo_scenarios()
        
    def _initialize_demo_scenarios(self) -> List[Dict[str, Any]]:
        """Initialize diverse emotional scenarios for testing"""
        return [
            {
                "name": "Joy and Achievement",
                "text": "I just got accepted into my dream university! I can't believe it happened. All those years of hard work finally paid off.",
                "context": {"topic": "achievement", "time_of_day": 10, "conversation_length": 2},
                "expected_emotions": ["joy", "pride", "optimism", "gratitude"]
            },
            {
                "name": "Loss and Grief",
                "text": "My grandmother passed away last night. She was the most important person in my life. I don't know how to cope with this emptiness.",
                "context": {"topic": "loss", "time_of_day": 22, "conversation_length": 1},
                "expected_emotions": ["sadness", "grief", "loneliness", "love"]
            },
            {
                "name": "Philosophical Contemplation",
                "text": "I've been thinking about the nature of consciousness lately. What does it mean to truly understand something? Are we just biological machines, or is there something more?",
                "context": {"topic": "philosophy", "time_of_day": 14, "conversation_length": 5},
                "expected_emotions": ["contemplation", "curiosity", "wonder", "introspection"]
            },
            {
                "name": "Social Anxiety",
                "text": "I have to give a presentation tomorrow in front of 100 people. My heart is racing just thinking about it. What if I mess up? What if they judge me?",
                "context": {"topic": "anxiety", "time_of_day": 23, "conversation_length": 3},
                "expected_emotions": ["fear", "anxiety", "shame", "anticipation"]
            },
            {
                "name": "Romantic Love",
                "text": "When I look into her eyes, the whole world disappears. I've never felt this way before. She makes me want to be a better person.",
                "context": {"topic": "love", "time_of_day": 19, "conversation_length": 4},
                "expected_emotions": ["love", "joy", "fascination", "gratitude"]
            },
            {
                "name": "Moral Conflict",
                "text": "I saw my friend cheating on the exam. I want to report it because it's wrong, but I don't want to betray our friendship. I feel terrible either way.",
                "context": {"topic": "moral dilemma", "time_of_day": 16, "conversation_length": 6},
                "expected_emotions": ["confusion", "guilt", "contempt", "empathy"]
            },
            {
                "name": "Creative Inspiration",
                "text": "The sunset over the mountains was breathtaking. Colors I've never seen before painted across the sky. I need to capture this feeling somehow.",
                "context": {"topic": "beauty", "time_of_day": 18, "conversation_length": 2},
                "expected_emotions": ["awe", "fascination", "serenity", "wonder"]
            },
            {
                "name": "Workplace Frustration",
                "text": "My boss keeps taking credit for my work. I've been putting in 60-hour weeks, and he presents my ideas as his own. This is so unfair!",
                "context": {"topic": "injustice", "time_of_day": 17, "conversation_length": 8},
                "expected_emotions": ["anger", "frustration", "contempt", "sadness"]
            }
        ]
    
    async def run_comprehensive_demo(self):
        """Run complete emotional intelligence demonstration"""
        print("🧠 MNEMIA Emotional Intelligence Demo")
        print("=" * 50)
        print("Features: VAD Model, 25+ Emotions, Temporal Tracking, Context Integration")
        print()
        
        if not self.emotion_engine:
            print("⚠️  Running in standalone demo mode")
            self._run_conceptual_demo()
            return
        
        # Run full integrated demo
        await self._run_integrated_demo()
    
    def _run_conceptual_demo(self):
        """Run conceptual demo showing capabilities"""
        print("📋 MNEMIA Emotional Intelligence Capabilities")
        print("-" * 50)
        
        print("\n1. VAD (Valence-Arousal-Dominance) Model")
        print("   • Valence: Positive/Negative emotional tone (-1 to +1)")
        print("   • Arousal: Activation level (0 to 1)")
        print("   • Dominance: Control/Submission (-1 to +1)")
        
        print("\n2. Comprehensive Emotion Recognition (25+ Emotions)")
        emotions_by_category = {
            "Primary": ["joy", "sadness", "anger", "fear", "trust", "disgust", "anticipation", "surprise"],
            "Secondary": ["love", "remorse", "contempt", "aggressiveness", "optimism", "pessimism", "submission", "awe"],
            "Complex": ["curiosity", "confusion", "contemplation", "fascination", "melancholy", "serenity", "introspection", "wonder"],
            "Social": ["empathy", "compassion", "loneliness", "gratitude", "shame", "pride", "envy"]
        }
        
        for category, emotions in emotions_by_category.items():
            print(f"   {category} ({len(emotions)}): {', '.join(emotions[:5])}" + ("..." if len(emotions) > 5 else ""))
        
        print(f"\n   Total: {sum(len(emotions) for emotions in emotions_by_category.values())} emotions")
        
        print("\n3. Temporal Emotional Trajectory")
        print("   • Linear regression trend analysis")
        print("   • Emotional stability measurement")
        print("   • Volatility tracking")
        print("   • 100 state history buffer")
        
        print("\n4. Context Integration")
        print("   • Time of day effects")
        print("   • Conversation length adaptation")
        print("   • Topic-based modulation")
        print("   • Stress level considerations")
        
        print("\n5. Memory Integration")
        print("   • Emotional relevance weighting")
        print("   • VAD distance calculations")
        print("   • Recency boost for high arousal")
        print("   • Stability-based threshold adjustment")
        
        print("\n6. Communication Style Adaptation")
        print("   • Tone adjustment (warm/supportive/neutral)")
        print("   • Formality modulation (gentle/confident/moderate)")
        print("   • Empathy level scaling")
        print("   • Response length optimization")
        
        print("\n📊 Demo Scenarios:")
        for i, scenario in enumerate(self.demo_scenarios, 1):
            print(f"   {i}. {scenario['name']}")
            print(f"      Expected: {', '.join(scenario['expected_emotions'])}")
    
    async def _run_integrated_demo(self):
        """Run full integrated demo with emotion engine"""
        # 1. Basic VAD Analysis Demo
        await self._demo_vad_analysis()
        
        # 2. Emotion Recognition Demo
        await self._demo_emotion_recognition()
        
        # 3. Temporal Tracking Demo
        await self._demo_temporal_tracking()
        
        # 4. Context Integration Demo
        await self._demo_context_integration()
        
        print("\n✅ Integrated demo completed successfully!")
    
    async def _demo_vad_analysis(self):
        """Demonstrate VAD (Valence-Arousal-Dominance) analysis"""
        print("\n1. VAD Analysis Demo")
        print("-" * 30)
        
        test_phrases = [
            "I love this beautiful sunny day!",
            "I'm terrified of what might happen.",
            "This is so boring and pointless.",
            "I feel calm and peaceful inside."
        ]
        
        for phrase in test_phrases:
            emotion_state = await self.emotion_engine.analyze_text_emotion(phrase)
            print(f"\nText: '{phrase}'")
            print(f"Valence: {emotion_state.valence:.3f} (negative ← → positive)")
            print(f"Arousal:  {emotion_state.arousal:.3f} (calm ← → excited)")
            print(f"Dominance: {emotion_state.dominance:.3f} (submissive ← → dominant)")
            print(f"Confidence: {emotion_state.confidence:.3f}")
    
    async def _demo_emotion_recognition(self):
        """Demonstrate comprehensive emotion recognition"""
        print("\n2. Emotion Recognition Demo (25+ Emotions)")
        print("-" * 45)
        
        for scenario in self.demo_scenarios[:4]:  # First 4 scenarios
            print(f"\n📝 Scenario: {scenario['name']}")
            print(f"Text: {scenario['text'][:100]}...")
            
            # Analyze emotions
            emotion_state = await self.emotion_engine.analyze_text_emotion(
                scenario['text'], scenario['context']
            )
            
            # Get emotion vectors
            emotions = self.emotion_engine.map_vad_to_emotion(
                emotion_state.valence, 
                emotion_state.arousal, 
                emotion_state.dominance,
                top_k=6
            )
            
            print("\n🎭 Detected Emotions:")
            for emotion in emotions:
                intensity_bar = "█" * int(emotion.intensity * 10)
                print(f"  {emotion.emotion:15} │{intensity_bar:<10}│ {emotion.intensity:.3f} ({emotion.category})")
                print(f"                  └─ {emotion.description}")
            
            # Check accuracy
            detected_names = [e.emotion for e in emotions[:4]]
            expected = scenario['expected_emotions']
            matches = len(set(detected_names) & set(expected))
            print(f"\n✓ Accuracy: {matches}/{len(expected)} expected emotions detected")
    
    async def _demo_temporal_tracking(self):
        """Demonstrate emotional trajectory tracking over time"""
        print("\n3. Temporal Tracking Demo")
        print("-" * 30)
        
        print("Simulating emotional journey over time...")
        
        # Simulate a sequence of emotional states
        emotional_journey = [
            ("Morning: Waking up refreshed", "I feel energized and ready for the day!"),
            ("Midday: Work stress", "This deadline is impossible. I'm overwhelmed."),
            ("Afternoon: Problem solved", "Finally figured it out! That was challenging but rewarding."),
            ("Evening: Reflection", "Today taught me a lot about perseverance and problem-solving."),
            ("Night: Gratitude", "I'm grateful for the growth and learning opportunities.")
        ]
        
        for time_point, text in emotional_journey:
            emotion_state = await self.emotion_engine.analyze_text_emotion(text)
            await self.emotion_engine.update_mood_state(emotion_state)
            
            # Get current emotional summary
            summary = self.emotion_engine.get_emotion_summary()
            dominant_emotion = list(summary["dominant_emotions"].keys())[0] if summary["dominant_emotions"] else "neutral"
            
            print(f"\n⏰ {time_point}")
            print(f"   Text: '{text}'")
            print(f"   Dominant emotion: {dominant_emotion}")
            print(f"   VAD: ({emotion_state.valence:.2f}, {emotion_state.arousal:.2f}, {emotion_state.dominance:.2f})")
        
        # Analyze trajectory
        trend_analysis = self.emotion_engine.emotional_trajectory.get_trend()
        print(f"\n📈 Trajectory Analysis:")
        print(f"   Emotional stability: {trend_analysis['stability']:.3f}")
        print(f"   Valence trend: {trend_analysis['valence_trend']:.3f}")
        print(f"   Volatility: {trend_analysis['volatility']:.3f}")
    
    async def _demo_context_integration(self):
        """Demonstrate context-aware emotional analysis"""
        print("\n4. Context Integration Demo")
        print("-" * 35)
        
        base_text = "I have to speak in public tomorrow."
        
        contexts = [
            {
                "name": "Late night anxiety",
                "context": {"time_of_day": 23, "topic": "anxiety", "conversation_length": 1},
                "description": "Late at night, first mention"
            },
            {
                "name": "Morning preparation",
                "context": {"time_of_day": 8, "topic": "preparation", "conversation_length": 5},
                "description": "Morning, after discussing preparation"
            },
            {
                "name": "Supportive environment",
                "context": {"time_of_day": 14, "topic": "support", "stress_level": 0.3},
                "description": "Afternoon, with emotional support"
            }
        ]
        
        print(f"Base text: '{base_text}'")
        
        for ctx in contexts:
            emotion_state = await self.emotion_engine.analyze_text_emotion(
                base_text, ctx["context"]
            )
            
            emotions = self.emotion_engine.map_vad_to_emotion(
                emotion_state.valence, emotion_state.arousal, emotion_state.dominance, top_k=3
            )
            
            print(f"\n🌍 Context: {ctx['name']}")
            print(f"   {ctx['description']}")
            print(f"   VAD: ({emotion_state.valence:.2f}, {emotion_state.arousal:.2f}, {emotion_state.dominance:.2f})")
            print(f"   Top emotions: {', '.join([e.emotion for e in emotions])}")
    
    def save_demo_results(self, filename: str = "emotion_demo_results.json"):
        """Save demo results for analysis"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "demo_type": "integrated" if self.emotion_engine else "conceptual",
            "scenarios_count": len(self.demo_scenarios),
            "capabilities": {
                "vad_model": True,
                "emotion_count": 25,
                "temporal_tracking": True,
                "context_integration": True,
                "memory_integration": True,
                "communication_adaptation": True
            }
        }
        
        if self.emotion_engine:
            results.update({
                "emotion_mappings_count": len(self.emotion_engine.emotion_mappings),
                "categories": list(set(data["category"] for data in self.emotion_engine.emotion_mappings.values())),
                "emotional_state": self.emotion_engine.save_emotional_state()
            })
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Demo results saved to {filename}")

async def main():
    """Run the emotional intelligence demo"""
    print("🚀 Starting MNEMIA Emotional Intelligence Demo...")
    
    demo = EmotionalIntelligenceDemo()
    
    try:
        await demo.run_comprehensive_demo()
        demo.save_demo_results()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 