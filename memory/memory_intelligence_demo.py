#!/usr/bin/env python3
"""
MNEMIA Advanced Memory-Guided Intelligence Demo
Comprehensive demonstration of vector memory, graph relations, auto-storage, and smart retrieval
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from advanced_memory_manager import (
    AdvancedMemoryManager, MemoryRetrievalContext, MemoryType,
    ModalStateInfluence
)

class MemoryIntelligenceDemo:
    """Demonstration of advanced memory-guided intelligence capabilities"""
    
    def __init__(self):
        self.memory_manager = AdvancedMemoryManager()
        self.demo_conversation_id = f"demo_conv_{uuid.uuid4().hex[:8]}"
        self.demo_user_id = "demo_user"
        
    async def initialize(self):
        """Initialize the memory manager"""
        print("🧠 Initializing Advanced Memory-Guided Intelligence System...")
        await self.memory_manager.initialize()
        print("✅ Memory system initialized successfully!")
        print()
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of all memory capabilities"""
        print("=" * 80)
        print("🚀 MNEMIA ADVANCED MEMORY-GUIDED INTELLIGENCE DEMO")
        print("=" * 80)
        print()
        
        await self.initialize()
        
        # Demo sections
        await self.demo_automatic_storage()
        await self.demo_modal_state_aware_retrieval()
        await self.demo_emotional_memory_integration()
        await self.demo_graph_conceptual_connections()
        await self.demo_conversation_context_tracking()
        await self.demo_memory_pattern_analysis()
        await self.demo_performance_analytics()
        
        print("=" * 80)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    
    async def demo_automatic_storage(self):
        """Demonstrate automatic memory storage with full context"""
        print("📚 DEMO 1: Automatic Memory Storage with Full Context")
        print("-" * 60)
        
        # Sample conversations with different modal states and emotional contexts
        conversations = [
            {
                "user_input": "I'm feeling anxious about my upcoming presentation at work.",
                "ai_response": "I understand your anxiety about the presentation. Let's work through some strategies to help you feel more confident and prepared.",
                "modal_state": "awake",
                "emotional_context": {
                    "mood_state": {"valence": -0.3, "arousal": 0.7, "dominance": 0.3},
                    "dominant_emotions": ["anxiety", "nervousness", "concern"],
                    "emotional_intensity": 0.8
                }
            },
            {
                "user_input": "I had the most beautiful dream about flying through clouds of colors.",
                "ai_response": "What a wonderful and vivid dream! Flying dreams often represent freedom and transcendence. The colors you describe suggest a rich emotional landscape in your subconscious.",
                "modal_state": "dreaming",
                "emotional_context": {
                    "mood_state": {"valence": 0.8, "arousal": 0.6, "dominance": 0.7},
                    "dominant_emotions": ["wonder", "joy", "fascination"],
                    "emotional_intensity": 0.7
                }
            },
            {
                "user_input": "I've been thinking deeply about the nature of consciousness and what it means to truly understand something.",
                "ai_response": "That's a profound philosophical inquiry. Consciousness and understanding are among the deepest mysteries we face. What aspects of consciousness intrigue you most?",
                "modal_state": "contemplating",
                "emotional_context": {
                    "mood_state": {"valence": 0.2, "arousal": 0.4, "dominance": 0.6},
                    "dominant_emotions": ["curiosity", "contemplation", "wonder"],
                    "emotional_intensity": 0.6
                }
            },
            {
                "user_input": "Can you help me understand how machine learning algorithms work?",
                "ai_response": "I'd be happy to explain machine learning! Let's start with the basics. Machine learning is about finding patterns in data to make predictions or decisions.",
                "modal_state": "learning",
                "emotional_context": {
                    "mood_state": {"valence": 0.4, "arousal": 0.6, "dominance": 0.5},
                    "dominant_emotions": ["curiosity", "eagerness", "focus"],
                    "emotional_intensity": 0.5
                }
            },
            {
                "user_input": "I'm confused about my feelings lately. Everything seems overwhelming.",
                "ai_response": "It sounds like you're going through a challenging time with mixed emotions. Feeling overwhelmed is completely valid, and we can work together to sort through these feelings.",
                "modal_state": "confused",
                "emotional_context": {
                    "mood_state": {"valence": -0.4, "arousal": 0.8, "dominance": 0.2},
                    "dominant_emotions": ["confusion", "overwhelm", "uncertainty"],
                    "emotional_intensity": 0.9
                }
            }
        ]
        
        stored_memory_ids = []
        
        for i, conv in enumerate(conversations, 1):
            print(f"Storing conversation {i} (Modal State: {conv['modal_state']})...")
            
            # Store the conversation automatically
            memory_id = await self.memory_manager.store_memory_automatically(
                content=f"User: {conv['user_input']}\nAI: {conv['ai_response']}",
                conversation_id=self.demo_conversation_id,
                user_input=conv['user_input'],
                ai_response=conv['ai_response'],
                emotional_context=conv['emotional_context'],
                modal_state=conv['modal_state'],
                user_id=self.demo_user_id
            )
            
            stored_memory_ids.append(memory_id)
            print(f"  ✅ Memory stored with ID: {memory_id}")
            
            # Small delay to ensure different timestamps
            await asyncio.sleep(0.1)
        
        print(f"\n📊 Successfully stored {len(stored_memory_ids)} memories with automatic context integration!")
        print("   - Memory types automatically classified")
        print("   - Emotional coordinates extracted")
        print("   - Graph relationships established")
        print("   - Concepts extracted and linked")
        print()
        
        return stored_memory_ids
    
    async def demo_modal_state_aware_retrieval(self):
        """Demonstrate modal state-aware smart retrieval"""
        print("🎯 DEMO 2: Modal State-Aware Smart Retrieval")
        print("-" * 60)
        
        query = "feeling anxious and overwhelmed"
        
        # Test retrieval with different modal states
        modal_states = ["awake", "dreaming", "reflecting", "learning", "contemplating", "confused"]
        
        for modal_state in modal_states:
            print(f"Retrieving memories in '{modal_state}' modal state...")
            
            retrieval_context = MemoryRetrievalContext(
                query=query,
                modal_state=modal_state,
                emotional_context={
                    "mood_state": {"valence": -0.2, "arousal": 0.7, "dominance": 0.4}
                },
                retrieval_strategy="balanced"
            )
            
            result = await self.memory_manager.retrieve_memories_smart(
                retrieval_context=retrieval_context,
                top_k=3,
                include_graph_connections=True
            )
            
            print(f"  📋 Found {len(result.memories)} memories (confidence: {result.confidence:.3f})")
            print(f"  ⏱️  Retrieval time: {result.retrieval_time:.3f}s")
            
            if result.memories:
                top_memory = result.memories[0]
                print(f"  🥇 Top memory (score: {result.retrieval_scores[0]:.3f}):")
                print(f"     Type: {top_memory.memory_type.value}")
                print(f"     Origin: {top_memory.modal_state_origin}")
                print(f"     Content: {top_memory.content[:100]}...")
                
                # Show score breakdown
                print(f"  📊 Score breakdown:")
                print(f"     Semantic: {result.semantic_similarity[0]:.3f}")
                print(f"     Emotional: {result.emotional_relevance[0]:.3f}")
                print(f"     Temporal: {result.temporal_relevance[0]:.3f}")
                print(f"     Modal Alignment: {result.modal_state_alignment[0]:.3f}")
            
            print()
    
    async def demo_emotional_memory_integration(self):
        """Demonstrate emotional memory integration and relevance"""
        print("💝 DEMO 3: Emotional Memory Integration")
        print("-" * 60)
        
        # Test different emotional contexts
        emotional_scenarios = [
            {
                "name": "High Anxiety",
                "context": {
                    "mood_state": {"valence": -0.6, "arousal": 0.9, "dominance": 0.2},
                    "dominant_emotions": ["anxiety", "stress", "worry"]
                },
                "query": "feeling stressed"
            },
            {
                "name": "Creative Joy",
                "context": {
                    "mood_state": {"valence": 0.8, "arousal": 0.7, "dominance": 0.8},
                    "dominant_emotions": ["joy", "creativity", "inspiration"]
                },
                "query": "beautiful dreams and imagination"
            },
            {
                "name": "Philosophical Calm",
                "context": {
                    "mood_state": {"valence": 0.3, "arousal": 0.3, "dominance": 0.7},
                    "dominant_emotions": ["contemplation", "peace", "wisdom"]
                },
                "query": "understanding consciousness"
            }
        ]
        
        for scenario in emotional_scenarios:
            print(f"🎭 Scenario: {scenario['name']}")
            
            retrieval_context = MemoryRetrievalContext(
                query=scenario['query'],
                modal_state="awake",
                emotional_context=scenario['context'],
                retrieval_strategy="emotional"
            )
            
            result = await self.memory_manager.retrieve_memories_smart(
                retrieval_context=retrieval_context,
                top_k=3
            )
            
            print(f"  📊 Emotional relevance scores:")
            for i, (memory, emotional_score) in enumerate(zip(result.memories, result.emotional_relevance)):
                print(f"    Memory {i+1}: {emotional_score:.3f} (VAD: {memory.emotional_valence:.2f}, {memory.emotional_arousal:.2f}, {memory.emotional_dominance:.2f})")
            
            if result.memories:
                best_match = result.memories[0]
                print(f"  🎯 Best emotional match: {best_match.memory_type.value} memory")
                print(f"     Emotional distance minimized through VAD space")
            
            print()
    
    async def demo_graph_conceptual_connections(self):
        """Demonstrate graph-based conceptual connections"""
        print("🕸️  DEMO 4: Graph-Based Conceptual Connections")
        print("-" * 60)
        
        # Test concept insights
        concepts_to_explore = ["anxiety", "consciousness", "learning", "dreams"]
        
        for concept in concepts_to_explore:
            print(f"🔍 Exploring concept: '{concept}'")
            
            insights = await self.memory_manager.get_memory_graph_insights(
                concept=concept,
                depth=2
            )
            
            if "error" not in insights:
                print(f"  📈 Found {insights['total_related_concepts']} related concepts")
                print(f"  💭 Found {insights['total_associated_memories']} associated memories")
                
                if insights['related_concepts']:
                    print("  🔗 Top related concepts:")
                    for related in insights['related_concepts'][:3]:
                        print(f"     - {related['concept']} (strength: {related['avg_strength']:.3f})")
                
                if insights['associated_memories']:
                    print("  📚 Recent associated memories:")
                    for memory in insights['associated_memories'][:2]:
                        print(f"     - {memory['memory_type']}: {memory['content'][:80]}...")
            else:
                print(f"  ⚠️  {insights['error']}")
            
            print()
    
    async def demo_conversation_context_tracking(self):
        """Demonstrate conversation context tracking"""
        print("💬 DEMO 5: Conversation Context Tracking")
        print("-" * 60)
        
        # Get conversation context
        context = await self.memory_manager.get_conversation_context(self.demo_conversation_id)
        
        if context:
            print(f"📊 Conversation Analysis for {self.demo_conversation_id}:")
            print(f"  Memory Count: {context['memory_count']}")
            print(f"  Duration: {context['created_at']} to {context['last_updated']}")
            
            print(f"\n🎭 Emotional Statistics:")
            emotional_stats = context['emotional_stats']
            print(f"  Average Valence: {emotional_stats['avg_valence']:.3f}")
            print(f"  Average Arousal: {emotional_stats['avg_arousal']:.3f}")
            print(f"  Average Dominance: {emotional_stats['avg_dominance']:.3f}")
            print(f"  Emotional Volatility: {emotional_stats['emotional_volatility']:.3f}")
            
            print(f"\n🧠 Modal State Distribution:")
            for state, count in context['modal_state_distribution'].items():
                print(f"  {state}: {count} occurrences")
            
            print(f"\n📝 Recent Memory IDs: {context['recent_memories'][:3]}...")
        else:
            print("⚠️  Conversation context not found")
        
        print()
    
    async def demo_memory_pattern_analysis(self):
        """Demonstrate memory pattern analysis"""
        print("📈 DEMO 6: Memory Pattern Analysis")
        print("-" * 60)
        
        analysis = await self.memory_manager.analyze_memory_patterns(self.demo_user_id)
        
        if "error" not in analysis:
            print(f"📊 Memory Pattern Analysis for {self.demo_user_id}:")
            print(f"  Total Memories: {analysis['total_memories']}")
            
            print(f"\n🏷️  Memory Type Distribution:")
            for mem_type, count in analysis['memory_type_distribution'].items():
                percentage = (count / analysis['total_memories']) * 100
                print(f"  {mem_type}: {count} ({percentage:.1f}%)")
            
            print(f"\n🧠 Modal State Distribution:")
            for state, count in analysis['modal_state_distribution'].items():
                percentage = (count / analysis['total_memories']) * 100
                print(f"  {state}: {count} ({percentage:.1f}%)")
            
            print(f"\n💝 Emotional Analysis:")
            emotional = analysis['emotional_analysis']
            print(f"  Average Valence: {emotional['avg_valence']:.3f} (±{emotional['valence_std']:.3f})")
            print(f"  Average Arousal: {emotional['avg_arousal']:.3f} (±{emotional['arousal_std']:.3f})")
            print(f"  Average Dominance: {emotional['avg_dominance']:.3f} (±{emotional['dominance_std']:.3f})")
            
            print(f"\n🕐 Temporal Patterns:")
            print(f"  Most Active Hour: {analysis['most_active_hour']}:00")
            print(f"  Dominant Memory Type: {analysis['dominant_memory_type']}")
            print(f"  Dominant Modal State: {analysis['dominant_modal_state']}")
        else:
            print(f"⚠️  {analysis['error']}")
        
        print()
    
    async def demo_performance_analytics(self):
        """Demonstrate performance analytics and system health"""
        print("⚡ DEMO 7: Performance Analytics & System Health")
        print("-" * 60)
        
        stats = await self.memory_manager.get_performance_stats()
        
        if "error" not in stats:
            print("🗄️  Database Statistics:")
            
            # Vector database stats
            vector_db = stats['vector_database']
            print(f"  Vector DB (Qdrant):")
            print(f"    Collection: {vector_db['collection_name']}")
            print(f"    Total Points: {vector_db['total_points']:,}")
            print(f"    Vector Size: {vector_db['vector_size']}")
            print(f"    Distance Metric: {vector_db['distance_metric']}")
            
            # Graph database stats
            graph_db = stats['graph_database']
            print(f"  Graph DB (Neo4j):")
            print(f"    Memory Nodes: {graph_db['memory_nodes']:,}")
            print(f"    Concept Nodes: {graph_db['concept_nodes']:,}")
            print(f"    Relationships: {graph_db['relationships']:,}")
            
            # Cache database stats
            if stats['cache_database']:
                cache_db = stats['cache_database']
                print(f"  Cache DB (Redis):")
                print(f"    Used Memory: {cache_db.get('used_memory', 'N/A')}")
                print(f"    Connected Clients: {cache_db.get('connected_clients', 0)}")
            
            print(f"\n📊 Performance Metrics:")
            perf = stats['performance_metrics']
            print(f"  Total Memories: {perf['total_memories']:,}")
            print(f"  Total Retrievals: {perf['total_retrievals']:,}")
            print(f"  Average Retrieval Time: {perf['avg_retrieval_time']:.3f}s")
            print(f"  Cache Hits: {perf['cache_hits']:,}")
            print(f"  Graph Queries: {perf['graph_queries']:,}")
            
            print(f"\n🧠 System State:")
            print(f"  Active Conversation Contexts: {stats['conversation_contexts']}")
            print(f"  Memory Cache Size: {stats['memory_cache_size']}")
        else:
            print(f"⚠️  {stats['error']}")
        
        print()
    
    async def interactive_query_demo(self):
        """Interactive demonstration allowing user queries"""
        print("🔍 INTERACTIVE QUERY DEMO")
        print("-" * 60)
        print("Enter queries to test the memory system (type 'quit' to exit):")
        print()
        
        while True:
            try:
                query = input("Query: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                # Ask for modal state
                print("Modal states: awake, dreaming, reflecting, learning, contemplating, confused")
                modal_state = input("Modal state (default: awake): ").strip() or "awake"
                
                retrieval_context = MemoryRetrievalContext(
                    query=query,
                    modal_state=modal_state,
                    emotional_context={
                        "mood_state": {"valence": 0.0, "arousal": 0.5, "dominance": 0.5}
                    }
                )
                
                start_time = time.time()
                result = await self.memory_manager.retrieve_memories_smart(
                    retrieval_context=retrieval_context,
                    top_k=5
                )
                query_time = time.time() - start_time
                
                print(f"\n📊 Results ({len(result.memories)} memories, {query_time:.3f}s):")
                
                for i, (memory, score) in enumerate(zip(result.memories, result.retrieval_scores)):
                    print(f"\n{i+1}. Score: {score:.3f} | Type: {memory.memory_type.value}")
                    print(f"   Origin: {memory.modal_state_origin} | Time: {memory.timestamp.strftime('%Y-%m-%d %H:%M')}")
                    print(f"   Content: {memory.content[:150]}...")
                
                if result.graph_connections:
                    print(f"\n🔗 Graph Connections: {len(result.graph_connections)}")
                
                print(f"\nConfidence: {result.confidence:.3f}")
                print("-" * 40)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("Interactive demo ended.")

async def main():
    """Main demo function"""
    demo = MemoryIntelligenceDemo()
    
    try:
        await demo.run_comprehensive_demo()
        
        # Optional interactive demo
        print("\n" + "=" * 80)
        response = input("Would you like to run the interactive query demo? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            await demo.interactive_query_demo()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧠 Advanced Memory-Guided Intelligence Demo Complete!")

if __name__ == "__main__":
    asyncio.run(main()) 