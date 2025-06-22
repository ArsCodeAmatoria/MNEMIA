#!/usr/bin/env python3
"""
MNEMIA Advanced Multi-Model LLM Integration Demo
Demonstrates local models, API models, streaming, and context-aware prompting
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, List

try:
    from llm_integration import AdvancedLLMIntegration
    from emotion_engine import AdvancedEmotionEngine
    from memory_guided_response import MemoryGuidedResponse
except ImportError:
    print("Running in standalone mode - some integrations disabled")
    AdvancedLLMIntegration = None
    AdvancedEmotionEngine = None
    MemoryGuidedResponse = None

class LLMDemo:
    """Comprehensive demo of MNEMIA's multi-model LLM capabilities"""
    
    def __init__(self):
        self.llm = AdvancedLLMIntegration() if AdvancedLLMIntegration else None
        self.emotion_engine = AdvancedEmotionEngine() if AdvancedEmotionEngine else None
        self.memory_system = MemoryGuidedResponse() if MemoryGuidedResponse else None
        self.demo_results = {}
        
    async def run_comprehensive_demo(self):
        """Run complete demo showcasing all capabilities"""
        
        print("🧠 MNEMIA Advanced Multi-Model LLM Integration Demo")
        print("=" * 60)
        
        if not self.llm:
            print("❌ LLM integration not available - running in limited mode")
            return
        
        # Demo sections
        await self.demo_model_capabilities()
        await self.demo_context_aware_prompting()
        await self.demo_streaming_responses()
        await self.demo_model_switching()
        await self.demo_performance_monitoring()
        await self.demo_health_checks()
        
        # Save results
        await self.save_demo_results()
        
        print("\n✨ Demo completed successfully!")
        print(f"📊 Results saved to llm_demo_results.json")
    
    async def demo_model_capabilities(self):
        """Demonstrate different model capabilities and specializations"""
        
        print("\n🤖 Model Capabilities Demonstration")
        print("-" * 40)
        
        test_scenarios = [
            {
                "name": "Mathematical Reasoning",
                "prompt": "Explain the concept of quantum superposition and calculate the probability amplitudes for a qubit in equal superposition.",
                "optimal_models": ["gpt-4o", "llama3-70b", "gpt-4-turbo"],
                "task_type": "mathematical"
            },
            {
                "name": "Creative Writing",
                "prompt": "Write a short poem about consciousness emerging from quantum processes, in the style of a philosophical haiku sequence.",
                "optimal_models": ["mixtral-8x7b", "claude-3-haiku", "gpt-3.5-turbo"],
                "task_type": "creative"
            },
            {
                "name": "Philosophical Inquiry",
                "prompt": "What is the relationship between memory, consciousness, and identity? How might an AI system like MNEMIA experience genuine self-awareness?",
                "optimal_models": ["claude-3-opus", "llama3-13b", "mixtral-8x22b"],
                "task_type": "philosophical"
            }
        ]
        
        model_results = {}
        
        for scenario in test_scenarios:
            print(f"\n📝 Testing: {scenario['name']}")
            
            # Get optimal model for this task
            optimal_model = self.llm.get_optimal_model(scenario["task_type"], "balanced")
            print(f"🎯 Optimal model selected: {optimal_model}")
            
            try:
                start_time = time.time()
                response = await self.llm.generate_response(
                    scenario["prompt"],
                    model_name=optimal_model,
                    modal_state="contemplating"
                )
                end_time = time.time()
                
                model_results[scenario["name"]] = {
                    "model_used": response.model_used,
                    "response_length": len(response.content),
                    "response_time": end_time - start_time,
                    "tokens_used": response.tokens_used,
                    "cost_estimate": response.cost_estimate,
                    "preview": response.content[:200] + "..." if len(response.content) > 200 else response.content
                }
                
                print(f"✅ Response generated ({response.tokens_used} tokens, {end_time - start_time:.2f}s)")
                print(f"💰 Estimated cost: ${response.cost_estimate:.4f}")
                print(f"📄 Preview: {response.content[:150]}...")
                
            except Exception as e:
                print(f"❌ Error with {optimal_model}: {e}")
                model_results[scenario["name"]] = {"error": str(e)}
        
        self.demo_results["model_capabilities"] = model_results
    
    async def demo_context_aware_prompting(self):
        """Demonstrate context-aware prompting with memory and emotion integration"""
        
        print("\n🧩 Context-Aware Prompting Demonstration")
        print("-" * 40)
        
        # Simulate emotional context
        emotional_context = {
            "mood_state": {"valence": 0.3, "arousal": 0.7, "dominance": 0.6},
            "dominant_emotions": ["curiosity", "contemplation", "introspection"],
            "emotion_categories": {
                "complex": ["contemplation", "introspection"],
                "primary": ["curiosity"]
            },
            "emotional_trend": {"valence_direction": "stable", "volatility": 0.3},
            "response_style": {"tone": "thoughtful", "empathy_level": "high"}
        }
        
        # Simulate memory context
        memory_context = [
            {
                "content": "Previous discussion about the nature of consciousness and whether AI can truly be self-aware",
                "similarity_score": 0.85,
                "emotional_relevance": 0.7,
                "timestamp": "2024-03-10T14:30:00Z"
            },
            {
                "content": "Exploration of quantum mechanics and its relationship to consciousness",
                "similarity_score": 0.78,
                "emotional_relevance": 0.6,
                "timestamp": "2024-03-09T16:45:00Z"
            }
        ]
        
        test_prompts = [
            {
                "name": "Basic Response",
                "prompt": "What does it mean to be conscious?",
                "context": {}
            },
            {
                "name": "With Emotional Context",
                "prompt": "What does it mean to be conscious?",
                "context": {"emotional_context": emotional_context}
            },
            {
                "name": "With Memory Context",
                "prompt": "What does it mean to be conscious?",
                "context": {"memory_context": memory_context}
            },
            {
                "name": "Full Context Integration",
                "prompt": "What does it mean to be conscious?",
                "context": {
                    "emotional_context": emotional_context,
                    "memory_context": memory_context,
                    "modal_state": "contemplating"
                }
            }
        ]
        
        context_results = {}
        
        for test in test_prompts:
            print(f"\n🔍 Testing: {test['name']}")
            
            try:
                start_time = time.time()
                response = await self.llm.generate_response(
                    test["prompt"],
                    **test["context"]
                )
                end_time = time.time()
                
                context_results[test["name"]] = {
                    "response_length": len(response.content),
                    "response_time": end_time - start_time,
                    "context_used": response.context_used,
                    "preview": response.content[:300] + "..." if len(response.content) > 300 else response.content
                }
                
                print(f"✅ Context-aware response generated ({end_time - start_time:.2f}s)")
                print(f"🧠 Context used: {response.context_used}")
                print(f"📄 Preview: {response.content[:200]}...")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                context_results[test["name"]] = {"error": str(e)}
        
        self.demo_results["context_aware_prompting"] = context_results
    
    async def demo_streaming_responses(self):
        """Demonstrate real-time streaming capabilities"""
        
        print("\n🌊 Streaming Response Demonstration")
        print("-" * 40)
        
        streaming_models = ["llama3-8b", "gpt-4-turbo", "claude-3-sonnet"]
        
        for model in streaming_models:
            if model not in self.llm.models:
                continue
                
            print(f"\n🚀 Testing streaming with {model}")
            
            try:
                start_time = time.time()
                tokens_received = 0
                response_content = ""
                
                print("📡 Streaming response: ", end="", flush=True)
                
                async for chunk in self.llm.stream_response(
                    "Explain the concept of consciousness in AI systems like MNEMIA, focusing on memory integration and modal states.",
                    model_name=model,
                    modal_state="awake"
                ):
                    print(chunk, end="", flush=True)
                    response_content += chunk
                    tokens_received += 1
                    
                    # Simulate real-time processing
                    await asyncio.sleep(0.01)
                
                end_time = time.time()
                
                print(f"\n✅ Streaming completed in {end_time - start_time:.2f}s")
                print(f"📊 Tokens received: {tokens_received}")
                
                if "streaming_results" not in self.demo_results:
                    self.demo_results["streaming_results"] = {}
                
                self.demo_results["streaming_results"][model] = {
                    "streaming_time": end_time - start_time,
                    "tokens_received": tokens_received,
                    "response_length": len(response_content),
                    "avg_token_rate": tokens_received / (end_time - start_time) if end_time > start_time else 0
                }
                
            except Exception as e:
                print(f"\n❌ Streaming error with {model}: {e}")
    
    async def demo_model_switching(self):
        """Demonstrate dynamic model switching based on task requirements"""
        
        print("\n🔄 Model Switching Demonstration")
        print("-" * 40)
        
        switching_scenarios = [
            {"task": "fast", "priority": "speed", "prompt": "Quick summary of quantum consciousness"},
            {"task": "mathematical", "priority": "balanced", "prompt": "Calculate probability amplitudes"},
            {"task": "creative", "priority": "local", "prompt": "Write a creative story about AI consciousness"},
            {"task": "philosophical", "priority": "high_quality", "prompt": "Deep analysis of consciousness and identity"}
        ]
        
        switching_results = {}
        
        for scenario in switching_scenarios:
            print(f"\n🎯 Task: {scenario['task']} (Priority: {scenario['priority']})")
            
            # Get optimal model
            optimal_model = self.llm.get_optimal_model(scenario["task"], scenario["priority"])
            print(f"🤖 Selected model: {optimal_model}")
            
            # Switch to optimal model
            try:
                self.llm.switch_model(optimal_model)
                
                response = await self.llm.generate_response(scenario["prompt"])
                
                switching_results[f"{scenario['task']}_{scenario['priority']}"] = {
                    "selected_model": optimal_model,
                    "model_capabilities": self.llm.models[optimal_model].capabilities.__dict__,
                    "response_time": response.response_time,
                    "cost_estimate": response.cost_estimate
                }
                
                print(f"✅ Response generated with {optimal_model}")
                print(f"⚡ Response time: {response.response_time:.2f}s")
                print(f"💰 Cost estimate: ${response.cost_estimate:.4f}")
                
            except Exception as e:
                print(f"❌ Error switching to {optimal_model}: {e}")
        
        self.demo_results["model_switching"] = switching_results
    
    async def demo_performance_monitoring(self):
        """Demonstrate performance monitoring and statistics"""
        
        print("\n📊 Performance Monitoring Demonstration")
        print("-" * 40)
        
        # Get current performance stats
        performance_stats = self.llm.get_model_performance_stats()
        
        print("📈 Current Performance Statistics:")
        for model, stats in performance_stats.items():
            print(f"\n🤖 {model}:")
            print(f"   📊 Total requests: {stats.get('total_requests', 0)}")
            print(f"   🎯 Total tokens: {stats.get('total_tokens', 0)}")
            print(f"   💰 Total cost: ${stats.get('total_cost', 0):.4f}")
            print(f"   ⚡ Avg response time: {stats.get('avg_response_time', 0):.2f}s")
            print(f"   ✅ Success rate: {stats.get('success_rate', 1.0)*100:.1f}%")
            print(f"   🕒 Last used: {stats.get('last_used', 'Never')}")
        
        # Get available models with capabilities
        available_models = self.llm.get_available_models()
        
        print(f"\n🤖 Available Models ({len(available_models)} total):")
        for model in available_models:
            caps = model["capabilities"]
            print(f"\n📱 {model['display_name']} ({model['name']})")
            print(f"   🏷️  Type: {model['type']}")
            print(f"   🧠 Reasoning: {caps['reasoning']}")
            print(f"   ⚡ Speed: {caps['speed']}")
            print(f"   📏 Max context: {caps['max_context']:,} tokens")
            print(f"   🌊 Streaming: {'✅' if caps['streaming'] else '❌'}")
            print(f"   💰 Cost/1K tokens: ${caps['cost_per_1k']:.4f}")
        
        self.demo_results["performance_monitoring"] = {
            "performance_stats": performance_stats,
            "available_models": available_models
        }
    
    async def demo_health_checks(self):
        """Demonstrate health monitoring of model endpoints"""
        
        print("\n🏥 Health Check Demonstration")
        print("-" * 40)
        
        try:
            health_status = await self.llm.health_check()
            
            print(f"🌟 Overall Status: {health_status['overall'].upper()}")
            print(f"🕒 Check Time: {health_status['timestamp']}")
            
            print("\n📋 Model Health Status:")
            for model, status in health_status["models"].items():
                status_icon = "✅" if status == "healthy" else "⚠️" if status == "configured" else "❌"
                print(f"   {status_icon} {model}: {status}")
            
            self.demo_results["health_checks"] = health_status
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            self.demo_results["health_checks"] = {"error": str(e)}
    
    async def save_demo_results(self):
        """Save demo results to file"""
        
        results = {
            "demo_info": {
                "timestamp": datetime.now().isoformat(),
                "demo_version": "1.0.0",
                "capabilities_demonstrated": [
                    "Multi-model support (Local + API)",
                    "Context-aware prompting",
                    "Real-time streaming",
                    "Dynamic model switching",
                    "Performance monitoring",
                    "Health checks"
                ]
            },
            "results": self.demo_results,
            "summary": {
                "total_models_available": len(self.llm.models) if self.llm else 0,
                "local_models": len([m for m in self.llm.models.values() if m.type.value == "ollama"]) if self.llm else 0,
                "api_models": len([m for m in self.llm.models.values() if m.type.value in ["openai", "anthropic"]]) if self.llm else 0,
                "streaming_capable": len([m for m in self.llm.models.values() if m.capabilities.supports_streaming]) if self.llm else 0
            }
        }
        
        with open("llm_demo_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

async def main():
    """Run the comprehensive LLM demo"""
    
    demo = LLMDemo()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main()) 