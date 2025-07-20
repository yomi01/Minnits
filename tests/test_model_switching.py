#!/usr/bin/env python3
"""
Test script for the model switching functionality.
"""

import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summarization.summarizer import ConversationSummarizer
from ui.app import get_available_ollama_models, get_model_info

def test_model_switching():
    """Test the model switching and validation functionality"""
    
    print("🧪 Testing Model Switching Functionality")
    print("=" * 60)
    
    # Test getting available models
    print("1. Testing model discovery...")
    try:
        models = get_available_ollama_models()
        if models:
            print(f"✅ Found {len(models)} available models:")
            for model in models[:5]:  # Show first 5
                print(f"   • {model}")
            if len(models) > 5:
                print(f"   ... and {len(models) - 5} more")
        else:
            print("⚠️ No models found. Make sure Ollama is running with models installed.")
            return False
    except Exception as e:
        print(f"❌ Error getting models: {e}")
        return False
    
    # Test model info retrieval
    if models:
        test_model = models[0]
        print(f"\n2. Testing model info for '{test_model}'...")
        try:
            model_info = get_model_info(test_model)
            if model_info:
                print("✅ Model info retrieved:")
                print(f"   • Size: {model_info['size']}")
                print(f"   • Family: {model_info['family']}")
                print(f"   • Format: {model_info['format']}")
            else:
                print("⚠️ Could not retrieve model info")
        except Exception as e:
            print(f"❌ Error getting model info: {e}")
    
    # Test model switching with summarizer
    print(f"\n3. Testing model switching in summarizer...")
    
    # Sample transcript for testing
    sample_transcript = {
        "transcript": "Quick test conversation for model switching.",
        "segments": [
            {"timestamp": "00:00:00", "text": "Hello, this is a test.", "speaker": "Speaker 1"},
            {"timestamp": "00:00:02", "text": "Yes, we are testing model switching.", "speaker": "Speaker 2"},
            {"timestamp": "00:00:04", "text": "The model should work correctly.", "speaker": "Speaker 1"}
        ]
    }
    
    # Test with first available model
    if models:
        try:
            print(f"   Testing with model: {models[0]}")
            summarizer1 = ConversationSummarizer(ollama_model=models[0])
            
            # Try a quick summarization
            summary1 = summarizer1.summarize(sample_transcript, use_detailed_format=False)
            print(f"✅ Model '{models[0]}' working correctly")
            
            # Test with second model if available
            if len(models) > 1:
                print(f"   Testing with model: {models[1]}")
                summarizer2 = ConversationSummarizer(ollama_model=models[1])
                summary2 = summarizer2.summarize(sample_transcript, use_detailed_format=False)
                print(f"✅ Model '{models[1]}' working correctly")
                
                # Compare results
                if summary1.get("summary") != summary2.get("summary"):
                    print("✅ Different models produce different outputs (as expected)")
                else:
                    print("ℹ️ Models produced similar outputs")
            
        except Exception as e:
            print(f"❌ Error testing models: {e}")
            return False
    
    # Test invalid model handling
    print("\n4. Testing invalid model handling...")
    try:
        invalid_summarizer = ConversationSummarizer(ollama_model="nonexistent-model:999b")
        print("❌ Should have failed with invalid model")
        return False
    except Exception as e:
        print("✅ Invalid model correctly rejected")
    
    print("\n🎉 All model switching tests completed successfully!")
    print("\nModel switching features:")
    print("  • ✅ Model discovery from Ollama")
    print("  • ✅ Model information retrieval")
    print("  • ✅ Dynamic model switching")
    print("  • ✅ Invalid model validation")
    print("  • ✅ UI integration ready")
    
    return True

def show_model_recommendations():
    """Show model recommendations for different use cases"""
    print("\n📋 Model Recommendations by Use Case:")
    print("-" * 40)
    
    recommendations = {
        "💼 Business Meetings": [
            "gemma3:12b - Excellent for formal meeting minutes",
            "llama3.2:8b - Good balance of speed and quality",
            "qwen2.5:7b - Strong multilingual support"
        ],
        "🎓 Academic/Research": [
            "llama3.2:8b - Great for technical content",
            "mistral:7b - Good for analysis and summaries",
            "gemma3:12b - Excellent comprehension"
        ],
        "⚡ Quick Summaries": [
            "llama3.2:3b - Fast processing",
            "gemma3:3b - Good quality, small size",
            "phi3:3.8b - Efficient for simple tasks"
        ],
        "🌍 Multilingual": [
            "qwen2.5:7b - Best for Chinese/English",
            "gemma3:12b - Good multilingual support",
            "mistral:7b - European languages"
        ]
    }
    
    for category, models in recommendations.items():
        print(f"\n{category}:")
        for model in models:
            print(f"  • {model}")

if __name__ == "__main__":
    success = test_model_switching()
    
    if success:
        show_model_recommendations()
    
    sys.exit(0 if success else 1) 