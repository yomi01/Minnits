#!/usr/bin/env python3
"""
Test script for the updated detailed summarization feature.
"""

import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from summarization.summarizer import ConversationSummarizer
from summarization.formatter import SummaryFormatter

def test_detailed_summary():
    """Test the detailed summarization feature"""
    
    # Sample transcript data for testing
    sample_transcript = {
        "transcript": "This is a sample conversation about project planning and budget allocation.",
        "segments": [
            {"timestamp": "00:00:00", "text": "Good morning everyone, let's start our project planning meeting.", "speaker": "Project Manager"},
            {"timestamp": "00:00:05", "text": "I think we need to set a deadline for the first milestone by March 15th.", "speaker": "Project Manager"},
            {"timestamp": "00:00:10", "text": "That sounds reasonable. I'll handle the design mockups.", "speaker": "Designer"},
            {"timestamp": "00:00:15", "text": "Great! Please also prepare the budget proposal by next Wednesday.", "speaker": "Project Manager"},
            {"timestamp": "00:00:20", "text": "I agree with the timeline. We should also consider the resource allocation.", "speaker": "Team Lead"},
            {"timestamp": "00:00:25", "text": "Yes, I'll review the current budget and send you an updated proposal.", "speaker": "Designer"},
            {"timestamp": "00:00:30", "text": "Perfect. Let's schedule a follow-up meeting for next Friday to review progress.", "speaker": "Project Manager"}
        ]
    }
    
    print("🧪 Testing Detailed Summarization Feature")
    print("=" * 60)
    
    try:
        # Initialize summarizer
        print("Initializing ConversationSummarizer...")
        summarizer = ConversationSummarizer()
        
        print("✅ Summarizer initialized successfully")
        
        # Test detailed format
        print("\n📋 Testing detailed meeting minutes format...")
        detailed_summary = summarizer.summarize(
            sample_transcript, 
            focus_areas=["timeline", "budget", "responsibilities"],
            use_detailed_format=True
        )
        
        print("✅ Detailed summary generated successfully")
        print(f"Format used: {detailed_summary.get('format_used', 'unknown')}")
        
        # Display the detailed summary
        print("\n📄 Detailed Meeting Minutes:")
        print("-" * 40)
        
        if detailed_summary.get("meeting_overview"):
            print(f"\n🏢 Meeting Overview:")
            print(detailed_summary["meeting_overview"])
        
        if detailed_summary.get("discussion_highlights"):
            print(f"\n💬 Discussion Highlights:")
            print(detailed_summary["discussion_highlights"])
        
        if detailed_summary.get("decisions_made"):
            print(f"\n✅ Decisions Made:")
            for decision in detailed_summary["decisions_made"]:
                print(f"  • {decision}")
        
        if detailed_summary.get("action_points"):
            print(f"\n📋 Action Items:")
            for action in detailed_summary["action_points"]:
                print(f"  • {action}")
        
        if detailed_summary.get("next_steps"):
            print(f"\n🔄 Next Steps:")
            print(detailed_summary["next_steps"])
        
        # Test simple format for comparison
        print("\n📝 Testing simple summary format...")
        simple_summary = summarizer.summarize(
            sample_transcript,
            focus_areas=["timeline", "budget"],
            use_detailed_format=False
        )
        
        print("✅ Simple summary generated successfully")
        print(f"Format used: {simple_summary.get('format_used', 'unknown')}")
        
        print("\n📄 Simple Summary:")
        print("-" * 40)
        print(simple_summary["summary"])
        
        if simple_summary.get("action_points"):
            print("\n📋 Action Points:")
            for i, action in enumerate(simple_summary["action_points"], 1):
                print(f"  {i}. {action}")
        
        # Test formatter with detailed format
        print("\n💾 Testing SummaryFormatter with detailed format...")
        formatter = SummaryFormatter()
        
        # Save as multiple formats
        output_path = "test_output/detailed_summary"
        os.makedirs("test_output", exist_ok=True)
        
        saved_files = formatter.save_multiple_formats(detailed_summary, output_path)
        
        print("✅ Summary saved in multiple formats:")
        for format_type, file_path in saved_files.items():
            print(f"  • {format_type.upper()}: {file_path}")
        
        print("\n🎉 All tests completed successfully!")
        print("\nTo use the detailed format in your application:")
        print("  • CLI: python main.py --summarize transcript.json --detailed")
        print("  • CLI: python main.py --summarize transcript.json --no-detailed (for simple format)")
        print("  • UI: The Streamlit interface now uses detailed format by default")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        print("\nMake sure:")
        print("  • Ollama is running on http://localhost:11434")
        print("  • The model 'gemma3:12b' is available")
        print("  • All dependencies are installed (langchain, langchain-community)")
        return False

if __name__ == "__main__":
    success = test_detailed_summary()
    sys.exit(0 if success else 1) 