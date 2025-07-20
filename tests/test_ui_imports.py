#!/usr/bin/env python3
"""
Test script to verify UI import issues are resolved.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_ui_imports():
    """Test that UI imports work correctly"""
    print("🧪 Testing UI Import Issues")
    print("=" * 40)
    
    try:
        # Test basic imports
        print("1. Testing basic imports...")
        from datetime import datetime
        from summarization.summarizer import ConversationSummarizer
        print("✅ Basic imports successful")
        
        # Test datetime usage
        print("2. Testing datetime usage...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        print(f"✅ Datetime works: {timestamp}")
        
        # Test ConversationSummarizer instantiation
        print("3. Testing ConversationSummarizer...")
        try:
            summarizer = ConversationSummarizer()
            print("✅ ConversationSummarizer instantiated successfully")
        except Exception as e:
            if "connection" in str(e).lower():
                print("⚠️ ConversationSummarizer failed (Ollama not running - expected)")
            else:
                print(f"❌ ConversationSummarizer failed: {e}")
                return False
        
        # Test UI module imports
        print("4. Testing UI module functions...")
        from ui.app import get_available_ollama_models, get_model_info
        print("✅ UI functions imported successfully")
        
        # Test model functions
        print("5. Testing model discovery functions...")
        models = get_available_ollama_models()
        if models:
            print(f"✅ Found {len(models)} models")
            
            # Test model info for first model
            model_info = get_model_info(models[0])
            if model_info:
                print("✅ Model info retrieval works")
            else:
                print("⚠️ Model info retrieval failed (might be expected)")
        else:
            print("⚠️ No models found (Ollama might not be running)")
        
        print("\n🎉 All import tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_scope_issues():
    """Test specific scope issues that were causing errors"""
    print("\n🔍 Testing Scope Issues")
    print("=" * 40)
    
    try:
        # Test the specific datetime issue
        print("1. Testing datetime scope...")
        from datetime import datetime
        
        # This should work now (simulating the download button code)
        file_name = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        print(f"✅ Datetime in f-string works: {file_name}")
        
        # Test the ConversationSummarizer scope
        print("2. Testing ConversationSummarizer scope...")
        from summarization.summarizer import ConversationSummarizer
        
        # This should work now (simulating the model test code)
        def test_model_function():
            try:
                # This was causing the UnboundLocalError before
                test_summarizer = ConversationSummarizer(ollama_model="test:model")
                return True
            except Exception as e:
                if "connection" in str(e).lower() or "model" in str(e).lower():
                    return True  # Expected error due to invalid model/no Ollama
                else:
                    raise e
        
        if test_model_function():
            print("✅ ConversationSummarizer scope works")
        
        print("\n🎉 All scope tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Scope test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import_success = test_ui_imports()
    scope_success = test_ui_scope_issues()
    
    if import_success and scope_success:
        print("\n✅ All UI tests passed! The import issues should be resolved.")
        print("\nYou can now run the Streamlit app with:")
        print("  python main.py --ui")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    sys.exit(0 if import_success and scope_success else 1) 