import streamlit as st
import os
import sys
import asyncio
import nest_asyncio
import time
import json
import base64
import signal
import atexit
from datetime import datetime
import random

# Fix asyncio event loop for Streamlit
nest_asyncio.apply()

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from audio_processing.recorder import AudioRecorder
from transcription.transcriber import Transcriber
from summarization.summarizer import ConversationSummarizer
from summarization.formatter import SummaryFormatter
from utils.api_keys import load_api_keys, save_api_keys
from document_processing.reader import DocumentReader

# Page configuration
st.set_page_config(
    page_title="Conversation Recorder & Summarizer",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state variables
if "recorder" not in st.session_state:
    st.session_state.recorder = AudioRecorder()
if "recording" not in st.session_state:
    st.session_state.recording = False
if "audio_file" not in st.session_state:
    st.session_state.audio_file = None
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "recording_start_time" not in st.session_state:
    st.session_state.recording_start_time = None
if "speaker_colors" not in st.session_state:
    st.session_state.speaker_colors = {}
if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = "gemma3:12b"  # Set default model name
if "ollama_host" not in st.session_state:
    st.session_state.ollama_host = "http://localhost:11434"
if "active_summarizers" not in st.session_state:
    st.session_state.active_summarizers = []
    
# Load configuration if available
config = load_api_keys()
if config["huggingface_token"] and "hf_token" not in st.session_state:
    st.session_state.hf_token = config["huggingface_token"]
if config["ollama_host"] and "ollama_host" not in st.session_state:
    st.session_state.ollama_host = config["ollama_host"]
if config["ollama_model"] and "ollama_model" not in st.session_state:
    st.session_state.ollama_model = config["ollama_model"]

# Cleanup functions for Streamlit
def cleanup_summarizers():
    """Clean up all active summarizers"""
    if "active_summarizers" in st.session_state:
        for summarizer in st.session_state.active_summarizers:
            try:
                summarizer.cleanup()
            except Exception as e:
                st.error(f"Error cleaning up summarizer: {e}")
        st.session_state.active_summarizers = []

def register_summarizer(summarizer):
    """Register a summarizer for cleanup"""
    if "active_summarizers" not in st.session_state:
        st.session_state.active_summarizers = []
    st.session_state.active_summarizers.append(summarizer)

# Global cleanup function for signal handling
def cleanup_on_exit():
    """Clean up resources on application exit"""
    try:
        cleanup_summarizers()
        from summarization.summarizer import ConversationSummarizer
        ConversationSummarizer.cleanup_all_instances()
    except Exception:
        pass  # Ignore errors during cleanup

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals to clean up resources"""
    cleanup_on_exit()
    sys.exit(0)

def register_ui_signal_handlers():
    """Register signal handlers for the UI application"""
    try:
        # Register signal handlers and exit cleanup
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
        atexit.register(cleanup_on_exit)
    except Exception as e:
        # In UI mode, we don't want to print errors directly, just log them
        try:
            import logging
            logging.warning(f"Could not register signal handlers: {e}")
        except:
            pass  # Ignore if logging isn't available

# Register cleanup and signal handlers
register_ui_signal_handlers()

# Helper functions
def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Generate a download link for a file"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

def time_formatter(seconds):
    """Format seconds as HH:MM:SS"""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def get_speaker_color(speaker):
    """Get a consistent color for a speaker"""
    if speaker not in st.session_state.speaker_colors:
        # Predefined colors for better readability with white text
        colors = [
            "#3366CC", "#DC3912", "#FF9900", "#109618", "#990099", 
            "#0099C6", "#DD4477", "#66AA00", "#B82E2E", "#316395"
        ]
        if len(st.session_state.speaker_colors) < len(colors):
            # Assign next color in sequence
            st.session_state.speaker_colors[speaker] = colors[len(st.session_state.speaker_colors)]
        else:
            # Generate a random color if we've used all predefined ones
            color = "#{:02x}{:02x}{:02x}".format(
                random.randint(50, 200), 
                random.randint(50, 200), 
                random.randint(50, 200)
            )
            st.session_state.speaker_colors[speaker] = color
            
    return st.session_state.speaker_colors[speaker]

# Functions for handling recording state
def start_recording(audio_source="microphone"):
    """Start audio recording
    
    Args:
        audio_source: The audio source to use ("microphone" or "system_audio")
    """
    # Set the audio source
    success = st.session_state.recorder.set_audio_source(audio_source)
    if not success and audio_source == "system_audio":
        st.error("System audio recording not available. No loopback devices found.")
        return False
        
    st.session_state.recording = True
    st.session_state.recording_start_time = time.time()
    return st.session_state.recorder.start_recording()

def stop_recording():
    """Stop audio recording"""
    if st.session_state.recording:
        st.session_state.recording = False
        audio_file = st.session_state.recorder.stop_recording()
        st.session_state.audio_file = audio_file
        return audio_file
    return None

def process_audio(focus_areas=None):
    """Process recorded audio file (transcribe and summarize)
    
    Args:
        focus_areas: Optional list of areas to focus on in the summary
    """
    if not st.session_state.audio_file or not os.path.exists(st.session_state.audio_file):
        st.error("No recording available to process.")
        return
    
    with st.spinner("Transcribing audio..."):
        # Get HuggingFace token if available
        hf_token = st.session_state.get("hf_token", None)
        
        # Initialize transcriber with Whisper model
        transcriber = Transcriber(
            model_size="base",  # Use base model by default
            auth_token=hf_token
        )
        
        # Transcribe audio
        transcript = transcriber.transcribe(
            st.session_state.audio_file,
            timestamp_interval=5  # Every 5 seconds
        )
        
        st.session_state.transcript = transcript
    
    with st.spinner("Generating summary..."):
        try:            # Initialize summarizer with Ollama configuration
            summarizer = ConversationSummarizer(
                ollama_host=st.session_state.get("ollama_host", "http://localhost:11434"),
                ollama_model=st.session_state.get("ollama_model", "gemma3:12b")  # Use gemma3:12b by default
            )
            # Register for cleanup
            register_summarizer(summarizer)
              # Generate summary with focus areas using detailed format by default
            summary = summarizer.summarize(
                transcript, 
                focus_areas=focus_areas, 
                use_detailed_format=True
            )
            
            if summary.get("error"):
                st.error(f"Error generating summary: {summary['error']}")
                return
            
            st.session_state.summary = summary
            
            # Show focus areas if provided
            if focus_areas:
                st.info(f"✨ Summary focused on: {', '.join(focus_areas)}")
            
            # Auto-save summary to output directory
            saved_files = save_summary_to_output(summary, st.session_state.audio_file, 'json')
            if saved_files:
                st.success(f"Summary saved to: {list(saved_files.values())[0]}")
            
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            st.error("Make sure Ollama is running and the model is available.")
            return
        
    # Reset speaker colors when processing a new recording
    st.session_state.speaker_colors = {}

def get_available_ollama_models(ollama_host="http://localhost:11434"):
    """Get list of available models from Ollama"""
    try:
        import requests
        url = f"{ollama_host.rstrip('/')}/api/tags"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        models_data = response.json()
        models = []
        for model in models_data.get("models", []):
            model_name = model.get("name", "")
            if model_name:
                models.append(model_name)
        
        return sorted(models)
    except Exception as e:
        return []

def get_model_info(model_name, ollama_host="http://localhost:11434"):
    """Get information about a specific model"""
    try:
        import requests
        url = f"{ollama_host.rstrip('/')}/api/show"
        payload = {"name": model_name}
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        
        model_info = response.json()
        return {
            "size": model_info.get("details", {}).get("parameter_size", "Unknown"),
            "family": model_info.get("details", {}).get("family", "Unknown"),
            "format": model_info.get("details", {}).get("format", "Unknown"),
            "modified": model_info.get("modified_at", "Unknown")
        }
    except Exception as e:
        return None

def save_config_settings():
    """Save configuration settings from Streamlit UI"""
    hf_token = st.session_state.get("hf_token", None)
    ollama_host = st.session_state.get("ollama_host", "http://localhost:11434")
    ollama_model = st.session_state.get("ollama_model", "gemma3:12b")
    
    # Save configuration
    save_result = save_api_keys(
        huggingface_token=hf_token,
        ollama_host=ollama_host,
        ollama_model=ollama_model
    )
            
    return save_result

def save_summary_to_output(summary, source_file=None, output_format='json'):
    """Save summary to output directory"""
    try:
        from summarization.formatter import SummaryFormatter
        
        # Create output directory if it doesn't exist
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename based on source or timestamp
        if source_file:
            filename_base = os.path.splitext(os.path.basename(source_file))[0]
            output_base = os.path.join(output_dir, f"{filename_base}_summary")
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_base = os.path.join(output_dir, f"summary_{timestamp}")
        
        # Save using formatter
        formatter = SummaryFormatter()
        
        if output_format == 'all':
            saved_files = formatter.save_multiple_formats(summary, output_base)
            return saved_files
        else:
            saved_file = formatter.save_summary(summary, output_base, output_format)
            return {output_format: saved_file}
    
    except Exception as e:
        st.error(f"Error saving summary: {e}")
        return {}

# Main UI
def main():
    st.title("🎙️ Conversation Recorder & Summarizer")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        
        # Configuration section
        st.subheader("Configuration")
        
        # Get current values from session state or config
        current_hf_token = st.session_state.get("hf_token", config["huggingface_token"] or "")
        current_ollama_host = st.session_state.get("ollama_host", config["ollama_host"])
        current_ollama_model = st.session_state.get("ollama_model", config["ollama_model"])
        
        # Ollama settings
        st.markdown("##### Ollama Configuration")
        ollama_host = st.text_input("Ollama Host URL", 
                                   value=current_ollama_host,
                                   help="URL where Ollama is running")
        if ollama_host:
            st.session_state.ollama_host = ollama_host
            
        # Get available models
        col1, col2 = st.columns([4, 1])
        with col1:
            available_models = get_available_ollama_models(st.session_state.get("ollama_host", "http://localhost:11434"))
        with col2:
            refresh_models_btn = st.button("🔄", key="refresh_models", help="Refresh available models list")
        
        if refresh_models_btn:
            # Clear cached models and refresh
            available_models = get_available_ollama_models(st.session_state.get("ollama_host", "http://localhost:11434"))
            if available_models:
                st.success(f"Found {len(available_models)} models")
            else:
                st.warning("No models found or Ollama not connected")
        
        # Model selection interface
        if available_models:
            # If current model is in the list, show dropdown
            if current_ollama_model in available_models:
                selected_model = st.selectbox(
                    "Select Ollama Model",
                    options=available_models,
                    index=available_models.index(current_ollama_model),
                    help="Choose from available models on your Ollama instance"
                )
                if selected_model != current_ollama_model:
                    st.session_state.ollama_model = selected_model
                    st.session_state.model_tested = True  # Models from Ollama are already available
                    st.session_state.test_model_name = selected_model
                ollama_model = selected_model
            else:
                # Show dropdown with option to use custom model
                model_options = available_models + ["Custom..."]
                selected_option = st.selectbox(
                    "Select Ollama Model",
                    options=model_options,
                    index=len(model_options) - 1,  # Default to "Custom..."
                    help="Choose from available models or enter a custom model name"
                )
                
                if selected_option == "Custom...":
                    ollama_model = st.text_input("Custom Model Name", 
                                                value=current_ollama_model,
                                                help="Enter model name (e.g., gemma3:12b, llama3.2, qwen2.5)")
                else:
                    ollama_model = selected_option
                    if ollama_model != current_ollama_model:
                        st.session_state.ollama_model = ollama_model
                        st.session_state.model_tested = True
                        st.session_state.test_model_name = ollama_model
        else:
            # No models found, show text input only
            st.info("💡 Cannot connect to Ollama or no models found. You can still enter a model name manually.")
            ollama_model = st.text_input("Ollama Model", 
                                        value=current_ollama_model,
                                        help="Model to use for summarization (e.g., gemma3:12b, llama3.2, qwen2.5)")
        
        # Test button for manual model entry
        if not available_models or ollama_model not in available_models:
            col1, col2 = st.columns([3, 1])
            with col2:
                test_model_btn = st.button("Test", key="test_model", help="Test if the model is available")
        else:
            test_model_btn = False  # Don't show test button for models from dropdown
        
        if ollama_model and ollama_model != current_ollama_model and ollama_model not in available_models:
            st.session_state.ollama_model = ollama_model
            st.session_state.model_tested = False  # Reset test status when model changes
        
        # Model testing functionality
        if test_model_btn and ollama_model:
            with st.spinner(f"Testing model '{ollama_model}'..."):
                try:
                    # Test model availability
                    test_summarizer = ConversationSummarizer(
                        ollama_host=st.session_state.get("ollama_host", "http://localhost:11434"),
                        ollama_model=ollama_model
                    )
                    # Clean up test instance immediately
                    test_summarizer.cleanup()
                    st.success(f"✅ Model '{ollama_model}' is available and ready!")
                    st.session_state.model_tested = True
                    st.session_state.test_model_name = ollama_model
                except Exception as e:
                    error_msg = str(e).lower()
                    if "connection" in error_msg or "connect" in error_msg:
                        st.error(f"❌ Cannot connect to Ollama at {st.session_state.get('ollama_host', 'localhost:11434')}")
                        st.info("Make sure Ollama is running: `ollama serve`")
                    elif "model" in error_msg or "not found" in error_msg:
                        st.error(f"❌ Model '{ollama_model}' not found")
                        st.info(f"Try pulling the model: `ollama pull {ollama_model}`")
                        
                        # Suggest popular models
                        st.markdown("**Popular models to try:**")
                        popular_models = [
                            "gemma3:12b - Google's Gemma 3 (12B parameters)",
                            "llama3.2:3b - Meta's Llama 3.2 (3B parameters)",
                            "qwen2.5:7b - Alibaba's Qwen 2.5 (7B parameters)",
                            "mistral:7b - Mistral 7B",
                            "codellama:7b - Meta's Code Llama"
                        ]
                        for model in popular_models:
                            st.markdown(f"• `{model}`")
                    else:
                        st.error(f"❌ Error: {e}")
                    st.session_state.model_tested = False
        
        # Show model status
        if st.session_state.get("model_tested") and st.session_state.get("test_model_name") == ollama_model:
            st.success(f"🟢 Model '{ollama_model}' verified")
        elif ollama_model in available_models:
            st.success(f"🟢 Model '{ollama_model}' available")
        elif ollama_model != current_ollama_model and ollama_model not in available_models:
            st.warning("⚠️ Model changed - click 'Test' to verify availability")
        
        # Show helpful info about the current model
        if ollama_model:
            col1, col2 = st.columns([3, 1])
            with col1:
                if ollama_model in available_models:
                    st.info(f"📦 Using model: **{ollama_model}** (available on your Ollama instance)")
                else:
                    st.info(f"📦 Using model: **{ollama_model}** (custom model)")
            with col2:
                if ollama_model in available_models:
                    show_info_btn = st.button("ℹ️", key="show_model_info", help="Show model details")
                    
                    if show_info_btn:
                        with st.spinner("Getting model information..."):
                            model_info = get_model_info(ollama_model, st.session_state.get("ollama_host", "http://localhost:11434"))
                            if model_info:
                                st.success("**Model Details:**")
                                st.markdown(f"• **Size:** {model_info['size']}")
                                st.markdown(f"• **Family:** {model_info['family']}")
                                st.markdown(f"• **Format:** {model_info['format']}")
                                if model_info['modified'] != "Unknown":
                                    try:
                                        modified_date = datetime.fromisoformat(model_info['modified'].replace('Z', '+00:00'))
                                        st.markdown(f"• **Last Modified:** {modified_date.strftime('%Y-%m-%d %H:%M')}")
                                    except:
                                        st.markdown(f"• **Last Modified:** {model_info['modified']}")
                            else:
                                st.error("Could not retrieve model information")
        
        # HuggingFace token for speaker diarization
        st.markdown("##### Speaker Diarization")
        if current_hf_token:
            masked_token = f"{current_hf_token[:4]}{'*' * 10}{current_hf_token[-4:]}"
            st.success(f"HuggingFace Token: {masked_token}")
            st.info("Speaker diarization is enabled")
            change_hf = st.checkbox("Change HuggingFace Token")
            if change_hf:
                hf_token = st.text_input("New HuggingFace Token", type="password")
                if hf_token:
                    st.session_state.hf_token = hf_token
        else:
            st.warning("HuggingFace Token not configured")
            st.info("**For better speaker differentiation**, get a HuggingFace token with access to pyannote/speaker-diarization-3.0")
            hf_token = st.text_input("HuggingFace Token", type="password", 
                                     help="Required for accurate speaker diarization")
            if hf_token:
                st.session_state.hf_token = hf_token
        
        # Save configuration button
        if st.button("Save Configuration"):
            result = save_config_settings()
            if result["huggingface"] or result["ollama"]:
                st.success("Configuration saved successfully!")
            else:
                st.info("No changes to save")
        
        # GPU Memory Management section
        st.markdown("---")
        st.subheader("🧠 GPU Memory Management")
        
        # Show number of active summarizers
        if "active_summarizers" in st.session_state:
            num_active = len(st.session_state.active_summarizers)
            if num_active > 0:
                st.info(f"📊 Active LLM instances: {num_active}")
            else:
                st.success("🟢 No active LLM instances")
        
        # Cleanup button
        if st.button("🧹 Free GPU Memory", help="Unload all LLM models from GPU memory"):
            with st.spinner("Freeing GPU memory..."):
                try:
                    cleanup_summarizers()
                    # Also cleanup any global instances
                    from summarization.summarizer import ConversationSummarizer
                    ConversationSummarizer.cleanup_all_instances()
                    st.success("✅ GPU memory freed successfully!")
                    st.rerun()  # Refresh to update the active instances count
                except Exception as e:
                    st.error(f"❌ Error freeing GPU memory: {e}")
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Record", "Upload", "Transcribe", "Summarize"])
      # Tab 1: Recording interface
    with tab1:
        st.header("Record Conversation")
        
        # Audio source selection
        st.subheader("Audio Source")
        available_sources = st.session_state.recorder.get_available_sources()
        
        source_options = {}
        for source in available_sources:
            source_options[source['name']] = source['id']
        
        selected_source_name = st.selectbox(
            "Choose audio source:",
            options=list(source_options.keys()),
            help="Select whether to record from microphone or system audio (computer playback)"
        )
        
        selected_source_id = source_options[selected_source_name]
        
        # Show description for selected source
        selected_source_info = next(s for s in available_sources if s['id'] == selected_source_id)
        st.info(f"📍 {selected_source_info['description']}")
        
        # System audio specific instructions
        if selected_source_id == "system_audio":
            st.warning("""
            **System Audio Recording Tips:**
            - Make sure audio is playing on your computer (YouTube, Zoom, etc.)
            - On Windows, you may need to enable "Stereo Mix" in sound settings
            - Some applications may block audio capture for security reasons
            - Test with a short recording first to ensure it's working
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display current recording status and timer
            if st.session_state.recording:
                elapsed_time = time.time() - st.session_state.recording_start_time
                st.markdown(f"### 🔴 Recording: {time_formatter(elapsed_time)}")
                
                # Display recording progress and stop button
                if elapsed_time >= 3600:  # 1 hour
                    stop_recording()
                    st.warning("Maximum recording time reached (1 hour)")
                else:
                    progress_percent = min(elapsed_time / 3600, 1.0)  # Scale to 1 hour max
                    st.progress(progress_percent)
                
                stop_button = st.button("Stop Recording", key="stop")
                if stop_button:
                    audio_file = stop_recording()
                    if audio_file:
                        st.success(f"Recording saved to {audio_file}")
            else:
                st.markdown("### 🎙️ Ready to Record")
                start_button = st.button("Start Recording", key="start")
                if start_button:
                    success = start_recording(selected_source_id)
                    if not success:
                        st.error("Failed to start recording. Please check your audio device settings.")
        
        with col2:
            # Display audio player for recorded file
            if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
                st.subheader("Recorded Audio")
                st.audio(st.session_state.audio_file)
                
                # Download button for audio
                st.markdown(get_binary_file_downloader_html(
                    st.session_state.audio_file, 'Audio File'), unsafe_allow_html=True)
                
                # Focus areas input
                st.subheader("Focus Areas (Optional)")
                focus_areas_input = st.text_input(
                    "Specify areas to focus on in the summary",
                    placeholder="e.g., budget, timeline, action items",
                    help="Enter comma-separated topics you want the summary to emphasize",
                    key="record_focus_areas"
                )
                
                # Process button
                process_btn = st.button("Transcribe & Summarize", key="process")
                if process_btn:
                    # Parse focus areas
                    focus_areas = []
                    if focus_areas_input and focus_areas_input.strip():
                        focus_areas = [area.strip() for area in focus_areas_input.split(',') if area.strip()]
                    process_audio(focus_areas)
        
        # Debug: Show available devices (optional)
        if st.checkbox("Show available audio devices (for debugging)", key="show_devices"):
            if st.button("List Audio Devices", key="list_devices"):
                st.text("Available Audio Devices:")
                devices_info = ""
                for device in st.session_state.recorder.devices:
                    device_type = "LOOPBACK" if device['is_loopback'] else "INPUT" 
                    devices_info += f"[{device['index']}] {device['name']}\n"
                    devices_info += f"    Type: {device_type}\n"
                    devices_info += f"    Channels: {device['channels']}\n"
                    devices_info += f"    Sample Rate: {device['rate']} Hz\n\n"
                
                st.text(devices_info)
      # Tab 2: Upload interface
    with tab2:
        st.header("📄 Upload Document or Transcript")
        st.write("Upload a document to extract text and generate summaries. Supported formats:")
        
        # Initialize document reader
        doc_reader = DocumentReader()
        supported_formats = doc_reader.get_supported_formats()
        
        # Show supported formats
        format_descriptions = {
            'docx': '📝 Microsoft Word documents (.docx)',
            'txt': '📄 Plain text files (.txt)', 
            'json': '🔗 JSON transcript files (.json)'
        }
        
        for fmt in supported_formats:
            if fmt in format_descriptions:
                st.write(f"• {format_descriptions[fmt]}")
        
        uploaded_file = st.file_uploader(
            "Choose a document or transcript file", 
            type=supported_formats,
            help="Select a document to extract text from and generate summaries"
        )
        
        if uploaded_file is not None:
            try:
                # Show file info
                st.info(f"📁 **File:** {uploaded_file.name} ({uploaded_file.size:,} bytes)")
                
                with st.spinner("Processing document..."):
                    # Use the document reader to process the file
                    transcript_data = doc_reader.read_document(
                        file_content=uploaded_file.getvalue(),
                        file_name=uploaded_file.name
                    )
                    
                    st.session_state.transcript = transcript_data
                    
                    # Show success message with file details
                    file_format = transcript_data.get('format', 'unknown').upper()
                    if 'metadata' in transcript_data:
                        meta = transcript_data['metadata']
                        st.success(
                            f"✅ **{file_format} document processed successfully!**\n\n"
                            f"📊 **Statistics:**\n"
                            f"• Word count: {meta.get('word_count', 'N/A'):,}\n"
                            f"• Character count: {meta.get('character_count', 'N/A'):,}\n"
                            f"• Paragraphs: {meta.get('total_paragraphs', 'N/A')}"
                        )
                    else:
                        st.success(f"✅ **{file_format} document processed successfully!**")
                    
                    # Show a preview of the extracted text
                    with st.expander("📖 Preview extracted text", expanded=False):
                        preview_text = transcript_data.get('transcript', '')
                        if len(preview_text) > 500:
                            st.text_area(
                                "Text preview (first 500 characters):",
                                preview_text[:500] + "...",
                                height=150,
                                disabled=True
                            )
                        else:
                            st.text_area(
                                "Full extracted text:",
                                preview_text,
                                height=150, 
                                disabled=True
                            )
                
            except Exception as e:
                st.error(f"❌ **Error processing document:** {str(e)}")
                st.info("💡 **Troubleshooting tips:**")
                st.write("• Make sure your file is not corrupted")
                st.write("• For DOCX files, ensure they're valid Microsoft Word documents")
                st.write("• For text files, ensure they use UTF-8 encoding")
                st.write("• For JSON files, ensure they have valid JSON syntax")
        
        # Show current transcript status
        if st.session_state.transcript:
            st.divider()
            
            # Focus areas input
            st.subheader("📋 Summary Focus Areas (Optional)")
            focus_areas_input = st.text_area(
                "What specific areas would you like the summary to focus on?",
                placeholder="Enter focus areas separated by commas (e.g., budget, timeline, action items, risks, decisions)",
                help="This helps the AI prioritize certain topics in the summary"
            )
            
            # Parse focus areas
            focus_areas = []
            if focus_areas_input.strip():
                focus_areas = [area.strip() for area in focus_areas_input.split(",") if area.strip()]
            
            # Process button
            if st.button("🤖 Generate Summary", key="process_uploaded", type="primary"):
                with st.spinner("Generating summary..."):
                    try:                        # Initialize summarizer with Ollama configuration
                        summarizer = ConversationSummarizer(
                            ollama_host=st.session_state.get("ollama_host", "http://localhost:11434"),
                            ollama_model=st.session_state.get("ollama_model", "gemma3:12b")
                        )
                        # Register for cleanup
                        register_summarizer(summarizer)
                        
                        # Generate summary with focus areas if provided
                        if focus_areas:
                            st.info(f"🎯 Focusing on: {', '.join(focus_areas)}")
                        
                        # Generate summary with focus areas using detailed format
                        summary = summarizer.summarize(
                            st.session_state.transcript, 
                            focus_areas=focus_areas if focus_areas else None,
                            use_detailed_format=True
                        )
                        
                        if summary.get("error"):
                            st.error(f"Error generating summary: {summary['error']}")
                        else:
                            st.session_state.summary = summary
                            
                            # Auto-save summary to output directory
                            saved_files = save_summary_to_output(summary, None, 'json')
                            if saved_files:
                                st.success(f"Summary generated and saved to: {list(saved_files.values())[0]}")
                            else:
                                st.success("Summary generated successfully!")
                    
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
                        st.error("Make sure Ollama is running and the model is available.")
    
    # Tab 3: Transcription view
    with tab3:
        st.header("Transcription")
        
        if st.session_state.transcript:
            transcript = st.session_state.transcript
            
            # Display transcript metadata
            st.subheader("Transcript Info")
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.write(f"**Duration:** {transcript['segments'][-1]['timestamp']}")
            with col2:
                st.write(f"**Speakers:** {len(transcript.get('speakers', ['Unknown']))}")
            with col3:
                # Display speaker legend with colors
                if transcript.get('speakers'):
                    legend_html = ""
                    for speaker in transcript.get('speakers'):
                        color = get_speaker_color(speaker)
                        legend_html += f'<span style="display:inline-block; margin-right:10px;"><span style="background-color:{color}; color:white; padding:2px 8px; border-radius:3px;">{speaker}</span></span>'
                    st.markdown(f"**Speaker Legend:** {legend_html}", unsafe_allow_html=True)
            
            # Option to download transcript as JSON
            transcript_json = json.dumps(transcript, indent=2)
            st.download_button(
                label="Download Transcript (JSON)",
                data=transcript_json,
                file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Display transcript with timestamps and speakers
            st.subheader("Conversation Transcript")
            
            # Group segments by speaker for better readability
            current_speaker = None
            speaker_segments = []
            
            for segment in transcript["segments"]:
                if segment["speaker"] != current_speaker:
                    current_speaker = segment["speaker"]
                    speaker_segments.append([])
                speaker_segments[-1].append(segment)
            
            # Display segments grouped by speaker
            for speaker_group in speaker_segments:
                if not speaker_group:
                    continue
                    
                speaker = speaker_group[0]["speaker"]
                speaker_color = get_speaker_color(speaker)
                
                # Speaker header with styled box
                st.markdown(
                    f'<div style="background-color:{speaker_color}; color:white; padding:5px 10px; border-radius:5px; margin-bottom:5px;">'
                    f'<strong>{speaker}</strong>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Speaker's segments
                with st.container():
                    for segment in speaker_group:
                        col1, col2 = st.columns([1, 9])
                        with col1:
                            st.code(segment["timestamp"])
                        with col2:
                            st.markdown(segment["text"])
        else:
            st.info("No transcript available. Record a conversation and process it first.")
    
    # Tab 4: Summary view
    with tab4:
        st.header("Summary & Action Points")
        
        if st.session_state.summary:
            summary = st.session_state.summary
            
            # Check if we have detailed format data
            has_detailed_format = summary.get("format_used") == "detailed"
            
            if has_detailed_format:
                # Display detailed meeting minutes format
                if summary.get("meeting_overview"):
                    st.subheader("🏢 Meeting Overview")
                    st.write(summary["meeting_overview"])
                
                if summary.get("discussion_highlights"):
                    st.subheader("💬 Discussion Highlights")
                    st.write(summary["discussion_highlights"])
                
                if summary.get("decisions_made"):
                    st.subheader("✅ Decisions Made")
                    for i, decision in enumerate(summary["decisions_made"], 1):
                        st.markdown(f"- {decision}")
                
                if summary.get("action_points"):
                    st.subheader("📋 Action Items")
                    for i, action in enumerate(summary["action_points"], 1):
                        st.markdown(f"- {action}")
                else:
                    st.info("No action items identified in this conversation.")
                
                if summary.get("next_steps"):
                    st.subheader("🔄 Next Steps/Follow-Up")
                    st.write(summary["next_steps"])
            else:
                # Display simple format (backward compatibility)
                st.subheader("Conversation Summary")
                st.write(summary["summary"])
                
                # Display action points
                st.subheader("Action Points")
                if summary["action_points"]:
                    for i, action in enumerate(summary["action_points"], 1):
                        st.markdown(f"- {action}")
                else:
                    st.info("No action points identified in this conversation.")
              # Display participants with their colors
            if summary.get("participants"):
                st.subheader("Participants")
                participants_html = ""
                for participant in summary.get("participants", []):
                    if participant != "Unknown":
                        color = get_speaker_color(participant)
                        participants_html += f'<span style="display:inline-block; margin-right:10px; margin-bottom:10px;"><span style="background-color:{color}; color:white; padding:2px 8px; border-radius:3px;">{participant}</span></span>'
                if participants_html:
                    st.markdown(participants_html, unsafe_allow_html=True)
            
            # Download section with format selection
            st.subheader("Download Summary")
            
            # Initialize formatter
            formatter = SummaryFormatter()
            
            # Format selection
            col1, col2 = st.columns([1, 1])
            with col1:
                download_format = st.selectbox(
                    "Select format:",
                    options=formatter.get_supported_formats(),
                    format_func=lambda x: {
                        'json': 'JSON (JavaScript Object Notation)',
                        'txt': 'Plain Text (.txt)',
                        'docx': 'Microsoft Word (.docx)'
                    }.get(x, x.upper())
                )
            
            with col2:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                if download_format == 'json':
                    # JSON download
                    summary_json = json.dumps(summary, indent=2)
                    st.download_button(
                        label="📄 Download as JSON",
                        data=summary_json,
                        file_name=f"summary_{timestamp}.json",
                        mime="application/json"
                    )
                
                elif download_format == 'txt':
                    # TXT download - use formatter to generate content
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                            temp_path = tmp_file.name
                        
                        # Use formatter to create TXT content
                        formatter._save_txt(summary, temp_path, include_metadata=True)
                        
                        with open(temp_path, 'r', encoding='utf-8') as f:
                            txt_content = f.read()
                        
                        # Clean up temp file
                        os.unlink(temp_path)
                        
                        st.download_button(
                            label="📝 Download as TXT",
                            data=txt_content,
                            file_name=f"summary_{timestamp}.txt",
                            mime="text/plain"
                        )
                    except Exception as e:
                        st.error(f"Error generating TXT format: {e}")
                
                elif download_format == 'docx':
                    # DOCX download - use formatter to generate content
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
                            temp_path = tmp_file.name
                        
                        # Use formatter to create DOCX file
                        formatter._save_docx(summary, temp_path, include_metadata=True)
                        
                        with open(temp_path, 'rb') as f:
                            docx_content = f.read()
                        
                        # Clean up temp file
                        os.unlink(temp_path)
                        
                        st.download_button(
                            label="📄 Download as DOCX",
                            data=docx_content,
                            file_name=f"summary_{timestamp}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    except Exception as e:
                        if "python-docx" in str(e):
                            st.error("DOCX format requires python-docx package. Please install it.")
                        else:
                            st.error(f"Error generating DOCX format: {e}")
            
            # Option to download all formats at once
            st.write("")
            if st.button("📦 Download All Formats"):
                try:
                    import tempfile
                    import zipfile
                    import io
                    
                    # Create a ZIP file in memory containing all formats
                    zip_buffer = io.BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # Add JSON
                        summary_json = json.dumps(summary, indent=2)
                        zip_file.writestr(f"summary_{timestamp}.json", summary_json)
                        
                        # Add TXT
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                            temp_txt_path = tmp_file.name
                        formatter._save_txt(summary, temp_txt_path, include_metadata=True)
                        with open(temp_txt_path, 'r', encoding='utf-8') as f:
                            txt_content = f.read()
                        zip_file.writestr(f"summary_{timestamp}.txt", txt_content)
                        os.unlink(temp_txt_path)
                        
                        # Add DOCX if available
                        if 'docx' in formatter.get_supported_formats():
                            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
                                temp_docx_path = tmp_file.name
                            formatter._save_docx(summary, temp_docx_path, include_metadata=True)
                            with open(temp_docx_path, 'rb') as f:
                                docx_content = f.read()
                            zip_file.writestr(f"summary_{timestamp}.docx", docx_content)
                            os.unlink(temp_docx_path)
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download ZIP File",
                        data=zip_buffer.getvalue(),
                        file_name=f"summary_all_formats_{timestamp}.zip",
                        mime="application/zip"
                    )
                    
                except Exception as e:
                    st.error(f"Error creating multi-format download: {e}")
        else:
            st.info("No summary available. Record a conversation and process it first.")

if __name__ == "__main__":
    main()