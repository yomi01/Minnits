import streamlit as st
import os
import sys
import asyncio
import nest_asyncio
import time
import json
import base64
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
from utils.api_keys import load_api_keys, save_api_keys

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
    
# Load configuration if available
config = load_api_keys()
if config["huggingface_token"] and "hf_token" not in st.session_state:
    st.session_state.hf_token = config["huggingface_token"]
if config["ollama_host"] and "ollama_host" not in st.session_state:
    st.session_state.ollama_host = config["ollama_host"]
if config["ollama_model"] and "ollama_model" not in st.session_state:
    st.session_state.ollama_model = config["ollama_model"]

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
def start_recording():
    """Start audio recording"""
    st.session_state.recording = True
    st.session_state.recording_start_time = time.time()
    st.session_state.recorder.start_recording()

def stop_recording():
    """Stop audio recording"""
    if st.session_state.recording:
        st.session_state.recording = False
        audio_file = st.session_state.recorder.stop_recording()
        st.session_state.audio_file = audio_file
        return audio_file
    return None

def process_audio():
    """Process recorded audio file (transcribe and summarize)"""
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
        try:
            # Initialize summarizer with Ollama configuration
            summarizer = ConversationSummarizer(
                ollama_host=st.session_state.get("ollama_host", "http://localhost:11434"),
                ollama_model=st.session_state.get("ollama_model", "gemma3:12b")  # Use gemma3:12b by default
            )
            
            # Generate summary
            summary = summarizer.summarize(transcript)
            
            if summary.get("error"):
                st.error(f"Error generating summary: {summary['error']}")
                return
            
            st.session_state.summary = summary
            
        except Exception as e:
            st.error(f"Error generating summary: {str(e)}")
            st.error("Make sure Ollama is running and the model is available.")
            return
        
    # Reset speaker colors when processing a new recording
    st.session_state.speaker_colors = {}

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
            
        ollama_model = st.text_input("Ollama Model", 
                                    value=current_ollama_model,
                                    help="Model to use for transcription and summarization")
        if ollama_model:
            st.session_state.ollama_model = ollama_model
        
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
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Record", "Upload", "Transcribe", "Summarize"])
    
    # Tab 1: Recording interface
    with tab1:
        st.header("Record Conversation")
        
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
                    start_recording()
        
        with col2:
            # Display audio player for recorded file
            if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
                st.subheader("Recorded Audio")
                st.audio(st.session_state.audio_file)
                
                # Download button for audio
                st.markdown(get_binary_file_downloader_html(
                    st.session_state.audio_file, 'Audio File'), unsafe_allow_html=True)
                
                # Process button
                process_btn = st.button("Transcribe & Summarize", key="process")
                if process_btn:
                    process_audio()
    
    # Tab 2: Upload interface
    with tab2:
        st.header("Upload Transcript")
        
        uploaded_file = st.file_uploader("Choose a transcript file", type=['txt', 'json'])
        
        if uploaded_file is not None:
            try:
                # Try to parse as JSON first
                try:
                    transcript_data = json.loads(uploaded_file.getvalue().decode())
                    if isinstance(transcript_data, dict) and "segments" in transcript_data:
                        st.session_state.transcript = transcript_data
                        st.success("Transcript uploaded successfully!")
                    else:
                        # If JSON doesn't have expected structure, treat as plain text
                        raise ValueError("Invalid transcript format")
                except json.JSONDecodeError:
                    # If not JSON, process as plain text
                    text_content = uploaded_file.getvalue().decode()
                    
                    # Create a basic transcript structure
                    transcript_data = {
                        "transcript": text_content,
                        "segments": [{
                            "timestamp": "00:00:00",
                            "text": text_content,
                            "speaker": "Speaker 1"
                        }]
                    }
                    st.session_state.transcript = transcript_data
                    st.success("Text file uploaded and converted to transcript format!")
                
                # Process button
                if st.button("Generate Summary", key="process_uploaded"):
                    with st.spinner("Generating summary..."):
                        try:
                            # Initialize summarizer with Ollama configuration
                            summarizer = ConversationSummarizer(
                                ollama_host=st.session_state.get("ollama_host", "http://localhost:11434"),
                                ollama_model=st.session_state.get("ollama_model", "gemma3:12b")
                            )
                            
                            # Generate summary
                            summary = summarizer.summarize(st.session_state.transcript)
                            
                            if summary.get("error"):
                                st.error(f"Error generating summary: {summary['error']}")
                            else:
                                st.session_state.summary = summary
                                st.success("Summary generated successfully!")
                        
                        except Exception as e:
                            st.error(f"Error generating summary: {str(e)}")
                            st.error("Make sure Ollama is running and the model is available.")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.info("Please upload a valid transcript file (JSON with segments or plain text)")
    
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
            
            # Display summary
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
            
            # Option to download summary as JSON
            summary_json = json.dumps(summary, indent=2)
            st.download_button(
                label="Download Summary (JSON)",
                data=summary_json,
                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Option to download as PDF (placeholder - would need additional library)
            st.write("")
            st.info("PDF export feature coming soon!")
        else:
            st.info("No summary available. Record a conversation and process it first.")

if __name__ == "__main__":
    main()