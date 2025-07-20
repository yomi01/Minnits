# Minnits - Local Recorder & Summarizer 

A powerful tool that records conversations, transcribes them with speaker recognition, and creates intelligent summaries with action items. It can also process uploaded transcripts and text files for summarization.

## Features

- 🎙️ **Audio Recording**: Record conversations for up to one hour
- 🔊 **Speaker Diarization**: Automatically recognize different speakers in the conversation
- ⏱️ **Timestamps**: Generate timestamps every 5 seconds for easy reference
- 📝 **Transcription**: Convert speech to text with high accuracy using OpenAI Whisper
- 📄 **Text Upload**: Upload existing transcripts or text files for summarization
- 📊 **Summarization**: Create concise summaries of conversations or uploaded text
- 📋 **Detailed Meeting Minutes**: Professional meeting minutes format with structured sections
- ✅ **Action Items**: Extract action points and tasks mentioned during the conversation
- 🖥️ **User Interface**: Clean and intuitive Streamlit interface with multiple tabs
- 🔄 **Command Line Interface**: Support for CLI operations for automation
- 🧠 **GPU Memory Management**: Automatic cleanup of LLM models from GPU memory when app closes

## System Requirements

### Operating System
- Windows 10 or later
- macOS 10.15 (Catalina) or later
- Linux (Ubuntu 20.04 or equivalent)

### Hardware Requirements
- Microphone (for audio recording)
- Speakers/Headphones (for audio playback)
- Minimum 16GB RAM
- 50GB free disk space
- CPU: Dual-core processor or better

### Software Dependencies
- Python 3.8 or higher
- Ollama (latest version)
  - Required for text summarization
  - Installation instructions: https://ollama.ai/download
- PortAudio (automatically installed with PyAudio on Windows)
  - Linux users may need to install: `sudo apt-get install portaudio19-dev`
  - macOS users may need: `brew install portaudio`

### Network Requirements
- Internet connection (for model downloads and Ollama updates)
- Access to HuggingFace API (optional, for speaker diarization)

## GPU Memory Management

This application now includes automatic GPU memory management to prevent Ollama models from staying loaded in GPU memory after the app closes:

### Automatic Cleanup
- **On App Shutdown**: Models are automatically unloaded when the application terminates (CLI or UI)
- **Signal Handling**: Properly handles Ctrl+C, termination signals, and Windows break signals
- **Instance Tracking**: Tracks all active ConversationSummarizer instances for cleanup

### Manual Control (UI)
- **GPU Memory Status**: View the number of active LLM instances in the sidebar
- **Free GPU Memory Button**: Manually unload all models from GPU memory
- **Real-time Monitoring**: See active instance count and cleanup status

### Technical Details
- Uses Ollama's `keep_alive=0` parameter to immediately unload models
- Fallback mechanism loads a tiny model to free GPU memory if primary method fails
- Cleanup happens automatically on module destruction and application exit

This ensures optimal GPU memory usage and prevents memory leaks when switching between different models or closing the application.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyAudio dependencies (on Windows, this should install automatically)
- Ollama (for text summarization)

### Setup

1. Clone this repository:
   ```
   git clone https://your-repository-url.git
   cd note_summarizer
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On macOS/Linux
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up API keys and configuration:
   - Create a `.env` file in the root directory
   - Add your configuration:
     ```
     HUGGINGFACE_TOKEN=your_huggingface_token  # Optional, for speaker diarization
     OLLAMA_HOST=http://localhost:11434  # Ollama API endpoint
     OLLAMA_MODEL=gemma3:12b  # Default model for summarization
     ```

### Environment Setup

1. Copy the example environment file:
   ```
   cp .env.example .env
   ```

2. Configure your environment variables in `.env`:
   - `HUGGINGFACE_TOKEN`: (Optional) Your HuggingFace token for speaker diarization
   - `OLLAMA_HOST`: URL where Ollama is running (default: http://localhost:11434)
   - `OLLAMA_MODEL`: Ollama model to use (default: gemma3:12b)

**Note**: Never commit your `.env` file to version control. It's listed in `.gitignore` for security.

## Usage

### Graphical User Interface

Launch the web interface:

```
python main.py --ui
```

The application has four main tabs:

#### 1. Record Tab
- Click "Start Recording" to begin capturing audio
- Recording progress and duration are shown in real-time
- Click "Stop Recording" when finished
- Listen to the recording and download the audio file
- Click "Transcribe & Summarize" to process the recording

#### 2. Upload Tab
- Upload existing transcript files (JSON format) or plain text files
- Supported formats:
  - JSON files with proper transcript structure
  - Plain text files (automatically converted to transcript format)
- Click "Generate Summary" to process the uploaded content

#### 3. Transcribe Tab
- View the complete transcript with timestamps
- See speaker identification and color coding
- Download the transcript in JSON format
- Transcript is organized by speaker for easy reading

#### 4. Summarize Tab
- View the generated summary of the conversation
- See extracted action points and tasks
- View participant information with color coding
- Download the summary in JSON format

### Command Line Interface

#### Record a conversation:

```
python main.py --record --duration 300
```
This will record for 5 minutes (300 seconds) and save the audio file to the `output` directory.

#### Transcribe an audio file:

```
python main.py --transcribe path/to/audiofile.wav --model base
```

Available models: `tiny`, `base`, `small`, `medium`, `large` (larger models are more accurate but slower).

#### Summarize a transcript:

```
python main.py --summarize path/to/transcript.json
```

#### Complete pipeline:

```
python main.py --record --duration 600 --model base --output my_meeting
```

This will record for 10 minutes, transcribe the audio, and generate a summary, saving all files to the `my_meeting` directory.

## Focus Areas Feature

The note summarizer now supports **focus areas** - you can specify specific topics or areas that you want the summary to emphasize. This helps create more targeted and relevant summaries based on your needs.

### Using Focus Areas

#### Command Line Interface (CLI)
Use the `--focus` parameter to specify areas to focus on:

```bash
# Focus on specific topics
python main.py --summarize transcript.json --focus "budget" "timeline" "risks"

# Focus on a single topic
python main.py --transcribe audio.wav --focus "action items"

# Multiple focus areas with document processing
python main.py --document meeting_notes.docx --focus "decisions" "next steps" "deadlines"
```

#### Streamlit Web Interface
1. **Upload Tab**: Enter focus areas in the "Focus Areas" text input field (comma-separated)
2. **Recording Tab**: Specify focus areas before clicking "Transcribe & Summarize"

### Examples of Focus Areas
- **Business Meetings**: "budget", "timeline", "action items", "decisions", "risks"
- **Product Planning**: "features", "deadlines", "resources", "priorities"  
- **Academic Sessions**: "key concepts", "assignments", "exam topics"
- **Project Reviews**: "progress", "blockers", "next steps", "deliverables"

### How It Works
When you specify focus areas, the AI will:
- Pay particular attention to those topics in the conversation
- Emphasize relevant information in the summary
- Ensure action items related to focus areas are highlighted
- Structure the summary to prioritize your specified areas

## Detailed Meeting Minutes Format

The note summarizer now supports a **detailed meeting minutes format** that creates professional, structured summaries similar to formal meeting minutes. This feature uses LangChain prompt templates for enhanced quality and consistency.

### Features of Detailed Format

The detailed format provides:
- **Meeting Overview**: Date, time, purpose, and key participants
- **Discussion Highlights**: Summary of main topics with critical points
- **Decisions Made**: Bullet points of decisions and agreements
- **Action Items**: Tasks with responsible persons and deadlines
- **Next Steps/Follow-Up**: Upcoming meetings and unresolved items

### Using Detailed Format

#### Command Line Interface (CLI)
The detailed format is now the **default** for all summarization:

```bash
# Use detailed format (default)
python main.py --summarize transcript.json

# Explicitly request detailed format
python main.py --summarize transcript.json --detailed

# Use simple format instead
python main.py --summarize transcript.json --no-detailed

# Combine with focus areas
python main.py --summarize transcript.json --focus "budget" "timeline" --detailed
```

#### Streamlit Web Interface
The web interface automatically uses the detailed format by default. The summary will display:
- 🏢 Meeting Overview
- 💬 Discussion Highlights  
- ✅ Decisions Made
- 📋 Action Items
- 🔄 Next Steps/Follow-Up

### Output Formats
The detailed format is supported in all output formats:
- **JSON**: Complete structured data with all sections
- **DOCX**: Professional Word document with proper headings
- **TXT**: Plain text with clear section divisions

### Backward Compatibility
- Existing scripts and integrations continue to work
- The simple format is still available using `--no-detailed`
- All output formats support both detailed and simple modes

## Dynamic Model Switching

The application now supports **dynamic model switching** directly from the user interface, allowing you to change Ollama models without restarting the application.

### Features

- **🔄 Model Discovery**: Automatically detects available models from your Ollama instance
- **✅ Model Validation**: Test model availability before using them
- **ℹ️ Model Information**: View detailed information about each model (size, family, format)
- **🎯 Smart Recommendations**: Get model suggestions based on use case
- **💾 Configuration Persistence**: Save your preferred model settings

### Using Model Switching

#### Streamlit Web Interface
1. **Open Settings**: Look for the "Settings" section in the sidebar
2. **Select Model**: Choose from the dropdown of available models or enter a custom model name
3. **Test Model**: Click the "Test" button to verify model availability
4. **View Details**: Click the "ℹ️" button to see model information
5. **Save Configuration**: Click "Save Configuration" to persist your settings

#### Popular Model Recommendations

**💼 Business Meetings:**
- `gemma3:12b` - Excellent for formal meeting minutes
- `llama3.2:8b` - Good balance of speed and quality
- `qwen2.5:7b` - Strong multilingual support

**⚡ Quick Summaries:**
- `llama3.2:3b` - Fast processing
- `gemma3:3b` - Good quality, small size
- `phi3:3.8b` - Efficient for simple tasks

**🌍 Multilingual Content:**
- `qwen2.5:7b` - Best for Chinese/English
- `gemma3:12b` - Good multilingual support
- `mistral:7b` - European languages

### Model Management Tips

- **Installing Models**: Use `ollama pull <model-name>` to install new models
- **Model Size**: Larger models (7B+) generally provide better quality but are slower
- **System Resources**: Consider your available RAM when choosing models
- **Use Cases**: Match model capabilities to your specific needs (speed vs. quality)

### Example Output
With focus areas `["budget", "timeline"]`, a summary might emphasize:
> "The conversation centered on budget allocation ($50,000 available) and timeline constraints (3-week deadline)..."

Without focus areas, the same conversation might produce a more general summary covering all topics equally.

## How It Works

1. **Recording**: The app uses PyAudio to capture audio from your default input device.
2. **Transcription**: OpenAI Whisper processes the audio file to generate an accurate transcript.
3. **Speaker Diarization**: The PyAnnote audio library analyzes voice patterns to identify different speakers.
4. **Text Processing**: Handles both recorded audio and uploaded text content.
5. **Summarization**: Uses Ollama's language models to generate summaries and extract action items.

## Project Structure

```
note_summarizer/
├── audio_processing/
│   └── recorder.py         # Audio recording functionality
├── transcription/
│   └── transcriber.py      # Speech-to-text with speaker diarization
├── summarization/
│   └── summarizer.py       # AI-powered summarization & action item extraction
├── ui/
│   └── app.py              # Streamlit user interface with multiple tabs
├── utils/
│   └── api_keys.py         # Configuration and API key management
├── main.py                 # Main application entry point
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

## Configuration

### Ollama Setup
1. Install Ollama on your system
2. Start the Ollama service
3. Configure the host URL and model in the application settings
4. Default configuration:
   - Host: http://localhost:11434
   - Model: gemma3:12b

### Speaker Diarization
- Optional but recommended for better speaker recognition
- Requires a HuggingFace token with access to pyannote/speaker-diarization-3.0
- Configure in the application settings

## Extending the Application

### Adding New Features

The modular design makes it easy to extend:

- **Custom Summarization**: Modify the prompts in `summarizer.py`
- **Additional UI Elements**: Extend the Streamlit interface in `ui/app.py`
- **Different Audio Formats**: Update the recorder in `audio_processing/recorder.py`

### Integration with Other Tools

The application can be integrated with:
- Calendar applications to automatically record and summarize meetings
- Team collaboration tools to share summaries
- Task management systems to import action items

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper and GPT models
- PyAnnote team for the speaker diarization library
- Streamlit for the user interface framework
