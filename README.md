# Minnits - Local Recorder & Summarizer 

A powerful tool that records conversations, transcribes them with speaker recognition, and creates intelligent summaries with action items. It can also process uploaded transcripts and text files for summarization.

## Features

- 🎙️ **Audio Recording**: Record conversations for up to one hour
- 🔊 **Speaker Diarization**: Automatically recognize different speakers in the conversation
- ⏱️ **Timestamps**: Generate timestamps every 5 seconds for easy reference
- 📝 **Transcription**: Convert speech to text with high accuracy using OpenAI Whisper
- 📄 **Text Upload**: Upload existing transcripts or text files for summarization
- 📊 **Summarization**: Create concise summaries of conversations or uploaded text
- ✅ **Action Items**: Extract action points and tasks mentioned during the conversation
- 🖥️ **User Interface**: Clean and intuitive Streamlit interface with multiple tabs
- 🔄 **Command Line Interface**: Support for CLI operations for automation

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
