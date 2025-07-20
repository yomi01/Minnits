import os
import sys
import argparse
import subprocess
import time
import signal
import atexit
from utils.api_keys import initialize_api_keys, prompt_for_api_keys

# Global cleanup function for signal handling
def cleanup_on_exit():
    """Clean up resources on application exit"""
    try:
        from summarization.summarizer import ConversationSummarizer
        ConversationSummarizer.cleanup_all_instances()
    except ImportError:
        pass  # Module might not be imported yet

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals to clean up resources"""
    print(f"\nReceived shutdown signal, cleaning up...")
    cleanup_on_exit()
    sys.exit(0)

def register_main_signal_handlers():
    """Register signal handlers for the main application"""
    try:
        # Register signal handlers and exit cleanup
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
        atexit.register(cleanup_on_exit)
    except Exception as e:
        print(f"Warning: Could not register signal handlers: {e}")

# Register cleanup and signal handlers
register_main_signal_handlers()

def main():
    """
    Main entry point for the Conversation Recorder & Summarizer application.
    """
    parser = argparse.ArgumentParser(
        description='Conversation Recorder & Summarizer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--ui', 
        action='store_true',
        help='Launch the Streamlit user interface'
    )
    
    parser.add_argument(
        '--record',
        action='store_true',
        help='Start recording immediately (CLI mode)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Recording duration in seconds (CLI mode, default: 60)'
    )
    
    parser.add_argument(
        '--transcribe',
        metavar='AUDIO_FILE',
        help='Transcribe an audio file (CLI mode)'
    )
    
    parser.add_argument(
        '--document',
        metavar='DOCUMENT_FILE',
        help='Process a document file (DOCX, TXT, JSON) for summarization (CLI mode)'
    )
    
    parser.add_argument(
        '--summarize',
        metavar='TRANSCRIPT_FILE',
        help='Summarize a transcript JSON file (CLI mode)'
    )
    
    parser.add_argument(
        '--output',
        metavar='OUTPUT_DIR',
        default='output',
        help='Output directory for files (CLI mode)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'docx', 'txt', 'all'],
        default='json',
        help='Output format for summaries: json (default), docx, txt, or all'
    )
    
    parser.add_argument(
        '--ollama-host',
        default='http://localhost:11434',
        help='Ollama API host URL (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--ollama-model',
        default='gemma3:12b',
        help='Ollama model to use (default: gemma3:12b)'
    )
    
    parser.add_argument(
        '--setup-keys',
        action='store_true',
        help='Set up HuggingFace token and Ollama configuration'
    )
    
    parser.add_argument(
        '--focus',
        metavar='FOCUS_AREAS',
        nargs='*',  # Allow multiple focus areas
        help='Specific areas to focus on in the summary (e.g., --focus "budget" "timeline" "risks")'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Use detailed meeting minutes format (default: True). Use --no-detailed for simple format'
    )
    
    parser.add_argument(
        '--no-detailed',
        dest='detailed',
        action='store_false',
        help='Use simple summary format instead of detailed meeting minutes'
    )
    
    args = parser.parse_args()
    
    # Handle configuration setup
    if args.setup_keys:
        keys = prompt_for_api_keys()
        print("\nConfiguration has been set up successfully!")
        return
    
    # Initialize configuration
    keys = initialize_api_keys()
    
    # Update Ollama configuration from command line if provided
    if args.ollama_host:
        keys["ollama_host"] = args.ollama_host
    if args.ollama_model:
        keys["ollama_model"] = args.ollama_model
    
    # Set environment variables for the current session
    if keys["huggingface_token"]:
        os.environ["HUGGINGFACE_TOKEN"] = keys["huggingface_token"]
    os.environ["OLLAMA_HOST"] = keys["ollama_host"]
    os.environ["OLLAMA_MODEL"] = keys["ollama_model"]
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Launch Streamlit UI if requested
    if args.ui:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ui_path = os.path.join(script_dir, 'ui', 'app.py')
        
        print("Starting Conversation Recorder & Summarizer UI...")
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', ui_path])
        return
    
    # CLI mode for recording    if args.record:
        from audio_processing.recorder import AudioRecorder
        
        print(f"Recording for {args.duration} seconds...")
        recorder = AudioRecorder(output_dir=args.output)
        recorder.start_recording()
        
        # Monitor recording time
        start_time = time.time()
        try:
            while time.time() - start_time < args.duration:
                elapsed = time.time() - start_time
                remaining = args.duration - elapsed
                print(f"\rRecording: {int(elapsed)}s / {args.duration}s (Press Ctrl+C to stop)", end='')
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nRecording stopped by user")
        
        audio_file = recorder.stop_recording()
        print(f"\nRecording saved to {audio_file}")
        
        # Ask if user wants to transcribe the recording
        if input("Transcribe this recording? (y/n): ").lower() == 'y':
            args.transcribe = audio_file
    
    # CLI mode for transcription
    if args.transcribe:
        from transcription.transcriber import Transcriber
        import json
        
        audio_file = args.transcribe
        if not os.path.exists(audio_file):
            print(f"Error: Audio file not found: {audio_file}")
            return
        
        print(f"Transcribing {audio_file}...")
        transcriber = Transcriber(
            ollama_host=keys["ollama_host"],
            ollama_model=keys["ollama_model"],
            auth_token=keys.get("huggingface_token")
        )
        transcript = transcriber.transcribe(audio_file, timestamp_interval=5)
        
        # Save transcript to output directory
        filename = os.path.splitext(os.path.basename(audio_file))[0]
        transcript_file = os.path.join(args.output, f"{filename}_transcript.json")
        with open(transcript_file, 'w') as f:
            json.dump(transcript, f, indent=2)
        
        print(f"Transcript saved to {transcript_file}")
        
        # Display a sample of the transcript
        print("\nTranscript preview:")
        for i, segment in enumerate(transcript["segments"][:5]):
            print(f"{segment['timestamp']} | {segment['speaker']}: {segment['text']}")
        
        if len(transcript["segments"]) > 5:
            print("...")
        
        # Ask if user wants to summarize the transcript
        if input("Summarize this transcript? (y/n): ").lower() == 'y':
            args.summarize = transcript_file    # CLI mode for document processing and summarization
    if args.document:
        from document_processing.reader import DocumentReader
        import json
        
        document_file = args.document
        if not os.path.exists(document_file):
            print(f"Error: Document file not found: {document_file}")
            return
        
        print(f"Processing document {document_file}...")
        doc_reader = DocumentReader()
        try:
            # Read the document
            transcript_data = doc_reader.read_document(file_path=document_file)
              # Show document info
            print(f"Document format: {transcript_data.get('format', 'unknown').upper()}")
            if 'metadata' in transcript_data:
                meta = transcript_data['metadata']
                print(f"Word count: {meta.get('word_count', 'N/A'):,}")
                print(f"Character count: {meta.get('character_count', 'N/A'):,}")
                if 'total_paragraphs' in meta:
                    print(f"Paragraphs: {meta['total_paragraphs']}")
            
            # Save processed document as transcript
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            transcript_file = os.path.join(args.output, f"document_transcript_{timestamp}.json")
            
            with open(transcript_file, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            print(f"Document processed and saved as: {transcript_file}")
            
            # Ask if user wants to summarize the document
            if input("Summarize this document? (y/n): ").lower() == 'y':
                args.summarize = transcript_file
                
        except Exception as e:
            print(f"Error processing document: {e}")
            return
    
    # CLI mode for summarization
    if args.summarize:
        from summarization.summarizer import ConversationSummarizer
        import json
        
        transcript_file = args.summarize
        if not os.path.exists(transcript_file):
            print(f"Error: Transcript file not found: {transcript_file}")
            return
        
        # Load transcript
        with open(transcript_file, 'r') as f:
            transcript = json.load(f)
        
        print("Generating summary...")
        summarizer = ConversationSummarizer(
            ollama_host=keys["ollama_host"],
            ollama_model=keys["ollama_model"]
        )
        
        try:
            # Prepare focus areas for the summarizer
            focus_areas = args.focus if args.focus else None
            if focus_areas:
                print(f"Focus areas: {', '.join(focus_areas)}")
            
            # Set detailed format preference (default True unless --no-detailed was specified)
            use_detailed_format = getattr(args, 'detailed', True)
            
            print(f"Using {'detailed meeting minutes' if use_detailed_format else 'simple summary'} format")
            
            # Generate summary with focus areas and format preference
            summary = summarizer.summarize(
                transcript, 
                focus_areas=focus_areas, 
                use_detailed_format=use_detailed_format
            )
        finally:
            # Always clean up the summarizer instance
            summarizer.cleanup()
        
        # Save summary using the new formatter
        from summarization.formatter import SummaryFormatter
        
        formatter = SummaryFormatter()
        filename = os.path.splitext(os.path.basename(transcript_file))[0]
        output_base = os.path.join(args.output, f"{filename}_summary")
        
        if args.format == 'all':
            saved_files = formatter.save_multiple_formats(summary, output_base)
            print("Summary saved in multiple formats:")
            for fmt, file_path in saved_files.items():
                print(f"  {fmt.upper()}: {file_path}")
        else:
            summary_file = formatter.save_summary(summary, output_base, args.format)
            print(f"Summary saved to {summary_file}")
        
        # Display summary
        print("\nSummary:")
        print(summary["summary"])
        
        print("\nAction Points:")
        if summary["action_points"]:
            for i, action in enumerate(summary["action_points"], 1):
                print(f"{i}. {action}")
        else:
            print("No action points identified.")
    
    # If no action specified, show help
    if not (args.ui or args.record or args.transcribe or args.summarize or args.setup_keys):
        parser.print_help()
        print("\nTip: Run with --setup-keys to configure the application.")

if __name__ == "__main__":
    main()