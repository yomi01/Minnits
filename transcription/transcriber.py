import os
import json
import whisper
import requests
import librosa
import numpy as np
import soundfile as sf
import tempfile
from typing import Dict, List, Tuple, Optional, Set
import re
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for librosa
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    logger.warning("Warning: librosa not found. Installing required packages...")
    LIBROSA_AVAILABLE = False

# Conditional import for pyannote
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    logger.warning("Warning: pyannote.audio not found. Speaker diarization will not be available.")
    PYANNOTE_AVAILABLE = False

class Transcriber:
    def __init__(self, model_size="base", auth_token=None):
        """
        Initialize the transcriber with Whisper model and pyannote for speaker diarization
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            auth_token: HuggingFace auth token for pyannote.audio
        """
        # Check for required packages
        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "librosa is required for audio processing. "
                "Please install it using: pip install librosa"
            )
            
        # Load Whisper model
        logger.info(f"Loading Whisper model: {model_size}...")
        self.model = whisper.load_model(model_size)
        
        # Initialize speaker diarization if auth token is provided
        self.diarization_pipeline = None
        self.has_diarization = False
        
        if PYANNOTE_AVAILABLE:
            if auth_token:
                try:
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.0",
                        use_auth_token=auth_token
                    )
                    self.has_diarization = True
                    logger.info("Speaker diarization model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load speaker diarization model: {e}")
                    if "401 Client Error" in str(e):
                        logger.error("Authentication error: Please check your HuggingFace token.")
                        logger.error("Speaker diarization requires a valid token with access to pyannote/speaker-diarization-3.0")
                    self.has_diarization = False
            else:
                logger.info("No HuggingFace token provided. Speaker diarization will use basic heuristics instead.")
                self.has_diarization = False
    
    def transcribe(self, audio_file: str, timestamp_interval: int = 5) -> Dict:
        """
        Transcribe audio file with timestamps and speaker diarization
        
        Args:
            audio_file: Path to audio file
            timestamp_interval: Interval in seconds for timestamps (default: 5)
            
        Returns:
            Dictionary containing transcription with timestamps and speakers
        """
        logger.info(f"Transcribing {audio_file}...")
        
        # Check if file exists
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Get audio duration for timestamp calculations
        audio_duration = librosa.get_duration(path=audio_file)
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Transcribe with Whisper
        result = self.model.transcribe(
            audio_file,
            word_timestamps=True,
            verbose=False
        )
        
        # Process speaker diarization
        speaker_map = {}
        detected_speakers = []
        
        # Use pyannote for speaker diarization if available
        if self.diarization_pipeline and self.has_diarization:
            logger.info("Using pyannote.audio for advanced speaker diarization...")
            speaker_map = self._perform_diarization(audio_file, result)
            # Extract unique speakers
            detected_speakers = list(set(speaker for _, speaker in speaker_map.values()))
            # Convert speaker IDs to friendly names
            detected_speakers = self._convert_to_speaker_names(detected_speakers)
        else:
            # Use basic heuristic speaker detection as fallback
            logger.info("Using basic heuristics for speaker detection...")
            result, speaker_map, detected_speakers = self._perform_heuristic_diarization(result)
        
        # Add timestamps at specified intervals
        timestamps = self._add_timestamps(result, audio_duration, timestamp_interval, speaker_map)
        
        # Format all segments with proper speaker names
        timestamps = self._format_speaker_names(timestamps, detected_speakers)
        
        return {
            "transcript": result["text"],
            "segments": timestamps,
            "speakers": detected_speakers if detected_speakers else ["Speaker 1"]
        }
    
    def _perform_diarization(self, audio_file: str, whisper_result: Dict) -> Dict:
        """
        Perform speaker diarization and map to transcription segments using pyannote
        
        Args:
            audio_file: Path to audio file
            whisper_result: Whisper transcription result
            
        Returns:
            Dictionary mapping timestamps to speaker IDs
        """
        logger.info("Performing advanced speaker diarization...")
        
        # Run diarization
        diarization = self.diarization_pipeline(audio_file)
        
        # Map speakers to timestamps
        speaker_map = {}
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            start_time = segment.start
            end_time = segment.end
            
            # Map each word to a speaker
            for segment_info in whisper_result["segments"]:
                # If the segment overlaps with the speaker segment
                segment_start = segment_info["start"]
                segment_end = segment_info["end"]
                
                # Check for overlap
                if not (segment_end <= start_time or segment_start >= end_time):
                    # This segment overlaps with the speaker segment
                    # Map words to speakers
                    for word in segment_info.get("words", []):
                        word_start = word["start"]
                        word_end = word["end"]
                        
                        # If word falls within this speaker segment
                        if word_start >= start_time and word_end <= end_time:
                            speaker_map[(word_start, word_end)] = (word["word"], speaker)
        
        return speaker_map
    
    def _perform_heuristic_diarization(self, result: Dict) -> Tuple[Dict, Dict, List[str]]:
        """
        Use heuristics to estimate speaker changes when pyannote is not available
        
        Args:
            result: Whisper transcription result
            
        Returns:
            Tuple of (updated result, speaker_map, detected_speakers)
        """
        speaker_map = {}
        current_speaker = "Speaker 1"
        speaker_changes = []
        
        # Look for speaker indicators in text like "Speaker 1:" or "John:"
        speaker_pattern = r'(?:^|\s)([A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)?):(?:\s|$)'
        
        # First pass: Detect potential speaker changes
        for i, segment in enumerate(result["segments"]):
            text = segment["text"]
            
            # Look for explicit speaker indicators
            matches = re.findall(speaker_pattern, text)
            if matches:
                for match in matches:
                    # Detected a likely speaker change
                    speaker_name = match.strip()
                    if speaker_name and speaker_name != current_speaker:
                        current_speaker = speaker_name
                        speaker_changes.append((i, current_speaker))
                
                # Clean up the text by removing speaker prefixes
                clean_text = re.sub(speaker_pattern, ' ', text).strip()
                result["segments"][i]["text"] = clean_text
            
            # Heuristic: Long pauses might indicate speaker changes
            if i > 0:
                prev_end = result["segments"][i-1]["end"]
                curr_start = segment["start"]
                if curr_start - prev_end > 1.0:  # Pause of more than 1 second
                    # Possible speaker change
                    if current_speaker == "Speaker 1":
                        current_speaker = "Speaker 2"
                    else:
                        speaker_num = int(current_speaker.split()[-1])
                        if speaker_num >= 3:
                            # Cycle between speakers we've already detected
                            current_speaker = f"Speaker {(speaker_num % 2) + 1}"
                        else:
                            current_speaker = f"Speaker {speaker_num + 1}"
                    speaker_changes.append((i, current_speaker))
        
        # Second pass: Apply speaker labels
        if not speaker_changes:
            # If no speaker changes detected, default to a single speaker
            speaker_changes = [(0, "Speaker 1")]
            
            # If the recording is over 30 seconds, assume it's likely a conversation
            total_duration = sum(segment.get("end", 0) - segment.get("start", 0) 
                               for segment in result["segments"])
            
            if total_duration > 30:
                mid_point = len(result["segments"]) // 2
                speaker_changes.append((mid_point, "Speaker 2"))
        
        current_speaker = "Speaker 1"
        detected_speakers = set(["Speaker 1"])
        
        for i, segment in enumerate(result["segments"]):
            # Check if there's a speaker change at this segment
            for change_idx, new_speaker in speaker_changes:
                if i == change_idx:
                    current_speaker = new_speaker
                    detected_speakers.add(new_speaker)
            
            # Apply current speaker to all words in this segment
            for word in segment.get("words", []):
                word_start = word["start"]
                word_end = word["end"]
                speaker_map[(word_start, word_end)] = (word["word"], current_speaker)
        
        return result, speaker_map, list(detected_speakers)
    
    def _add_timestamps(self, result: Dict, duration: float, interval: int, speaker_map: Dict) -> List[Dict]:
        """
        Add timestamps at specified intervals to transcription
        
        Args:
            result: Whisper transcription result
            duration: Audio duration in seconds
            interval: Interval in seconds for timestamps
            speaker_map: Mapping of timestamps to speakers
            
        Returns:
            List of segments with timestamps and speakers
        """
        timestamps = []
        current_segment = {"timestamp": "00:00:00", "text": "", "speaker": "Unknown"}
        
        # Process each segment from Whisper
        for segment in result["segments"]:
            segment_start = segment["start"]
            
            # Get timestamp in format HH:MM:SS
            timestamp = self._format_timestamp(segment_start)
            
            # Get speaker if available
            speaker = "Unknown"
            for word in segment.get("words", []):
                word_time = (word["start"], word["end"])
                if word_time in speaker_map:
                    word_speaker = speaker_map[word_time][1]
                    speaker = word_speaker
                    break
            
            # If we've reached a new timestamp interval, create a new segment
            interval_time = int(segment_start / interval) * interval
            segment_timestamp = self._format_timestamp(interval_time)
            
            # New logic: Also create a new segment when the speaker changes
            if segment_timestamp != current_segment["timestamp"] or (speaker != "Unknown" and speaker != current_segment["speaker"]):
                if current_segment["text"]:
                    timestamps.append(current_segment)
                
                current_segment = {
                    "timestamp": segment_timestamp,
                    "text": segment["text"].strip(),
                    "speaker": speaker
                }
            else:
                current_segment["text"] += " " + segment["text"].strip()
            
        # Add the last segment
        if current_segment["text"]:
            timestamps.append(current_segment)
        
        return timestamps
    
    def _convert_to_speaker_names(self, technical_ids: List[str]) -> List[str]:
        """
        Convert technical speaker IDs to friendly speaker names
        
        Args:
            technical_ids: List of technical speaker IDs (e.g. SPEAKER_00, SPEAKER_01)
            
        Returns:
            List of speaker names (e.g. Speaker 1, Speaker 2)
        """
        speaker_names = []
        
        for i, speaker_id in enumerate(technical_ids):
            # Convert technical ID to user-friendly name
            if isinstance(speaker_id, str) and speaker_id.startswith("SPEAKER_"):
                # Extract number from technical ID
                try:
                    num = int(speaker_id.split("_")[1]) + 1  # Convert to 1-based indexing
                    speaker_names.append(f"Speaker {num}")
                except (IndexError, ValueError):
                    speaker_names.append(f"Speaker {i+1}")
            else:
                # Handle case where ID is already in readable format
                speaker_names.append(f"Speaker {i+1}")
        
        return speaker_names
    
    def _format_speaker_names(self, segments: List[Dict], speaker_names: List[str]) -> List[Dict]:
        """
        Format speaker names in segments to be user-friendly
        
        Args:
            segments: List of transcript segments
            speaker_names: List of speaker names
            
        Returns:
            Updated list of segments with user-friendly speaker names
        """
        # Create mapping from technical IDs to friendly names
        name_map = {}
        
        # Process each segment
        for segment in segments:
            speaker_id = segment["speaker"]
            
            # Skip if already properly formatted
            if speaker_id.startswith("Speaker "):
                continue
                
            # Map technical ID to friendly name
            if speaker_id not in name_map:
                if speaker_id == "Unknown":
                    # If we have at least one identified speaker, use the first one
                    if speaker_names and len(speaker_names) > 0:
                        name_map[speaker_id] = speaker_names[0]
                    else:
                        name_map[speaker_id] = "Speaker 1"
                else:
                    # Try to extract index from technical ID
                    try:
                        idx = int(speaker_id.split("_")[1])
                        if idx < len(speaker_names):
                            name_map[speaker_id] = speaker_names[idx]
                        else:
                            name_map[speaker_id] = f"Speaker {idx+1}"
                    except (IndexError, ValueError):
                        # If we can't extract an index, assign the next available speaker number
                        name_map[speaker_id] = f"Speaker {len(name_map)+1}"
            
            # Update segment with friendly speaker name
            segment["speaker"] = name_map.get(speaker_id, speaker_id)
        
        return segments
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    # Example usage
    auth_token = None  # Replace with your HuggingFace token if available
    
    transcriber = Transcriber(
        model_size="base",
        auth_token=auth_token
    )
    
    # Replace with your audio file
    audio_file = "../recordings/test_recording.wav"
    if os.path.exists(audio_file):
        result = transcriber.transcribe(audio_file, timestamp_interval=5)
        print("\nTranscription Summary:")
        print(f"Total segments: {len(result['segments'])}")
        print(f"Speakers detected: {', '.join(result['speakers'])}")
        
        print("\nSample transcript with timestamps:")
        for i, segment in enumerate(result['segments'][:5]):
            print(f"{segment['timestamp']} | {segment['speaker']}: {segment['text']}")
    else:
        print(f"Test file {audio_file} not found. Please create a recording first.")