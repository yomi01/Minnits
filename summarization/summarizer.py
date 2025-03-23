import os
import json
import requests
import logging
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationSummarizer:
    def __init__(self, ollama_host: str = "http://localhost:11434", ollama_model: str = "gemma3:12b"):
        """
        Initialize the conversation summarizer using Ollama
        
        Args:
            ollama_host: URL for Ollama API (default: http://localhost:11434)
            ollama_model: Ollama model name (default: gemma3:12b)
        """
        self.ollama_host = ollama_host.rstrip('/')  # Remove trailing slash if present
        self.ollama_model = ollama_model
        
        # Test Ollama connection
        try:
            self._test_ollama_connection()
            logger.info(f"Successfully connected to Ollama using model {ollama_model}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.error("Make sure Ollama is running and the model is downloaded.")
            raise

    def _test_ollama_connection(self):
        """Test the connection to Ollama"""
        try:
            # First check if Ollama is running
            try:
                url = f"{self.ollama_host}/api/version"
                response = requests.get(url, timeout=5)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"Could not connect to Ollama at {self.ollama_host}. Make sure Ollama is running.")

            # Then check available models
            url = f"{self.ollama_host}/api/tags"
            response = requests.get(url)
            response.raise_for_status()
            
            # Check if model is available, if not try to pull it
            if not self._is_model_available():
                logger.info(f"Model {self.ollama_model} not found. Attempting to pull...")
                self._pull_model()
                
        except Exception as e:
            raise Exception(f"Failed to initialize Ollama connection: {str(e)}")

    def _is_model_available(self) -> bool:
        """Check if the specified model is available in Ollama"""
        try:
            url = f"{self.ollama_host}/api/tags"
            response = requests.get(url)
            response.raise_for_status()
            models = response.json()
            
            # Check both the exact model name and without version number
            base_model_name = self.ollama_model.split(':')[0]
            for model in models.get("models", []):
                model_name = model.get("name", "")
                if model_name == self.ollama_model or model_name.startswith(f"{base_model_name}:"):
                    self.ollama_model = model_name  # Use the exact model name that's available
                    return True
            return False
            
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    def _pull_model(self):
        """Pull the specified model from Ollama"""
        try:
            url = f"{self.ollama_host}/api/pull"
            payload = {"name": self.ollama_model}
            response = requests.post(url, json=payload)
            response.raise_for_status()
            logger.info(f"Successfully pulled model {self.ollama_model}")
            
        except Exception as e:
            raise Exception(f"Failed to pull model {self.ollama_model}: {str(e)}")
    
    def summarize(self, transcript: Dict[str, Any], max_length: int = 500) -> Dict[str, Any]:
        """
        Generate a summary of the conversation transcript including action points
        
        Args:
            transcript: Dictionary containing the transcription data
            max_length: Maximum length of the summary in words
            
        Returns:
            Dictionary containing the summary and action points
        """
        if not transcript or not transcript.get("segments"):
            raise ValueError("Invalid transcript data provided")
        
        # Format the transcript for the AI
        formatted_transcript = self._format_transcript_for_ai(transcript)
        
        # Create the prompt for the AI
        prompt = self._create_summary_prompt(formatted_transcript, max_length)
        
        try:
            # Request summary from Ollama
            api_url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "system": "You are a professional conversation summarizer. You analyze conversations and extract key points and action items. Be concise and clear.",
                "options": {
                    "temperature": 0.3,  # Lower temperature for more focused responses
                    "top_k": 40,
                    "top_p": 0.9,
                    "num_predict": 1024  # Limit response length
                }
            }
            
            logger.info("Sending request to Ollama for summary generation...")
            response = requests.post(api_url, json=payload, timeout=120)  # Increased timeout
            response.raise_for_status()
            
            result = response.json()
            summary_text = result.get("response", "").strip()
            
            # Extract summary and action points from the response
            result = self._parse_summary_response(summary_text)
            
            # If the response doesn't contain the expected format, try to fix it
            if not result["summary"] or not result["action_points"]:
                logger.warning("Response not in expected format, attempting to reformat...")
                # Try asking Ollama to reformat the response
                reformatted = self._reformat_with_ollama(summary_text)
                if reformatted["summary"]:
                    result = reformatted
                else:
                    # Fallback to local reformatting
                    result = self._reformat_unstructured_response(summary_text, max_length)
            
            return {
                "original_transcript": transcript["transcript"],
                "summary": result["summary"],
                "action_points": result["action_points"],
                "participants": list(set(segment["speaker"] for segment in transcript["segments"])),
                "duration": self._calculate_duration(transcript["segments"])
            }
            
        except requests.exceptions.Timeout:
            logger.error("Timeout while generating summary")
            return {
                "summary": "Error: Request timed out while generating summary",
                "action_points": [],
                "error": "Request timeout"
            }
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                "summary": "Error generating summary",
                "action_points": [],
                "error": str(e)
            }
    
    def _reformat_with_ollama(self, unstructured_text: str) -> Dict[str, Any]:
        """
        Use Ollama to reformat an unstructured response
        
        Args:
            unstructured_text: The text to reformat
            
        Returns:
            Dictionary with reformatted summary and action points
        """
        prompt = f"""Please reformat this text into a clear summary and action points using exactly this format:
SUMMARY: [A clear, concise summary]
ACTION POINTS:
- [First action item]
- [Second action item]
- [etc.]

Text to reformat:
{unstructured_text}"""
        
        try:
            api_url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "top_k": 40,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            reformatted_text = result.get("response", "").strip()
            
            return self._parse_summary_response(reformatted_text)
            
        except Exception as e:
            logger.error(f"Error reformatting with Ollama: {e}")
            return {"summary": "", "action_points": []}
    
    def _format_transcript_for_ai(self, transcript: Dict[str, Any]) -> str:
        """Format the transcript for the AI prompt"""
        formatted_text = ""
        current_speaker = None
        
        for segment in transcript["segments"]:
            # Only add speaker name when it changes
            if segment["speaker"] != current_speaker:
                current_speaker = segment["speaker"]
                formatted_text += f"\n{current_speaker}:"
            
            formatted_text += f" [{segment['timestamp']}] {segment['text']}"
        
        return formatted_text.strip()
    
    def _create_summary_prompt(self, formatted_transcript: str, max_length: int) -> str:
        """Create the prompt for the AI"""
        return f"""Please analyze this conversation transcript and provide:

1. A clear, concise summary of the main discussion points (maximum {max_length} words)
2. A list of specific action items, tasks, or follow-ups mentioned

Use exactly this format:
SUMMARY: [write the summary here]
ACTION POINTS:
- [First action item]
- [Second action item]
- [etc.]

Transcript:
{formatted_transcript}

Important: Maintain the exact format with SUMMARY: and ACTION POINTS: headings. Be specific and actionable."""
    
    def _parse_summary_response(self, response: str) -> Dict[str, Any]:
        """Parse the AI response to extract summary and action points"""
        result = {
            "summary": "",
            "action_points": []
        }
        
        try:
            # Extract summary
            if "SUMMARY:" in response:
                summary_text = response.split("SUMMARY:")[1]
                if "ACTION POINTS:" in summary_text:
                    summary_text = summary_text.split("ACTION POINTS:")[0]
                result["summary"] = summary_text.strip()
            
            # Extract action points
            if "ACTION POINTS:" in response:
                action_points_text = response.split("ACTION POINTS:")[1].strip()
                action_points = []
                
                for line in action_points_text.split("\n"):
                    line = line.strip()
                    if line.startswith("- "):
                        action_points.append(line[2:])  # Remove the "- " prefix
                    elif line and line[0].isdigit() and ". " in line:
                        action_points.append(line.split(". ", 1)[1])  # For numbered lists
                    elif line and not any(line.startswith(x) for x in ["SUMMARY:", "ACTION POINTS:"]):
                        # Include lines that look like action items but aren't properly formatted
                        action_points.append(line)
                
                result["action_points"] = [point.strip() for point in action_points if point.strip()]
        except Exception as e:
            logger.error(f"Error parsing summary response: {e}")
        
        return result
    
    def _reformat_unstructured_response(self, text: str, max_length: int) -> Dict[str, Any]:
        """
        Attempt to reformat an unstructured response into summary and action points
        
        Args:
            text: The unstructured response text
            max_length: Maximum length for the summary
            
        Returns:
            Dictionary with reformatted summary and action points
        """
        # Initialize result
        result = {
            "summary": "",
            "action_points": []
        }
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        if not paragraphs:
            return result
        
        # First substantial paragraph is likely the summary
        summary_candidates = [p for p in paragraphs if len(p.split()) >= 10]
        if summary_candidates:
            # Truncate to max length if needed
            words = summary_candidates[0].split()
            if len(words) > max_length:
                words = words[:max_length]
                result["summary"] = " ".join(words) + "..."
            else:
                result["summary"] = summary_candidates[0]
        
        # Look for potential action items in the remaining text
        action_markers = [
            "need to", "should", "must", "will", "agreed to", "plan to", "going to",
            "task:", "action:", "follow up:", "to do:", "assigned to"
        ]
        
        for paragraph in paragraphs[1:]:
            sentences = [s.strip() for s in paragraph.split(".") if s.strip()]
            for sentence in sentences:
                # Check if sentence contains action-oriented language
                if any(marker.lower() in sentence.lower() for marker in action_markers):
                    # Clean up and format as action item
                    action = sentence.strip()
                    if not action.endswith("."):
                        action += "."
                    result["action_points"].append(action)
        
        return result
    
    def _calculate_duration(self, segments: List[Dict[str, Any]]) -> str:
        """Calculate the duration of the conversation from the timestamps"""
        if not segments:
            return "00:00:00"
        
        # Get the last timestamp (assuming segments are ordered)
        last_timestamp = segments[-1]["timestamp"]
        return last_timestamp


if __name__ == "__main__":
    # Example usage
    summarizer = ConversationSummarizer()
    
    # Sample transcript data (normally this would come from the transcriber)
    sample_transcript = {
        "transcript": "This is a sample conversation about project planning.",
        "segments": [
            {"timestamp": "00:00:00", "text": "Hi everyone, let's discuss the project timeline.", "speaker": "Speaker 1"},
            {"timestamp": "00:00:05", "text": "I think we need to finish the first draft by next Friday.", "speaker": "Speaker 1"},
            {"timestamp": "00:00:10", "text": "That sounds good. I'll work on the design part.", "speaker": "Speaker 2"},
            {"timestamp": "00:00:15", "text": "Great! Please also send me the budget proposal.", "speaker": "Speaker 1"}
        ]
    }
    
    try:
        summary_result = summarizer.summarize(sample_transcript)
        print("\nSummary Result:")
        print(f"Summary: {summary_result['summary']}")
        print("\nAction Points:")
        for action in summary_result['action_points']:
            print(f"- {action}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Make sure Ollama is running and the gemma:12b model is installed.")