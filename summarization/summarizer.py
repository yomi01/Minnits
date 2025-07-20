import os
import json
import requests
import logging
import signal
import atexit
from typing import Dict, List, Any, Optional
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.schema import LLMResult

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConversationSummarizer:
    # Class variable to keep track of all instances for cleanup
    _instances = []
    
    def __init__(self, ollama_host: str = "http://localhost:11434", ollama_model: str = "gemma3:12b", register_signals: bool = False):
        """
        Initialize the conversation summarizer using Ollama with LangChain
        
        Args:
            ollama_host: URL for Ollama API (default: http://localhost:11434)
            ollama_model: Ollama model name (default: gemma3:12b)
            register_signals: Whether to register signal handlers for cleanup (default: False)
        """
        self.ollama_host = ollama_host.rstrip('/')  # Remove trailing slash if present
        self.ollama_model = ollama_model
        self.llm = None
        self._is_cleaned_up = False
        
        # Register this instance for cleanup
        ConversationSummarizer._instances.append(self)
        
        # Optionally register signal handlers
        if register_signals:
            register_signal_handlers()
        
        # Initialize LangChain Ollama LLM
        try:
            self.llm = Ollama(
                base_url=ollama_host,
                model=ollama_model,
                temperature=0.3,
                top_k=40,
                top_p=0.9,
                num_predict=4096,  # Increased for longer documents
            )
            logger.info(f"Successfully initialized LangChain Ollama LLM with model {ollama_model}")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain Ollama LLM: {e}")
            raise
        
        # Initialize detailed meeting minutes prompt template
        self.meeting_prompt = PromptTemplate(
            input_variables=["transcript", "focus_areas"],
            template="""Role: You are a professional content analyst capable of summarizing meetings, documents, and other text content.

Task: Analyze the following content and produce a comprehensive, structured summary. For meeting transcripts, format as meeting minutes. For documents, format as a professional summary. Include:

**Meeting Overview** (or **Content Overview** for documents) – For meetings: Date, time, purpose, and key participants. For documents: Purpose, context, source, and key themes.

**Discussion Highlights** (or **Key Points** for documents) – For meetings: Summary of main topics discussed with critical points. For documents: Main arguments, findings, important information, and key insights. Include specifics like figures, names, or data.

**Decisions Made** (or **Conclusions** for documents) – For meetings: Bullet points for decisions and agreements. For documents: Main conclusions, determinations, or key takeaways from the content.

**Action Items** (or **Actionable Items** for documents) – For meetings: Tasks with responsible parties and deadlines. For documents: Recommendations, suggested actions, implementation steps, or follow-up items mentioned.

**Next Steps/Follow-Up** (or **Implications** for documents) – For meetings: Upcoming meetings and deadlines. For documents: Implications, future considerations, or next steps suggested by the content.

**Important Guidelines:**
- Provide substantive content in each section. Don't just say "not specified" unless truly empty
- Extract maximum value from the provided content
- Use clear section headings and bullet points for readability
- Write in a formal, professional tone
- Include specific details like names, figures, dates, and data when mentioned
- For documents, focus on extracting actionable insights and key information
- Ensure the summary is comprehensive and captures the essence of the content
- If content spans multiple topics, organize highlights by theme or section

{focus_areas}

**Content:**
{transcript}

**Structured Summary:**"""
        )
        
        # Initialize a simplified prompt template for backward compatibility
        self.simple_prompt = PromptTemplate(
            input_variables=["transcript", "max_length", "focus_areas"],
            template="""Please analyze this conversation transcript and provide:

1. A clear, concise summary of the main discussion points (maximum {max_length} words)
2. A list of specific action items, tasks, or follow-ups mentioned

{focus_areas}

Use exactly this format:
SUMMARY: [write the summary here]
ACTION POINTS:
- [First action item]
- [Second action item]
- [etc.]

Transcript:
{transcript}

Important: Maintain the exact format with SUMMARY: and ACTION POINTS: headings. Be specific and actionable."""
        )
        
        # Test Ollama connection
        try:
            self._test_ollama_connection()
            logger.info(f"Successfully connected to Ollama using model {ollama_model}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.error("Make sure Ollama is running and the model is downloaded.")
            raise

    def cleanup(self):
        """
        Clean up resources and unload the model from Ollama
        """
        if self._is_cleaned_up:
            return
            
        try:
            logger.info(f"Cleaning up ConversationSummarizer - unloading model {self.ollama_model}")
            
            # Unload the model from Ollama to free GPU memory
            self._unload_model()
            
            # Clear the LLM instance
            self.llm = None
            
            # Remove from instances list
            if self in ConversationSummarizer._instances:
                ConversationSummarizer._instances.remove(self)
            
            self._is_cleaned_up = True
            logger.info("ConversationSummarizer cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _unload_model(self):
        """
        Unload the model from Ollama to free memory
        """
        try:
            # Method 1: Make a request with keep_alive=0 to immediately unload
            url = f"{self.ollama_host}/api/generate"
            payload = {
                "model": self.ollama_model,
                "prompt": "unload",
                "stream": False,
                "keep_alive": 0  # Immediately unload after this request
            }
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Model {self.ollama_model} unloaded successfully (keep_alive=0)")
            else:
                logger.warning(f"Model unload returned status {response.status_code}: {response.text}")
                
        except Exception as e:
            logger.warning(f"Could not unload model {self.ollama_model}: {e}")
            
            # Fallback method: Try to load a tiny model to free GPU memory
            try:
                logger.info("Attempting fallback: loading tiny model to free GPU memory")
                fallback_url = f"{self.ollama_host}/api/generate"
                fallback_payload = {
                    "model": "tinyllama:1.1b",  # Use a small model if available
                    "prompt": "hello",
                    "stream": False,
                    "keep_alive": 0
                }
                fallback_response = requests.post(fallback_url, json=fallback_payload, timeout=10)
                if fallback_response.status_code == 200:
                    logger.info("Fallback: loaded tiny model to free memory")
            except Exception as fallback_error:
                logger.warning(f"Fallback method also failed: {fallback_error}")

    @classmethod
    def cleanup_all_instances(cls):
        """
        Clean up all ConversationSummarizer instances
        """
        logger.info("Cleaning up all ConversationSummarizer instances...")
        instances_copy = cls._instances.copy()  # Create a copy to avoid modification during iteration
        for instance in instances_copy:
            try:
                instance.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up instance: {e}")
        logger.info("All ConversationSummarizer instances cleaned up")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()

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
    
    def summarize(self, transcript: Dict[str, Any], max_length: int = 500, focus_areas: Optional[List[str]] = None, use_detailed_format: bool = True) -> Dict[str, Any]:
        """
        Generate a summary of the conversation transcript including action points
        
        Args:
            transcript: Dictionary containing the transcription data
            max_length: Maximum length of the summary in words (used only for simple format)
            focus_areas: Optional list of specific areas to focus on in the summary
            use_detailed_format: Whether to use the detailed meeting minutes format (default: True)
            
        Returns:
            Dictionary containing the summary and action points
        """
        if not transcript or not transcript.get("segments"):
            raise ValueError("Invalid transcript data provided")
        
        # Format the transcript for the AI
        formatted_transcript = self._format_transcript_for_ai(transcript)
        
        # Prepare focus areas text
        focus_text = ""
        if focus_areas and len(focus_areas) > 0:
            focus_list = ", ".join(focus_areas)
            focus_text = f"\n**SPECIAL FOCUS:** Pay particular attention to and emphasize these areas in your summary: {focus_list}\n"
        
        try:
            if use_detailed_format:
                # Use the detailed meeting minutes prompt
                prompt = self.meeting_prompt.format(
                    transcript=formatted_transcript,
                    focus_areas=focus_text
                )
                
                logger.info("Generating detailed meeting minutes using LangChain...")
                # Use LangChain LLM for generation
                response = self.llm.invoke(prompt)
                summary_text = response.strip()
                
                # Parse the detailed response
                result = self._parse_detailed_meeting_response(summary_text)
                
            else:
                # Use the simple prompt for backward compatibility
                prompt = self.simple_prompt.format(
                    transcript=formatted_transcript,
                    max_length=max_length,
                    focus_areas=focus_text if focus_text else ""
                )
                
                logger.info("Generating simple summary using LangChain...")
                # Use LangChain LLM for generation
                response = self.llm.invoke(prompt)
                summary_text = response.strip()
                
                # Parse the simple response
                result = self._parse_summary_response(summary_text)
            
            # Check if we got a meaningful response and log the raw response for debugging
            logger.info(f"Raw model response length: {len(summary_text)} characters")
            logger.debug(f"Raw model response: {summary_text[:500]}...")
            
            # If the response doesn't contain the expected format, try to fix it
            if (not result.get("summary") and not result.get("meeting_overview")) or \
               (result.get("meeting_overview") and "not specified" in result.get("meeting_overview", "").lower() and 
                len(result.get("discussion_highlights", "")) < 100):
                logger.warning("Response not in expected format, attempting to reformat...")
                # Fallback to local reformatting
                if use_detailed_format:
                    result = self._reformat_to_detailed_response(summary_text)
                else:
                    result = self._reformat_unstructured_response(summary_text, max_length)
            
            return {
                "original_transcript": transcript["transcript"],
                "summary": result.get("summary", ""),
                "action_points": result.get("action_points", []),
                "participants": list(set(segment["speaker"] for segment in transcript["segments"])),
                "duration": self._calculate_duration(transcript["segments"]),
                "meeting_overview": result.get("meeting_overview", ""),
                "discussion_highlights": result.get("discussion_highlights", ""),
                "decisions_made": result.get("decisions_made", []),
                "next_steps": result.get("next_steps", ""),
                "format_used": "detailed" if use_detailed_format else "simple"
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback to a basic summary
            basic_summary = self._create_basic_summary(formatted_transcript, max_length)
            return {
                "original_transcript": transcript["transcript"],
                "summary": basic_summary,
                "action_points": [],
                "participants": list(set(segment["speaker"] for segment in transcript["segments"])),
                "duration": self._calculate_duration(transcript["segments"]),
                "meeting_overview": "Summary generation failed, basic transcript provided",
                "discussion_highlights": "",
                "decisions_made": [],
                "next_steps": "",
                "format_used": "fallback",
                "error": str(e)
            }

    def _parse_detailed_meeting_response(self, response: str) -> Dict[str, Any]:
        """Parse the detailed meeting minutes response"""
        result = {
            "meeting_overview": "",
            "discussion_highlights": "",
            "decisions_made": [],
            "action_points": [],
            "next_steps": "",
            "summary": ""  # For backward compatibility
        }
        
        try:
            # Split response into sections - be more flexible with section detection
            sections = {
                "meeting overview": "meeting_overview",
                "discussion highlights": "discussion_highlights", 
                "decisions made": "decisions_made",
                "action items": "action_points",
                "action points": "action_points",  # Alternative name
                "next steps": "next_steps",
                "follow-up": "next_steps",
                "follow up": "next_steps"
            }
            
            current_section = None
            current_content = []
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line is a section header (be more flexible)
                section_found = False
                line_lower = line.lower()
                for section_name, result_key in sections.items():
                    # Look for section headers with various markers
                    if (section_name in line_lower and 
                        (any(char in line for char in ['*', '#', ':', '🏢', '💬', '✅', '📋', '🔄']) or
                         line.isupper() or
                         line.startswith('**') or
                         (line.endswith(':') or line.endswith('**')))):
                        
                        # Save previous section content
                        if current_section and current_content:
                            content = '\n'.join(current_content).strip()
                            if current_section in ["decisions_made", "action_points"]:
                                # Parse bullet points
                                result[current_section] = self._parse_bullet_points(content)
                            else:
                                result[current_section] = content
                        
                        current_section = result_key
                        current_content = []
                        section_found = True
                        break
                
                if not section_found and current_section:
                    current_content.append(line)
                elif not section_found and not current_section:
                    # If we haven't found any sections yet, this might be unstructured content
                    # Add it to discussion highlights as a fallback
                    if not result["discussion_highlights"]:
                        result["discussion_highlights"] = line
                    else:
                        result["discussion_highlights"] += " " + line
            
            # Handle the last section
            if current_section and current_content:
                content = '\n'.join(current_content).strip()
                if current_section in ["decisions_made", "action_points"]:
                    result[current_section] = self._parse_bullet_points(content)
                else:
                    result[current_section] = content
            
            # If we didn't find structured content, try to extract what we can
            if all(not result[key] for key in ["meeting_overview", "discussion_highlights", "decisions_made", "action_points", "next_steps"]):
                # No structured content found, treat entire response as discussion highlights
                result["discussion_highlights"] = response.strip()
                
                # Try to extract action items from the full text
                action_items = self._extract_action_items_from_text(response)
                if action_items:
                    result["action_points"] = action_items
            
            # Create a summary for backward compatibility
            summary_parts = []
            if result["meeting_overview"] and "not specified" not in result["meeting_overview"].lower():
                summary_parts.append(result["meeting_overview"])
            if result["discussion_highlights"]:
                summary_parts.append(result["discussion_highlights"])
            result["summary"] = " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error parsing detailed meeting response: {e}")
            # Fallback: treat entire response as summary
            result["summary"] = response
            result["discussion_highlights"] = response
        
        return result

    def _parse_bullet_points(self, content: str) -> List[str]:
        """Extract bullet points from content"""
        points = []
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('- '):
                points.append(line[2:].strip())
            elif line.startswith('• '):
                points.append(line[2:].strip())
            elif line and line[0].isdigit() and '. ' in line:
                points.append(line.split('. ', 1)[1].strip())
            elif line and not line.startswith('*') and not line.startswith('#'):
                # Include non-header lines that might be action items
                points.append(line)
        return [point for point in points if point]
    
    def _extract_action_items_from_text(self, text: str) -> List[str]:
        """Extract potential action items from unstructured text"""
        action_items = []
        
        # Action keywords that indicate tasks or next steps
        action_keywords = [
            "need to", "should", "must", "will", "agreed to", "plan to", "going to",
            "task:", "action:", "follow up:", "to do:", "assigned to", "responsible for",
            "deadline", "due", "complete", "finish", "implement", "create", "develop",
            "review", "analyze", "investigate", "research", "contact", "schedule"
        ]
        
        sentences = []
        # Split by both periods and newlines
        for chunk in text.replace('\n', '. ').split('.'):
            chunk = chunk.strip()
            if chunk:
                sentences.append(chunk)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in action_keywords):
                # Clean up the sentence
                clean_sentence = sentence.strip()
                if clean_sentence and len(clean_sentence) > 10:  # Avoid very short fragments
                    action_items.append(clean_sentence)
        
        return action_items[:10]  # Limit to top 10 action items

    def _reformat_to_detailed_response(self, text: str) -> Dict[str, Any]:
        """Reformat unstructured text to detailed meeting format"""
        # Extract action items using the more sophisticated method
        action_items = self._extract_action_items_from_text(text)
        
        # Get the main content without action items for highlights
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        main_content = text
        
        # Try to identify if this is a document summary vs meeting transcript
        is_document = any(word in text.lower() for word in [
            "document", "article", "paper", "report", "study", "text", "content",
            "chapter", "section", "author", "written", "published"
        ])
        
        result = {
            "meeting_overview": "Document Summary" if is_document else "Content Overview",
            "discussion_highlights": "",
            "decisions_made": [],
            "action_points": action_items,
            "next_steps": "",
            "summary": text[:1000] + "..." if len(text) > 1000 else text
        }
        
        # Use the full text as discussion highlights if it's reasonable length
        if len(text) <= 2000:
            result["discussion_highlights"] = text
        else:
            # Use first few paragraphs for very long content
            if paragraphs:
                result["discussion_highlights"] = "\n\n".join(paragraphs[:3])
            else:
                # Fallback to first 1500 characters
                result["discussion_highlights"] = text[:1500] + "..."
        
        # Try to extract decisions and next steps
        decision_keywords = ["decided", "agreed", "concluded", "determined", "resolved"]
        next_step_keywords = ["next", "future", "upcoming", "plan", "schedule", "later"]
        
        sentences = [s.strip() for s in text.replace('\n', '. ').split('.') if s.strip()]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in decision_keywords):
                if sentence not in result["action_points"]:  # Avoid duplicates
                    result["decisions_made"].append(sentence.strip())
            elif any(keyword in sentence_lower for keyword in next_step_keywords):
                if sentence not in result["action_points"]:  # Avoid duplicates
                    result["next_steps"] += sentence.strip() + " "
        
        # Clean up next steps
        result["next_steps"] = result["next_steps"].strip()
        
        return result

    def _create_basic_summary(self, transcript: str, max_length: int) -> str:
        """Create a basic summary when AI generation fails"""
        words = transcript.split()
        if len(words) <= max_length:
            return transcript
        else:
            return " ".join(words[:max_length]) + "..."

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


# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals to clean up resources"""
    try:
        logger.info(f"Received signal {signum}, cleaning up...")
        # Use globals() to access the ConversationSummarizer class safely
        if 'ConversationSummarizer' in globals():
            ConversationSummarizer.cleanup_all_instances()
    except Exception as e:
        logger.error(f"Error during signal cleanup: {e}")
    finally:
        import sys
        sys.exit(0)

def register_signal_handlers():
    """Register signal handlers for graceful shutdown (optional)"""
    try:
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, signal_handler)
        logger.debug("Signal handlers registered successfully")
    except Exception as e:
        logger.warning(f"Could not register signal handlers: {e}")

# Safe cleanup function for atexit registration
def _safe_cleanup_all():
    """Safe wrapper for cleanup_all_instances that handles scope issues"""
    try:
        if 'ConversationSummarizer' in globals():
            ConversationSummarizer.cleanup_all_instances()
    except Exception as e:
        # Use print instead of logger since logging might not be available during shutdown
        print(f"Warning: Error during atexit cleanup: {e}")

# Register cleanup function to run at exit
atexit.register(_safe_cleanup_all)


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
    finally:
        # Ensure cleanup happens
        summarizer.cleanup()