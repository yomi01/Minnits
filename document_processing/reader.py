"""
Document Reader Module

This module provides functionality to read and extract text from various document formats.
Currently supports:
- DOCX files (Microsoft Word documents)
- TXT files (plain text)
- JSON files (transcript format)
"""

import os
import json
import io
from typing import Dict, Any, Optional, Union
from datetime import datetime

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class DocumentReader:
    """
    A class to read and process various document formats for the note summarizer.
    """
    
    def __init__(self):
        """Initialize the document reader."""
        self.supported_formats = ['txt', 'json']
        if DOCX_AVAILABLE:
            self.supported_formats.append('docx')
    
    def get_supported_formats(self) -> list:
        """Return list of supported file formats."""
        return self.supported_formats.copy()
    
    def read_document(self, file_path: str = None, file_content: bytes = None, 
                     file_name: str = None) -> Dict[str, Any]:
        """
        Read a document and return structured text data.
        
        Args:
            file_path: Path to the file (for local files)
            file_content: Raw file content as bytes (for uploaded files)
            file_name: Name of the file to determine format
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        if file_path:
            file_extension = os.path.splitext(file_path)[1].lower().lstrip('.')
            with open(file_path, 'rb') as f:
                content = f.read()
        elif file_content and file_name:
            file_extension = os.path.splitext(file_name)[1].lower().lstrip('.')
            content = file_content
        else:
            raise ValueError("Either file_path or (file_content + file_name) must be provided")
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: {self.supported_formats}")
        
        # Process based on file type
        if file_extension == 'docx':
            return self._read_docx(content, file_name or file_path)
        elif file_extension == 'txt':
            return self._read_text(content, file_name or file_path)
        elif file_extension == 'json':
            return self._read_json(content, file_name or file_path)
        else:
            raise ValueError(f"Handler not implemented for format: {file_extension}")
    
    def _read_docx(self, content: bytes, source_name: str) -> Dict[str, Any]:
        """
        Read a DOCX file and extract text content.
        
        Args:
            content: Raw file content
            source_name: Name or path of the source file
            
        Returns:
            Structured document data
        """
        if not DOCX_AVAILABLE:
            raise ImportError("python-docx is required to read DOCX files. Install it with: pip install python-docx")
        
        try:
            # Create a document object from bytes
            document = Document(io.BytesIO(content))
            
            # Extract text from all paragraphs
            paragraphs = []
            full_text = []
            
            for i, paragraph in enumerate(document.paragraphs):
                para_text = paragraph.text.strip()
                if para_text:  # Only include non-empty paragraphs
                    paragraphs.append({
                        "paragraph_number": i + 1,
                        "text": para_text,
                        "timestamp": f"Para {i + 1:03d}"
                    })
                    full_text.append(para_text)
            
            # Join all text
            combined_text = "\n\n".join(full_text)
            
            # Create transcript-like structure
            transcript_data = {
                "source": source_name,
                "source_type": "document",
                "format": "docx",
                "processed_at": datetime.now().isoformat(),
                "transcript": combined_text,
                "paragraphs": paragraphs,
                "segments": [{
                    "timestamp": "00:00:00",
                    "text": combined_text,
                    "speaker": "Document",
                    "source": "DOCX Document"
                }],
                "metadata": {
                    "total_paragraphs": len(paragraphs),
                    "word_count": len(combined_text.split()),
                    "character_count": len(combined_text)
                }
            }
            
            return transcript_data
            
        except Exception as e:
            raise ValueError(f"Error reading DOCX file: {str(e)}")
    
    def _read_text(self, content: bytes, source_name: str) -> Dict[str, Any]:
        """
        Read a plain text file.
        
        Args:
            content: Raw file content
            source_name: Name or path of the source file
            
        Returns:
            Structured document data
        """
        try:
            # Decode text content
            text_content = content.decode('utf-8')
            
            # Split into paragraphs (by double newlines or single newlines)
            paragraphs = []
            if '\n\n' in text_content:
                para_list = [p.strip() for p in text_content.split('\n\n') if p.strip()]
            else:
                para_list = [p.strip() for p in text_content.split('\n') if p.strip()]
            
            for i, para_text in enumerate(para_list):
                if para_text:
                    paragraphs.append({
                        "paragraph_number": i + 1,
                        "text": para_text,
                        "timestamp": f"Para {i + 1:03d}"
                    })
            
            # Create transcript-like structure
            transcript_data = {
                "source": source_name,
                "source_type": "document", 
                "format": "txt",
                "processed_at": datetime.now().isoformat(),
                "transcript": text_content,
                "paragraphs": paragraphs,
                "segments": [{
                    "timestamp": "00:00:00",
                    "text": text_content,
                    "speaker": "Document",
                    "source": "Text Document"
                }],
                "metadata": {
                    "total_paragraphs": len(paragraphs),
                    "word_count": len(text_content.split()),
                    "character_count": len(text_content)
                }
            }
            
            return transcript_data
            
        except UnicodeDecodeError:
            raise ValueError("Error decoding text file. Please ensure it's in UTF-8 format.")
        except Exception as e:
            raise ValueError(f"Error reading text file: {str(e)}")
    
    def _read_json(self, content: bytes, source_name: str) -> Dict[str, Any]:
        """
        Read a JSON file (existing transcript format).
        
        Args:
            content: Raw file content
            source_name: Name or path of the source file
            
        Returns:
            Structured document data
        """
        try:
            text_content = content.decode('utf-8')
            json_data = json.loads(text_content)
            
            # If it's already in transcript format, return as-is but add metadata
            if isinstance(json_data, dict) and "segments" in json_data:
                json_data["source"] = source_name
                json_data["format"] = "json"
                json_data["processed_at"] = datetime.now().isoformat()
                return json_data
            else:
                # Convert generic JSON to transcript format
                json_str = json.dumps(json_data, indent=2)
                transcript_data = {
                    "source": source_name,
                    "source_type": "document",
                    "format": "json",
                    "processed_at": datetime.now().isoformat(),
                    "transcript": json_str,
                    "segments": [{
                        "timestamp": "00:00:00",
                        "text": json_str,
                        "speaker": "Document",
                        "source": "JSON Document"
                    }],
                    "metadata": {
                        "word_count": len(json_str.split()),
                        "character_count": len(json_str)
                    }
                }
                return transcript_data
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error reading JSON file: {str(e)}")


def create_document_reader() -> DocumentReader:
    """Factory function to create a DocumentReader instance."""
    return DocumentReader()
