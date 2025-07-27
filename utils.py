"""
Shared utilities for PDF processing and text analysis.
"""
import fitz  # PyMuPDF
import re
import json
from typing import List, Dict, Tuple, Any
import numpy as np


class PDFProcessor:
    """Base class for PDF processing functionality."""
    
    def __init__(self):
        self.font_size_threshold = 1.2  # Multiplier for identifying headings
        
    def extract_text_with_formatting(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text with formatting information from PDF.
        
        Returns:
            List of text blocks with formatting metadata
        """
        doc = fitz.open(pdf_path)
        formatted_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if text:
                                formatted_text.append({
                                    "text": text,
                                    "page": page_num + 1,
                                    "font_size": span["size"],
                                    "font_name": span["font"],
                                    "flags": span["flags"],  # Bold, italic, etc.
                                    "bbox": span["bbox"]
                                })
        
        doc.close()
        return formatted_text
    
    def is_bold(self, flags: int) -> bool:
        """Check if text is bold based on font flags."""
        return bool(flags & 2**4)  # Bold flag
    
    def is_italic(self, flags: int) -> bool:
        """Check if text is italic based on font flags."""
        return bool(flags & 2**1)  # Italic flag
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove common artifacts
        text = re.sub(r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\'\"]', '', text)
        return text
    
    def detect_heading_patterns(self, text: str) -> bool:
        """
        Detect common heading patterns using text analysis.
        """
        # Common heading patterns
        patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 Introduction"
            r'^[A-Z][A-Z\s]{2,}$',  # All caps text
            r'^Chapter\s+\d+',  # "Chapter 1"
            r'^Section\s+\d+',  # "Section 1"
            r'^\d+\.\d+\.?\s+',  # "1.1 Subsection"
            r'^[IVX]+\.?\s+[A-Z]',  # Roman numerals
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def calculate_font_statistics(self, formatted_text: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate font size statistics to identify body text baseline."""
        font_sizes = [item["font_size"] for item in formatted_text if len(item["text"]) > 10]
        
        if not font_sizes:
            return {"mean": 12.0, "median": 12.0, "std": 2.0}
        
        return {
            "mean": np.mean(font_sizes),
            "median": np.median(font_sizes),
            "std": np.std(font_sizes)
        }


class HeadingClassifier:
    """Classify text as headings based on multiple features."""
    
    def __init__(self, font_stats: Dict[str, float]):
        self.font_stats = font_stats
        self.base_font_size = font_stats["median"]
        
    def classify_heading_level(self, text_item: Dict[str, Any]) -> str:
        """
        Classify text as H1, H2, H3, or None based on multiple features.
        """
        text = text_item["text"]
        font_size = text_item["font_size"]
        flags = text_item["flags"]
        
        # Skip very short text or common non-heading patterns
        if len(text) < 3 or len(text) > 200:
            return None
            
        if re.match(r'^\d+$|^page\s+\d+|^\w{1,2}$', text.lower()):
            return None
        
        # Calculate relative font size
        size_ratio = font_size / self.base_font_size if self.base_font_size > 0 else 1.0
        
        # Feature scoring
        score = 0
        
        # Font size factor (most important)
        if size_ratio >= 1.8:
            score += 3
        elif size_ratio >= 1.4:
            score += 2
        elif size_ratio >= 1.1:
            score += 1
        
        # Bold text
        if text_item.get("flags", 0) & 16:  # Bold flag
            score += 2
        
        # Pattern matching
        if self._matches_heading_pattern(text):
            score += 2
        
        # Position and formatting
        if self._is_likely_heading_position(text):
            score += 1
        
        # Determine heading level based on score and font size
        if score >= 4:
            if size_ratio >= 1.8:
                return "H1"
            elif size_ratio >= 1.4:
                return "H2"
            else:
                return "H3"
        elif score >= 2 and size_ratio >= 1.3:
            if size_ratio >= 1.6:
                return "H2"
            else:
                return "H3"
        
        return None
    
    def _matches_heading_pattern(self, text: str) -> bool:
        """Check if text matches common heading patterns."""
        patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction"
            r'^[A-Z][A-Z\s]{2,}$',  # All caps
            r'^Chapter\s+\d+',
            r'^Section\s+\d+',
            r'^\d+\.\d+\.?\s+',  # "1.1 Subsection"
            r'^[IVX]+\.?\s+[A-Z]',  # Roman numerals
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]*)*$',  # Title case
        ]
        
        return any(re.match(pattern, text) for pattern in patterns)
    
    def _is_likely_heading_position(self, text: str) -> bool:
        """Check formatting indicators for headings."""
        # Check for title case
        words = text.split()
        if len(words) > 1:
            title_case_words = sum(1 for word in words if word[0].isupper())
            if title_case_words / len(words) >= 0.7:
                return True
        
        # Check for sentence-like structure (headings are usually not full sentences)
        if not text.endswith('.') and not text.endswith('!') and not text.endswith('?'):
            return True
        
        return False


def extract_document_title(formatted_text: List[Dict[str, Any]], font_stats: Dict[str, float]) -> str:
    """
    Extract document title from the first page.
    """
    first_page_text = [item for item in formatted_text if item["page"] == 1]
    
    if not first_page_text:
        return "Untitled Document"
    
    # Look for the largest text on the first page
    max_font_size = max(item["font_size"] for item in first_page_text)
    largest_texts = [item for item in first_page_text if item["font_size"] == max_font_size]
    
    if largest_texts:
        # Take the first largest text as title
        title_candidate = largest_texts[0]["text"]
        
        # Clean and validate title
        title = re.sub(r'[^\w\s\-\.\,\;\:\!\?]', '', title_candidate).strip()
        
        if len(title) > 3 and len(title) < 200:
            return title
    
    # Fallback: look for text that looks like a title
    for item in first_page_text:
        text = item["text"].strip()
        if (len(text) > 5 and len(text) < 100 and 
            not re.match(r'^\d+$|^page\s+\d+', text.lower())):
            return text
    
    return "Untitled Document"


def save_json_output(data: Dict[str, Any], output_path: str) -> None:
    """Save data as JSON with proper formatting."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False) 