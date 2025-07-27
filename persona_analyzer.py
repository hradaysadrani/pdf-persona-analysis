#!/usr/bin/env python3
"""
Persona-Driven Document Intelligence
Analyzes document collections based on user persona and job-to-be-done.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime
import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentSection:
    """Represents a section of a document."""
    
    def __init__(self, title: str, content: str, page: int, document_name: str, 
                 start_position: int = 0, font_size: float = 12.0):
        self.title = title
        self.content = content
        self.page = page
        self.document_name = document_name
        self.start_position = start_position
        self.font_size = font_size
        self.importance_score = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "document_name": self.document_name,
            "page": self.page,
            "section_title": self.title,
            "importance_rank": 0,  # Will be set during ranking
            "content_preview": self.content[:200] + "..." if len(self.content) > 200 else self.content
        }


class PersonaAnalyzer:
    """Persona-driven document intelligence system."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with lightweight sentence transformer model (~80MB)."""
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def extract_document_sections(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract meaningful sections from a PDF document."""
        document_name = Path(pdf_path).name
        logger.info(f"Extracting sections from: {document_name}")
        
        try:
            doc = fitz.open(pdf_path)
            sections = []
            current_section = None
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]
                
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if not text:
                                    continue
                                
                                # Check if this is a section header
                                if self._is_section_header(span, text):
                                    # Save previous section if it has substantial content
                                    if (current_section and 
                                        len(current_section["content"].strip()) > 100):
                                        sections.append({
                                            "document": document_name,
                                            "section_title": current_section["title"],
                                            "page_number": current_section["page"],
                                            "content": current_section["content"].strip()
                                        })
                                    
                                    # Start new section
                                    current_section = {
                                        "title": text,
                                        "content": "",
                                        "page": page_num + 1  # 1-based for output
                                    }
                                else:
                                    # Add to current section content
                                    if current_section:
                                        current_section["content"] += " " + text
                                    else:
                                        # Handle documents without clear headers
                                        current_section = {
                                            "title": "Document Content",
                                            "content": text,
                                            "page": page_num + 1
                                        }
                
                # Handle page breaks - if we have a very long section, split it
                if (current_section and 
                    len(current_section["content"].strip()) > 2000):
                    sections.append({
                        "document": document_name,
                        "section_title": current_section["title"],
                        "page_number": current_section["page"],
                        "content": current_section["content"].strip()
                    })
                    current_section = None
            
            # Add final section
            if (current_section and 
                len(current_section["content"].strip()) > 100):
                sections.append({
                    "document": document_name,
                    "section_title": current_section["title"],
                    "page_number": current_section["page"],
                    "content": current_section["content"].strip()
                })
            
            doc.close()
            logger.info(f"Extracted {len(sections)} sections from {document_name}")
            return sections
            
        except Exception as e:
            logger.error(f"Error extracting sections from {pdf_path}: {str(e)}")
            return []
    
    def _is_section_header(self, span: Dict[str, Any], text: str) -> bool:
        """Determine if text span is likely a section header."""
        font_size = span.get("size", 12)
        flags = span.get("flags", 0)
        
        # Basic length and content filters
        if len(text) < 5 or len(text) > 120:
            return False
        
        # Skip obviously non-header text
        if re.match(r'^\d+$|^page\s+\d+|^figure\s+\d+|^table\s+\d+', text.lower()):
            return False
        
        # Pattern-based detection (domain-agnostic)
        header_patterns = [
            r'^\d+\.?\s+[A-Z\u4e00-\u9fff]',  # "1. Introduction" 
            r'^[A-Z\u4e00-\u9fff][A-Z\u4e00-\u9fff\s]{2,}$',  # All caps
            r'^(Chapter|Section|Part)\s+\d+',  # Chapter/Section indicators
            r'^\d+\.\d+\.?\s+',  # "1.1 Subsection"
            r'^(Abstract|Introduction|Conclusion|References|Bibliography)',  # Common academic sections
            r'^(Executive Summary|Overview|Background|Methodology)',  # Business document sections
            r'^[A-Z][a-z]+(\s+[A-Z][a-z]*){1,4}$',  # Title case (2-5 words)
        ]
        
        pattern_match = any(re.match(pattern, text, re.IGNORECASE) for pattern in header_patterns)
        
        # Font-based detection
        is_bold = bool(flags & 16)
        is_larger = font_size > 12  # Assuming 12pt as baseline
        
        # Combined scoring
        score = 0
        if pattern_match:
            score += 2
        if is_bold:
            score += 1
        if is_larger:
            score += 1
        
        return score >= 2
    
    def calculate_relevance_scores(self, sections: List[Dict[str, Any]], 
                                   persona: str, job_to_be_done: str) -> List[Dict[str, Any]]:
        """Calculate relevance scores using semantic similarity."""
        logger.info("Calculating relevance scores using semantic analysis...")
        
        if not sections:
            return sections
        
        # Create comprehensive query
        query = f"Persona: {persona}. Task: {job_to_be_done}"
        logger.info(f"Query: {query}")
        
        # Prepare section texts for embedding
        section_texts = []
        for section in sections:
            # Combine title and content snippet for better semantic matching
            content_snippet = section["content"][:800]  # Reasonable length for embeddings
            combined_text = f"Title: {section['section_title']}. Content: {content_snippet}"
            section_texts.append(combined_text)
        
        try:
            # Generate embeddings
            logger.info("Generating semantic embeddings...")
            query_embedding = self.model.encode([query])
            section_embeddings = self.model.encode(section_texts, show_progress_bar=False)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, section_embeddings)[0]
            
            # Assign relevance scores
            for i, section in enumerate(sections):
                section["relevance_score"] = float(similarities[i])
            
            # Sort by relevance score (descending)
            sections.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            logger.info(f"Calculated relevance scores for {len(sections)} sections")
            return sections
            
        except Exception as e:
            logger.error(f"Error calculating relevance scores: {str(e)}")
            # Return sections with default scores
            for section in sections:
                section["relevance_score"] = 0.0
            return sections
    
    def extract_subsection_analysis(self, top_sections: List[Dict[str, Any]], 
                                   persona: str, job_to_be_done: str, 
                                   max_subsections: int = 5) -> List[Dict[str, Any]]:
        """Extract and rank subsections from top-ranked sections."""
        logger.info("Extracting subsection analysis...")
        
        subsections = []
        query = f"Persona: {persona}. Task: {job_to_be_done}"
        
        # Process top 3-5 sections for subsection analysis
        for section in top_sections[:min(5, len(top_sections))]:
            content = section["content"]
            
            # Strategy 1: Split by paragraph breaks
            paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 150]
            
            # Strategy 2: Split by sentence groups if no clear paragraphs
            if len(paragraphs) < 2:
                sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 50]
                # Group sentences into meaningful chunks
                paragraphs = []
                for i in range(0, len(sentences), 4):  # Groups of 4 sentences
                    chunk = '. '.join(sentences[i:i+4])
                    if len(chunk) > 150:
                        paragraphs.append(chunk)
            
            # Strategy 3: Use content windows if still no good chunks
            if len(paragraphs) < 2:
                chunk_size = max(200, len(content) // 3)
                for i in range(0, len(content), chunk_size):
                    chunk = content[i:i+chunk_size]
                    if len(chunk) > 150:
                        paragraphs.append(chunk)
            
            # Select best subsections from this section
            for paragraph in paragraphs[:2]:  # Max 2 per section
                if len(paragraph.strip()) > 150:  # Ensure substantial content
                    subsections.append({
                        "document": section["document"],
                        "refined_text": paragraph.strip(),
                        "page_number": section["page_number"],
                        "relevance_score": 0.0,
                        "parent_section": section["section_title"]
                    })
        
        # Calculate relevance scores for subsections
        if subsections:
            try:
                subsection_texts = [sub["refined_text"][:500] for sub in subsections]
                query_embedding = self.model.encode([query])
                subsection_embeddings = self.model.encode(subsection_texts, show_progress_bar=False)
                
                similarities = cosine_similarity(query_embedding, subsection_embeddings)[0]
                
                for i, subsection in enumerate(subsections):
                    subsection["relevance_score"] = float(similarities[i])
                
                # Sort by relevance
                subsections.sort(key=lambda x: x["relevance_score"], reverse=True)
                
                logger.info(f"Analyzed {len(subsections)} subsections")
                
            except Exception as e:
                logger.error(f"Error calculating subsection relevance: {str(e)}")
        
        # Return top subsections, removing internal scoring fields
        result_subsections = []
        for i, sub in enumerate(subsections[:max_subsections]):
            result_subsections.append({
                "document": sub["document"],
                "refined_text": sub["refined_text"],
                "page_number": sub["page_number"]
            })
        
        return result_subsections
    
    def analyze_document_collection(self, pdf_paths: List[str], persona: str, 
                                   job_to_be_done: str) -> Dict[str, Any]:
        """Analyze a collection of documents for persona-specific insights."""
        logger.info(f"Analyzing collection of {len(pdf_paths)} documents")
        logger.info(f"Persona: {persona}")
        logger.info(f"Job to be done: {job_to_be_done}")
        
        start_time = time.time()
        
        # Extract sections from all documents
        all_sections = []
        input_documents = []
        
        for pdf_path in pdf_paths:
            input_documents.append(Path(pdf_path).name)
            sections = self.extract_document_sections(pdf_path)
            all_sections.extend(sections)
        
        logger.info(f"Extracted total of {len(all_sections)} sections")
        
        # Calculate relevance scores
        ranked_sections = self.calculate_relevance_scores(all_sections, persona, job_to_be_done)
        
        # Prepare extracted sections (top 5 for output)
        extracted_sections = []
        for i, section in enumerate(ranked_sections[:5]):
            extracted_sections.append({
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": i + 1,
                "page_number": section["page_number"]
            })
        
        # Extract subsection analysis
        subsection_analysis = self.extract_subsection_analysis(ranked_sections, persona, job_to_be_done)
        
        processing_time = time.time() - start_time
        
        # Create ISO timestamp
        timestamp = datetime.now().isoformat()
        
        result = {
            "metadata": {
                "input_documents": input_documents,
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": timestamp
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        return result

def determine_persona_and_job(pdf_files: List[str]) -> Tuple[str, str]:
    """
    Determine persona and job based on document collection characteristics.
    This handles the generic requirement by analyzing document types.
    """
    
    # Analyze document names and content patterns to infer likely use case
    doc_names = [Path(f).name.lower() for f in pdf_files]
    all_names = ' '.join(doc_names)
    
    # Research/Academic patterns
    academic_keywords = ['paper', 'research', 'study', 'journal', 'ieee', 'acm', 'proceedings', 'conference']
    if any(keyword in all_names for keyword in academic_keywords):
        return "PhD Researcher", "Conduct comprehensive literature review and identify key methodologies, findings, and research gaps"
    
    # Financial/Business patterns  
    business_keywords = ['annual', 'report', 'financial', 'earnings', 'quarterly', 'revenue', 'investor']
    if any(keyword in all_names for keyword in business_keywords):
        return "Investment Analyst", "Analyze financial performance, revenue trends, market positioning, and investment opportunities"
    
    # Educational/Textbook patterns
    education_keywords = ['chapter', 'textbook', 'chemistry', 'physics', 'math', 'biology', 'learn', 'guide']
    if any(keyword in all_names for keyword in education_keywords):
        return "Graduate Student", "Extract key concepts, methodologies, and important information for comprehensive understanding"
    
    # Travel/Lifestyle patterns
    travel_keywords = ['travel', 'guide', 'city', 'hotel', 'restaurant', 'tourism', 'trip']
    if any(keyword in all_names for keyword in travel_keywords):
        return "Travel Planner", "Plan comprehensive itinerary with activities, accommodations, and practical recommendations"
    
    # Food/Culinary patterns
    food_keywords = ['recipe', 'cooking', 'cuisine', 'dinner', 'lunch', 'breakfast', 'food']
    if any(keyword in all_names for keyword in food_keywords):
        return "Food Contractor", "Design comprehensive menu with diverse options including dietary restrictions and preparation guidelines"
    
    # Technical/Software patterns
    tech_keywords = ['software', 'api', 'programming', 'development', 'technical', 'manual', 'documentation']
    if any(keyword in all_names for keyword in tech_keywords):
        return "Software Developer", "Extract technical specifications, implementation guidelines, and best practices"
    
    # Default fallback for generic documents
    return "Business Analyst", "Extract key insights, important information, and actionable recommendations"

def main():
    """Main entry point for persona-driven analysis."""
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    logger.info("Persona-Driven Document Intelligence")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files
    pdf_paths = [str(p) for p in Path(input_dir).glob("*.pdf")]
    
    if not pdf_paths:
        logger.error("No PDF files found to process")
        sys.exit(1)
    
    logger.info(f"Found {len(pdf_paths)} PDF files")
    
    # Determine persona and job based on document collection
    # This provides the "generic" capability to handle diverse inputs
    persona, job_to_be_done = determine_persona_and_job(pdf_paths)
    
    # Allow override via environment variables for specific test cases
    persona = os.getenv("PERSONA", persona)
    job_to_be_done = os.getenv("JOB_TO_BE_DONE", job_to_be_done)
    
    logger.info(f"Persona: {persona}")
    logger.info(f"Job to be done: {job_to_be_done}")
    
    # Initialize analyzer and process
    analyzer = PersonaAnalyzer()
    result = analyzer.analyze_document_collection(pdf_paths, persona, job_to_be_done)
    
    # Save output
    output_file = os.path.join(output_dir, "challenge1b_output.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Analysis complete. Results saved to: {output_file}")

if __name__ == "__main__":
    main() 