# Technical Guide: persona_analyzer.py 🤖

This document explains how `persona_analyzer.py` works internally to help with debugging and understanding the AI-powered document analysis.

## 📋 Overview

The AI analysis follows this pipeline:
1. **Document Section Extraction** → Break PDFs into meaningful sections
2. **Persona Detection** → Auto-detect or use provided persona/job
3. **Semantic Embedding** → Convert text to AI vector representations
4. **Relevance Scoring** → Calculate similarity between query and content
5. **Hierarchical Analysis** → Extract actionable subsections
6. **JSON Output** → Generate structured results

## 🔍 Function-by-Function Breakdown

### `PersonaAnalyzer.__init__(model_name)`
**Purpose**: Initialize the AI model for semantic analysis

**What it does**:
```python
# Load sentence transformer model (~80MB)
self.model = SentenceTransformer('all-MiniLM-L6-v2')

# This model converts text to 384-dimensional vectors
# Each dimension captures different semantic aspects
```

**Debug tips**:
- If model loading fails → Check internet connection (first run only)
- If memory issues → Model requires ~200MB RAM
- Model is cached after first download

### `extract_document_sections(pdf_path)`
**Purpose**: Break PDF into meaningful sections with content

**Algorithm**:
```python
# Open PDF with PyMuPDF
doc = fitz.open(pdf_path)

# For each page, extract text blocks
for page_num in range(len(doc)):
    page = doc[page_num]
    blocks = page.get_text("dict")["blocks"]
    
    # For each text span, check if it's a section header
    if self._is_section_header(span, text):
        # Save previous section, start new one
        sections.append({
            "document": filename,
            "section_title": title,
            "page_number": page_num + 1,  # 1-based
            "content": accumulated_content
        })
```

**Section Detection Strategy**:
1. **Header patterns**: "1. Introduction", "Chapter 1", "Abstract"
2. **Font analysis**: Bold text, larger fonts
3. **Content accumulation**: Collect text until next header
4. **Smart splitting**: Break very long sections (>2000 chars)

**Debug tips**:
```python
# Add logging to see section detection:
print(f"Found {len(sections)} sections:")
for i, section in enumerate(sections[:5]):
    print(f"{i+1}. {section['section_title']} (page {section['page_number']})")
    print(f"   Content length: {len(section['content'])} chars")
```

### `_is_section_header(span, text)`
**Purpose**: Detect if text is likely a section header

**Scoring Algorithm**:
```python
score = 0

# Pattern matching (+2 points)
if matches_header_patterns:
    score += 2

# Font formatting (+1 point each)  
if is_bold: score += 1
if is_larger_font: score += 1

# Threshold: score >= 2 to be considered header
return score >= 2
```

**Header Patterns**:
```python
patterns = [
    r'^\d+\.?\s+[A-Z]',           # "1. Introduction"
    r'^[A-Z][A-Z\s]{2,}$',        # "INTRODUCTION" 
    r'^(Chapter|Section|Part)\s+\d+', # "Chapter 1"
    r'^\d+\.\d+\.?\s+',           # "1.1 Subsection"
    r'^(Abstract|Introduction|Conclusion)', # Common sections
    r'^[A-Z][a-z]+(\s+[A-Z][a-z]*){1,4}$' # "Title Case Headers"
]
```

**Debug tips**:
```python
# Test header detection on specific text:
test_spans = [
    {"size": 16, "flags": 16, "text": "1. Introduction"},
    {"size": 12, "flags": 0, "text": "This is body text"},
    {"size": 14, "flags": 16, "text": "Chapter 2: Methods"}
]

for span in test_spans:
    is_header = self._is_section_header(span, span["text"])
    print(f"'{span['text']}' → Header: {is_header}")
```

### `determine_persona_and_job(pdf_files)`
**Purpose**: Auto-detect appropriate persona based on document types

**Pattern Matching**:
```python
# Analyze filenames for keywords
doc_names = [Path(f).name.lower() for f in pdf_files]
all_names = ' '.join(doc_names)

# Academic patterns
if any(keyword in all_names for keyword in ['paper', 'research', 'journal']):
    return "PhD Researcher", "Conduct comprehensive literature review..."

# Financial patterns
if any(keyword in all_names for keyword in ['annual', 'report', 'financial']):
    return "Investment Analyst", "Analyze financial performance..."

# Travel patterns  
if any(keyword in all_names for keyword in ['travel', 'guide', 'hotel']):
    return "Travel Planner", "Plan comprehensive itinerary..."
```

**Supported Auto-Detection**:
- **Academic**: paper, research, study, journal, ieee, acm
- **Financial**: annual, report, financial, earnings, quarterly
- **Educational**: textbook, chemistry, physics, math, biology
- **Travel**: travel, guide, city, hotel, restaurant, tourism
- **Food**: recipe, cooking, cuisine, dinner, lunch, breakfast
- **Technical**: software, api, programming, development, manual

**Debug tips**:
```python
# See what keywords are detected:
doc_names = [Path(f).name.lower() for f in pdf_files]
print(f"Document names: {doc_names}")
print(f"Combined text: {' '.join(doc_names)}")

persona, job = determine_persona_and_job(pdf_files)
print(f"Detected persona: {persona}")
print(f"Detected job: {job}")
```

### `calculate_relevance_scores(sections, persona, job_to_be_done)`
**Purpose**: Core AI analysis - rank sections by semantic relevance

**Embedding Process**:
```python
# Create comprehensive query
query = f"Persona: {persona}. Task: {job_to_be_done}"

# Prepare section texts for AI
section_texts = []
for section in sections:
    # Combine title + content for better context
    content_snippet = section["content"][:800]  # Limit for efficiency
    combined_text = f"Title: {section['section_title']}. Content: {content_snippet}"
    section_texts.append(combined_text)

# Generate AI embeddings (384-dimensional vectors)
query_embedding = self.model.encode([query])           # Shape: (1, 384)
section_embeddings = self.model.encode(section_texts)  # Shape: (N, 384)

# Calculate cosine similarity
similarities = cosine_similarity(query_embedding, section_embeddings)[0]

# Assign scores and sort
for i, section in enumerate(sections):
    section["relevance_score"] = float(similarities[i])
sections.sort(key=lambda x: x["relevance_score"], reverse=True)
```

**Understanding Similarity Scores**:
- **0.8-1.0**: Extremely relevant (rare, perfect match)
- **0.6-0.8**: Highly relevant (likely top results)
- **0.4-0.6**: Moderately relevant (good supporting content)
- **0.2-0.4**: Somewhat relevant (background information)
- **0.0-0.2**: Low relevance (likely unrelated)

**Debug tips**:
```python
# Print top similarities to understand scoring:
print(f"Query: {query}")
print("\nTop 5 section relevance scores:")
for i, section in enumerate(ranked_sections[:5]):
    print(f"{i+1}. Score: {section['relevance_score']:.4f}")
    print(f"   Title: {section['section_title']}")
    print(f"   Preview: {section['content'][:100]}...")
    print()
```

### `extract_subsection_analysis(top_sections, persona, job_to_be_done)`
**Purpose**: Extract actionable paragraphs from most relevant sections

**Multi-Strategy Text Splitting**:
```python
# Strategy 1: Split by paragraph breaks (preferred)
paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 150]

# Strategy 2: Group sentences if no clear paragraphs  
if len(paragraphs) < 2:
    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 50]
    paragraphs = ['. '.join(sentences[i:i+4]) for i in range(0, len(sentences), 4)]

# Strategy 3: Fixed-size content windows
if len(paragraphs) < 2:
    chunk_size = max(200, len(content) // 3)
    for i in range(0, len(content), chunk_size):
        chunk = content[i:i+chunk_size]
        if len(chunk) > 150:
            paragraphs.append(chunk)
```

**Subsection Selection**:
- Process top 5 sections maximum
- Extract up to 2 paragraphs per section
- Minimum paragraph length: 150 characters
- AI ranking applied to selected paragraphs

**Debug tips**:
```python
# See how content is being split:
for section in top_sections[:3]:
    content = section["content"]
    paragraphs = content.split('\n\n')
    print(f"Section: {section['section_title']}")
    print(f"Content length: {len(content)}")
    print(f"Paragraphs found: {len(paragraphs)}")
    print(f"Substantial paragraphs: {len([p for p in paragraphs if len(p.strip()) > 150])}")
    print()
```

## 🧪 Testing Individual Components

### Test Persona Detection:
```python
test_files = [
    "financial_report_2023.pdf",
    "quarterly_earnings.pdf", 
    "revenue_analysis.pdf"
]
persona, job = determine_persona_and_job(test_files)
print(f"Detected: {persona} → {job}")
```

### Test Section Extraction:
```python
sections = analyzer.extract_document_sections("test.pdf")
print(f"Extracted {len(sections)} sections:")
for section in sections:
    print(f"- {section['section_title']} (page {section['page_number']})")
```

### Test AI Embeddings:
```python
# Test similarity calculation
test_query = "Investment analysis and financial performance"
test_texts = [
    "Revenue growth increased by 15% year over year",
    "The weather was sunny and warm yesterday",
    "Quarterly earnings exceeded analyst expectations"
]

query_emb = analyzer.model.encode([test_query])
text_embs = analyzer.model.encode(test_texts)
similarities = cosine_similarity(query_emb, text_embs)[0]

for text, sim in zip(test_texts, similarities):
    print(f"Similarity: {sim:.4f} | Text: {text}")
```

## 🐛 Common Issues & Solutions

### Issue: "No sections found"
**Debugging**:
```python
# Check raw text extraction
import fitz
doc = fitz.open("problem.pdf")
for page_num in range(min(3, len(doc))):  # Check first 3 pages
    page = doc[page_num]
    text = page.get_text()
    print(f"Page {page_num} text length: {len(text)}")
    print(f"First 200 chars: {text[:200]}")
```

**Common causes**:
- Image-based PDFs (scanned documents)
- Password-protected files
- Unusual PDF structure

### Issue: "Low relevance scores"
**Debugging**:
```python
# Check query construction
query = f"Persona: {persona}. Task: {job_to_be_done}"
print(f"Query: {query}")

# Check section text preparation
for section in sections[:3]:
    content_snippet = section["content"][:200]
    combined_text = f"Title: {section['section_title']}. Content: {content_snippet}"
    print(f"Section text: {combined_text}")
```

**Solutions**:
- Make persona more specific and detailed
- Clarify job-to-be-done with concrete objectives
- Ensure documents are actually related to the task

### Issue: "Empty subsection analysis"
**Debugging**:
```python
# Check content splitting
for section in top_sections[:3]:
    content = section["content"]
    print(f"Section: {section['section_title']}")
    print(f"Content length: {len(content)}")
    
    # Test different splitting strategies
    para_split = content.split('\n\n')
    print(f"Paragraph splits: {len([p for p in para_split if len(p.strip()) > 150])}")
    
    sent_split = content.split('.')
    print(f"Sentence splits: {len([s for s in sent_split if len(s.strip()) > 50])}")
```

## 📊 Performance Monitoring

### Track Processing Time:
```python
import time

# Section extraction timing
start = time.time()
sections = analyzer.extract_document_sections(pdf_path)
print(f"Section extraction: {time.time() - start:.2f}s")

# AI embedding timing  
start = time.time()
ranked_sections = analyzer.calculate_relevance_scores(sections, persona, job)
print(f"AI analysis: {time.time() - start:.2f}s")

# Total analysis timing
start = time.time()
result = analyzer.analyze_document_collection(pdf_paths, persona, job)
print(f"Total analysis: {time.time() - start:.2f}s")
```

### Monitor Memory Usage:
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")

# Check after model loading
analyzer = PersonaAnalyzer()
print(f"After model load: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

## 🎯 Optimization Tips

### For Better Relevance:
1. **Specific personas**: "Investment Analyst with 5+ years in tech stocks"
2. **Concrete jobs**: "Identify revenue growth drivers and market risks"
3. **Domain keywords**: Include industry-specific terms

### For Performance:
1. **Content limits**: Already optimized (800 chars for sections, 500 for subsections)
2. **Batch processing**: Model encodes multiple texts efficiently
3. **Memory management**: Process documents sequentially for large collections

### For Accuracy:
1. **Quality sections**: Better section detection → better AI analysis
2. **Relevant documents**: Ensure all PDFs relate to the task
3. **Clear structure**: Well-formatted PDFs with headings work best

This guide should help you understand and debug the AI-powered document analysis process! 