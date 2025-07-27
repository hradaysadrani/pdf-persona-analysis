# Challenge 1B: Persona-Driven Document Intelligence 🧠

Analyze document collections and extract the most relevant sections based on user persona and specific job-to-be-done using advanced AI.

## 🎯 What This Does

This solution automatically:
- **Analyzes multiple PDFs** as a collection (3-10 documents)
- **Understands user personas** (researchers, analysts, students, etc.)
- **Matches content to jobs** (literature review, financial analysis, exam prep)
- **Ranks sections by relevance** using semantic AI analysis
- **Extracts key subsections** with detailed text for actionable insights

## 🚀 Quick Start (For Complete Beginners)

### Step 1: Install Docker
(Same as Challenge 1A - see [installation guide](../part1-outline-extractor/README.md#step-1-install-docker))

### Step 2: Prepare Your Documents
1. Create a folder called `input` in this directory
2. Put 3-10 related PDF files inside the `input` folder
3. Create an empty folder called `output`

Example document collections:
- **Research Papers**: Related academic papers on a topic
- **Financial Reports**: Company annual reports, earnings statements
- **Travel Guides**: City guides, restaurant lists, activity recommendations
- **Technical Manuals**: Software documentation, user guides

Your folder structure:
```
part2-persona-analysis/
├── input/              ← Put related PDF files here
│   ├── research_paper1.pdf
│   ├── research_paper2.pdf
│   └── research_paper3.pdf
├── output/             ← Results will appear here
├── persona_analyzer.py
├── Dockerfile
└── README.md
```

### Step 3: Build the AI Solution
```bash
docker build --platform linux/amd64 -t persona-analyzer .
```

### Step 4: Run the Analysis

**Option A: Automatic Detection (Recommended)**
The system automatically detects your persona and job based on document types:
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  persona-analyzer
```

**Option B: Specify Custom Persona and Job**
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -e PERSONA="Investment Analyst with 5+ years experience" \
  -e JOB_TO_BE_DONE="Analyze revenue trends and identify investment risks" \
  --network none \
  persona-analyzer
```

**For Windows PowerShell:**
```powershell
docker run --rm -v ${PWD}/input:/app/input -v ${PWD}/output:/app/output -e PERSONA="Research Student" -e JOB_TO_BE_DONE="Prepare literature review" --network none persona-analyzer
```

### Step 5: Analyze Results
Check your `output` folder for `challenge1b_output.json` with ranked insights!

## 📊 Example Output

### Input: Travel Planning Documents
**Persona**: "Travel Planner"  
**Job**: "Plan a 4-day trip for 10 college friends"

**Output**:
```json
{
  "metadata": {
    "input_documents": [
      "South of France - Cities.pdf",
      "South of France - Cuisine.pdf", 
      "South of France - Things to Do.pdf"
    ],
    "persona": "Travel Planner",
    "job_to_be_done": "Plan a trip of 4 days for a group of 10 college friends",
    "processing_timestamp": "2024-01-15T10:30:45"
  },
  "extracted_sections": [
    {
      "document": "South of France - Things to Do.pdf",
      "section_title": "Coastal Adventures", 
      "importance_rank": 1,
      "page_number": 2
    },
    {
      "document": "South of France - Cuisine.pdf",
      "section_title": "Budget-Friendly Restaurants",
      "importance_rank": 2, 
      "page_number": 5
    }
  ],
  "subsection_analysis": [
    {
      "document": "South of France - Things to Do.pdf",
      "refined_text": "Beach activities perfect for groups: Nice offers vibrant beaches with group-friendly volleyball courts and affordable beach clubs. Antibes provides budget-conscious options with public beaches and nearby hostels...",
      "page_number": 2
    }
  ]
}
```

## 🤖 How the AI Works

### 1. Automatic Persona Detection
The system analyzes document names and types to automatically determine appropriate personas:

- **Research Papers** → "PhD Researcher"
- **Financial Reports** → "Investment Analyst"  
- **Travel Guides** → "Travel Planner"
- **Technical Docs** → "Software Developer"
- **Food/Recipe PDFs** → "Food Contractor"
- **Academic Textbooks** → "Graduate Student"

### 2. Semantic Document Understanding
- Uses **sentence transformers** (all-MiniLM-L6-v2, ~80MB AI model)
- Converts text to **semantic embeddings** (meaning vectors)
- Matches content meaning to user needs, not just keywords

### 3. Relevance Scoring
```
Query = "Persona: [User Role]. Task: [Job to be done]"
For each document section:
  1. Create combined text = section_title + content_preview
  2. Generate AI embedding for combined text
  3. Calculate similarity to query embedding  
  4. Rank by similarity score
```

### 4. Hierarchical Analysis
- **Section-level**: Ranks document sections by relevance
- **Subsection-level**: Extracts specific actionable paragraphs
- **Multi-strategy extraction**: Paragraphs → sentence groups → content windows

## 🌍 Multi-Domain Examples

### Academic Research
```bash
# Documents: research_paper1.pdf, research_paper2.pdf, journal_article.pdf
# Auto-detected: "PhD Researcher" + "Literature review and methodology analysis"

docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  persona-analyzer
```

### Investment Analysis
```bash
# Documents: annual_report_2023.pdf, quarterly_earnings.pdf, competitor_analysis.pdf  
# Custom persona for specific focus:

docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -e PERSONA="Investment Analyst" \
  -e JOB_TO_BE_DONE="Analyze revenue trends and market positioning for investment decision" \
  --network none \
  persona-analyzer
```

### Student Exam Preparation
```bash
# Documents: chemistry_chapter1.pdf, chemistry_chapter2.pdf, practice_problems.pdf

docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -e PERSONA="Undergraduate Chemistry Student" \
  -e JOB_TO_BE_DONE="Identify key concepts and mechanisms for organic chemistry exam" \
  --network none \
  persona-analyzer
```

## 📏 Performance Specifications

- **Processing Time**: ≤60 seconds for 3-5 documents
- **AI Model Size**: ~80MB (well under 1GB limit)
- **Architecture**: CPU-only processing (no GPU needed)
- **Memory**: Optimized for documents up to 10MB each
- **Offline**: No internet required after model download

## 🎯 Supported Document Types

✅ **Research Papers** (PDF, academic journals)  
✅ **Financial Reports** (annual reports, earnings statements)  
✅ **Technical Documentation** (user manuals, API docs)  
✅ **Travel Guides** (city guides, hotel listings)  
✅ **Educational Content** (textbooks, lecture notes)  
✅ **Business Documents** (proposals, executive summaries)  
✅ **Food & Recipe Collections** (cookbooks, menu planning)

## 🔧 Customizing Personas and Jobs

### Effective Persona Examples:
- `"Investment Analyst with 5+ years experience in tech stocks"`
- `"PhD Researcher specializing in machine learning and NLP"`
- `"Undergraduate student preparing for organic chemistry final exam"`
- `"Travel planner organizing group trips for young adults"`
- `"Software engineer implementing REST API integrations"`

### Effective Job-to-be-Done Examples:
- `"Analyze revenue trends and identify investment opportunities"`
- `"Conduct literature review focusing on recent advances and limitations"`
- `"Extract key reaction mechanisms and synthesis pathways for exam"`
- `"Plan 5-day itinerary with budget-friendly activities for 8 people"`
- `"Understand authentication flows and error handling best practices"`

## 🐛 Troubleshooting

### "No sections found"
**Check**:
1. PDFs contain text (not just images)
2. Documents are related to each other
3. PDFs are not password-protected

### "Low relevance scores"
**Solutions**:
1. Make persona more specific
2. Clarify the job-to-be-done with concrete objectives
3. Ensure documents match the stated task

### "Processing takes too long"
**Causes**:
- Very large PDF files (>50MB each)
- Too many documents (>10 files)
- Complex document structure

**Solutions**:
- Reduce file sizes or number of documents
- Split large documents into smaller parts

### "Subsection analysis is empty"
**Reasons**:
- Document sections too short
- No clear paragraph structure
- Content not relevant to query

## 🧪 Testing with Adobe Test Cases

```bash
# Test with Adobe Collection 1 (Travel Planning)
cp ../Adobe-India-Hackathon25/Challenge_1b/Collection\ 1/PDFs/*.pdf input/

docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  -e PERSONA="Travel Planner" \
  -e JOB_TO_BE_DONE="Plan a trip of 4 days for a group of 10 college friends" \
  --network none \
  persona-analyzer

# Validate output
python ../validate.py 1b output/challenge1b_output.json
```

## 📁 File Structure

```
part2-persona-analysis/
├── persona_analyzer.py       # Main AI analysis script
├── Dockerfile                # Container with AI model
├── requirements.txt           # Python dependencies  
├── README.md                 # This file
├── TECHNICAL_GUIDE.md        # Debugging guide
├── approach_explanation.md   # Methodology explanation
├── input/                    # Put PDF collections here
└── output/                   # Results appear here
    └── challenge1b_output.json
```

The semantic AI approach ensures content is matched by meaning rather than just keywords, providing superior relevance for diverse document types and user needs.

## 🚀 Next Steps

After running the analysis:
1. **Review extracted sections** - Are they relevant to your needs?
2. **Read subsection analysis** - These contain the most actionable insights  
3. **Adjust persona/job** if needed for better results
4. **Process new document collections** for different tasks

The AI learns from your document patterns and provides increasingly relevant results! 