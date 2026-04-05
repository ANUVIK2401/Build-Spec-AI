# 🏗️ BuildSpec AI

**AI QA/QC Copilot for Construction Engineering Documents**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://buildspec-ai.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI GPT-4o](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Product Overview

BuildSpec AI is a **production-ready AI-powered document review system** designed specifically for construction engineering workflows. It transforms manual specification review into an intelligent, evidence-grounded QA/QC process.

### The Problem

Engineering teams spend **hundreds of hours** manually reviewing specifications, often missing critical issues that become costly problems in the field:
- Compliance gaps that trigger code violations
- Contradictions that cause RFIs and change orders
- Coordination conflicts between disciplines
- Missing specifications that delay construction

### The Solution

BuildSpec AI uses **RAG (Retrieval-Augmented Generation)** with a multi-pass review pipeline to:

1. **Extract & Index** — Parse PDFs page-by-page with section detection
2. **Retrieve Evidence** — Find relevant passages using semantic search
3. **Analyze Multi-Pass** — Run focused review passes (completeness, contradictions, compliance)
4. **Score & Prioritize** — Rank findings by severity and impact
5. **Export Reports** — Generate structured outputs for team review

Every finding is **grounded in document evidence** with page citations.

---

## 🏢 Why This Maps to Structured AI

[Structured AI](https://structuredlabs.com) is a YC-backed company building AI infrastructure for construction. BuildSpec AI demonstrates the same core competencies:

| Structured AI Focus | BuildSpec AI Implementation |
|---------------------|----------------------------|
| Construction document intelligence | Multi-pass RAG analysis of specs |
| Engineering QA/QC automation | Severity + priority scoring |
| Multi-discipline coordination | Cross-discipline risk detection |
| Compliance checking | Code/standard gap identification |
| Grounded AI outputs | Page-cited evidence for all findings |

### What This Project Demonstrates

- **Domain Expertise**: Deep understanding of construction engineering workflows
- **Production Architecture**: Clean, modular, deployment-ready code
- **Real RAG Implementation**: Actual retrieval with semantic search (not prompt stuffing)
- **UX Design**: Professional interface for technical users
- **Startup Thinking**: Focus on value delivery and user experience

---

## ✨ Core Features

### Multi-Pass Review Pipeline

Unlike single-prompt approaches, BuildSpec AI runs **focused review passes**:

| Pass | Focus | Issue Types |
|------|-------|-------------|
| Completeness | Missing sections, incomplete specs | `missing_section`, `unclear_requirement` |
| Contradictions | Conflicting requirements, ambiguity | `contradiction`, `unclear_requirement` |
| Compliance | Code gaps, coordination risks | `compliance_gap`, `coordination_risk` |

This structured approach catches issues that broad prompts often miss.

### Review Modes

| Mode | Passes | Evidence Depth | Best For |
|------|--------|----------------|----------|
| Quick Review | 1 | 8 chunks | Initial screening |
| Standard Review | 2 | 15 chunks | Most documents |
| Deep Review | 3 | 25 chunks | Critical specifications |

### Focus Modes

Emphasize specific concerns:
- **Full Review** — Comprehensive across all categories
- **Compliance Focus** — Prioritize code/regulatory gaps
- **Coordination Focus** — Cross-discipline conflicts
- **Completeness Focus** — Missing sections and specs

### Priority Scoring

Each finding receives a priority score based on:
- Severity (high/medium/low)
- Confidence level
- Issue type impact
- Cross-reference patterns

Results: **Critical**, **Important**, or **Review** priority tags.

### Evidence Grounding

Every finding includes:
- Page citation from source document
- Evidence snippet from retrieved chunks
- Confidence level
- Recommended action

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         BuildSpec AI v2.0                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │    PDF       │───▶│   Section    │───▶│   Chunking   │          │
│  │  Extraction  │    │  Detection   │    │  (800/100)   │          │
│  │  (PyMuPDF)   │    │ (Heuristics) │    │ (LangChain)  │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                    │
│         └───────────────────┴───────────────────┘                   │
│                             │                                        │
│                             ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Embedding & Retrieval Layer                      │  │
│  │  • text-embedding-3-small                                     │  │
│  │  • Cosine similarity search                                   │  │
│  │  • Multi-query retrieval for diverse evidence                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                             │                                        │
│                             ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  Multi-Pass Analysis                          │  │
│  │  Pass 1: Completeness Check                                   │  │
│  │  Pass 2: Contradiction Analysis                               │  │
│  │  Pass 3: Compliance & Coordination                            │  │
│  │  ─────────────────────────────────                            │  │
│  │  • GPT-4o with structured prompts                             │  │
│  │  • JSON schema enforcement                                    │  │
│  │  • Robust parsing with repair                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                             │                                        │
│                             ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Post-Processing Pipeline                         │  │
│  │  • Enum normalization                                         │  │
│  │  • Priority scoring                                           │  │
│  │  • Deduplication                                              │  │
│  │  • Sorting & filtering                                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                             │                                        │
│                             ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  Streamlit UI Layer                           │  │
│  │  • Document snapshot                                          │  │
│  │  • Metrics dashboard                                          │  │
│  │  • Tabbed results (Findings, Summary, Evidence, Export)       │  │
│  │  • Filters & search                                           │  │
│  │  • Export (JSON, CSV, Markdown, TXT)                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 RAG Pipeline Details

### 1. PDF Extraction

```
PyMuPDF (fitz)
├── Page-by-page text extraction
├── Character count tracking
├── Section header detection (heuristics)
└── Graceful handling of extraction errors
```

**Section Detection Heuristics:**
- ALL CAPS lines (common headers)
- Numbered sections (1.0, SECTION 1, etc.)
- Article/Part/Division markers
- Colon-terminated titles

### 2. Chunking Strategy

```
RecursiveCharacterTextSplitter
├── chunk_size: 800 characters
├── chunk_overlap: 100 characters
├── Separators: ["\n\n", "\n", ". ", "; ", ", ", " "]
└── Metadata preserved: page_number, section, chunk_id
```

### 3. Embedding & Retrieval

```
OpenAI text-embedding-3-small
├── Batch processing (100 texts/batch)
├── Cosine similarity ranking
├── Multi-query retrieval:
│   ├── Key specs & requirements
│   ├── Safety & compliance
│   ├── Cross-discipline requirements
│   └── Focus-specific queries
└── Top-K selection based on review mode
```

### 4. Multi-Pass Analysis

Each pass uses a focused prompt template:

```python
REVIEW_PASSES = {
    'completeness': {
        'query': 'What sections appear missing or incomplete?',
        'focus_types': [MISSING_SECTION, UNCLEAR_REQUIREMENT]
    },
    'contradictions': {
        'query': 'What conflicting requirements exist?',
        'focus_types': [CONTRADICTION, UNCLEAR_REQUIREMENT]
    },
    'compliance': {
        'query': 'What compliance gaps or coordination risks?',
        'focus_types': [COMPLIANCE_GAP, COORDINATION_RISK]
    }
}
```

### 5. JSON Repair & Normalization

Robust handling of model output:
- Markdown fence stripping
- JSON array extraction
- Trailing comma fixes
- Enum value normalization
- Default value fallbacks
- Near-duplicate detection

---

## 🚀 Local Setup

### Prerequisites

- Python 3.9+
- OpenAI API key with GPT-4o access

### Installation

```bash
# Clone repository
git clone https://github.com/ANUVIK2401/Build-Spec-AI.git
cd Build-Spec-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Run Locally

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## ☁️ Streamlit Cloud Deployment

### 1. Push to GitHub

Ensure your repository is up to date:

```bash
git add .
git commit -m "Deploy BuildSpec AI"
git push origin main
```

### 2. Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository
4. Set main file: `app.py`

### 3. Configure Secrets

In Streamlit Cloud settings → Secrets:

```toml
OPENAI_API_KEY = "sk-your-api-key-here"
```

### 4. Deploy

Click "Deploy" — your app will be live in minutes!

---

## 📸 Example Findings

### Sample Output

```json
{
  "id": "F-A3B7C912",
  "type": "compliance_gap",
  "severity": "high",
  "priority": "critical",
  "discipline": "electrical",
  "confidence": "high",
  "page": 23,
  "title": "Missing emergency lighting coverage requirements",
  "description": "Section 4.2 specifies general lighting but omits emergency lighting for egress paths. This is typically required by IBC Section 1008 and could result in code violations during permitting.",
  "evidence": "The Lighting Requirements section specifies illumination levels for general spaces but makes no mention of emergency fixtures, battery backup, or egress route coverage.",
  "recommended_action": "Add emergency lighting requirements per IBC 1008 with specifications for battery backup duration, coverage area, and fixture placement along all egress routes."
}
```

### Discipline Summary

```
Mechanical: 4 findings
Electrical: 6 findings  
Structural: 2 findings
General: 3 findings
```

---

## ⚠️ Limitations

1. **Scanned PDFs** — Requires OCR preprocessing (not included)
2. **Document Size** — Very large documents (200+ pages) may hit API limits
3. **Language** — English documents only
4. **Domain Scope** — Optimized for construction/engineering; general PDFs may have lower quality results
5. **False Positives** — Findings should be verified by qualified engineers

---

## 🗺️ Future Roadmap

### Near-term
- [ ] Multi-document comparison (spec vs drawing)
- [ ] Custom review templates
- [ ] Issue tracking integration (Jira, Asana)
- [ ] Team collaboration features

### Medium-term
- [ ] OCR integration for scanned documents
- [ ] Code compliance database (IBC, NEC, ASHRAE)
- [ ] Fine-tuned models for specific disciplines
- [ ] Bulk document processing

### Long-term
- [ ] Drawing/plan analysis (CAD, PDF plans)
- [ ] Change tracking across revisions
- [ ] Automated RFI generation
- [ ] Integration with construction management platforms

---

## 💡 Founder-Facing Pitch

> **BuildSpec AI** transforms construction document review from a manual, error-prone process into an AI-assisted workflow that catches issues before they become costly field problems.
>
> ### The Problem
> Engineering teams spend hundreds of hours reviewing specifications, often missing critical compliance gaps or coordination issues that lead to RFIs, change orders, and project delays.
>
> ### Our Solution
> A RAG-powered multi-pass document review copilot that:
> - Extracts and indexes technical PDFs with section detection
> - Runs focused review passes for comprehensive coverage
> - Surfaces compliance gaps, contradictions, and coordination risks
> - Provides page-cited evidence for every finding
> - Exports structured reports for team collaboration
>
> ### Technical Differentiation
> - **Multi-pass analysis** — Not just one broad prompt, but focused passes that catch different issue types
> - **Priority scoring** — Findings ranked by severity, confidence, and impact
> - **Real RAG grounding** — Every finding backed by retrieved evidence
> - **Production architecture** — Clean code ready for deployment
>
> ### Traction
> Production-ready implementation demonstrating real engineering value, deployed and accessible on Streamlit Cloud.
>
> ### Ask
> Internship opportunity to bring this capability to Structured AI's construction intelligence platform.

---

## 🔒 Git Hygiene

Before committing, ensure these files are in `.gitignore`:

```gitignore
# Environment
.env

# Development prompts
cursor_prompt.md
claude.md
quick_start_steps.md
buildspec_cursor_prompt.md
buildspec_claude.md

# Cache
__pycache__/
.streamlit/secrets.toml
```

**Never commit API keys or development prompts to the repository.**

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Anuvik Thota**

Building AI tools for construction engineering workflows.

[GitHub](https://github.com/ANUVIK2401) • [LinkedIn](https://linkedin.com/in/anuvikthota)

---

<p align="center">
  <strong>🏗️ BuildSpec AI v2.0</strong><br>
  <em>AI QA/QC Copilot for Construction Engineering Documents</em><br>
  <sub>Multi-Pass Review • Priority Scoring • RAG-Grounded • Production Ready</sub>
</p>
