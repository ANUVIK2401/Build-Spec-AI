# 🏗️ BuildSpec AI

**AI QA/QC Copilot for Construction Engineering Documents**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://buildspec-ai.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Product Overview

BuildSpec AI is a production-ready AI-powered document review tool designed specifically for **construction engineering documents**. It uses **Retrieval-Augmented Generation (RAG)** to analyze technical PDFs and surface critical quality issues with page-cited evidence.

### What It Does

Upload any construction engineering document (mechanical specs, electrical designs, structural drawings, etc.) and BuildSpec AI will identify:

- 🔴 **Compliance Gaps** - Missing code requirements, incomplete safety specifications
- 🟡 **Contradictions** - Conflicting requirements between sections
- 🟢 **Missing Sections** - Required specifications that appear absent
- 🔵 **Unclear Requirements** - Vague or ambiguous specifications
- 🟣 **Coordination Risks** - Potential MEP conflicts, unclear handoffs

Each finding includes:
- Severity classification (High/Medium/Low)
- Discipline tag (Mechanical/Electrical/Structural/General)
- Confidence level
- **Page citation** from the source document
- Evidence snippet from the retrieved context
- Recommended action to resolve

---

## 🏢 Why This Maps to Structured AI

[Structured AI](https://structuredlabs.com) is a YC-backed company building AI tools for construction and engineering workflows. BuildSpec AI demonstrates the same core competencies:

| Structured AI Focus | BuildSpec AI Implementation |
|---------------------|----------------------------|
| Construction document intelligence | RAG-powered PDF analysis for specs |
| QA/QC automation | Automated issue detection with severity |
| Engineering compliance | Compliance gap and code issue detection |
| Multi-discipline coordination | MEP coordination risk identification |
| Grounded AI outputs | Page-cited findings with evidence |

This project showcases:
- **Domain expertise** in construction engineering workflows
- **RAG implementation** with real retrieval (not prompt stuffing)
- **Production-ready code** deployable to Streamlit Cloud
- **Clean UX** designed for technical professionals

---

## ✨ Features

### Core Analysis
- **Real RAG Pipeline** - Embeddings + vector similarity retrieval
- **GPT-4o Analysis** - Structured JSON output with validation
- **Page-Level Citations** - Every finding linked to source pages
- **Multi-Query Retrieval** - Diverse evidence gathering

### Review Modes
| Mode | Chunks Retrieved | Max Issues | Use Case |
|------|-----------------|------------|----------|
| Quick Review | 5 | 5 | Initial screening |
| Standard Review | 10 | 10 | Most documents |
| Deep Review | 20 | 20 | Critical documents |

### User Experience
- 🎨 Premium dark theme with modern UI
- 📊 Metrics dashboard with severity breakdown
- 🔍 Filters for type, severity, discipline, confidence
- 📋 Table and card views for findings
- 📥 Export to JSON, CSV, Markdown

### Developer Features
- 🔧 Debug mode with raw model output
- 📑 Retrieved chunk inspection
- 🚀 Single-file deployment
- ☁️ Streamlit Cloud ready

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BuildSpec AI                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │   PDF    │───▶│  Chunk   │───▶│  Embed   │───▶│ Retrieve │  │
│  │ Extract  │    │  Pages   │    │ Chunks   │    │ Top-K    │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │         │
│       │               │               │               │         │
│       ▼               ▼               ▼               ▼         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    GPT-4o Analysis                        │  │
│  │  • Grounded findings only                                 │  │
│  │  • Structured JSON output                                 │  │
│  │  • Page-cited evidence                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Streamlit UI                           │  │
│  │  • Metrics dashboard                                      │  │
│  │  • Finding cards with badges                              │  │
│  │  • Filters and exports                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 RAG Pipeline

### 1. PDF Extraction (PyMuPDF)
- Page-by-page text extraction
- Metadata preservation (page numbers)
- Graceful handling of blank/corrupt pages

### 2. Chunking (LangChain)
- `RecursiveCharacterTextSplitter`
- Chunk size: 800 characters
- Overlap: 100 characters
- Page number preserved per chunk

### 3. Embedding (OpenAI)
- Model: `text-embedding-3-small`
- Efficient batch embedding
- In-memory vector storage

### 4. Retrieval (Cosine Similarity)
- Multi-query retrieval for diverse evidence
- Top-K selection based on review mode
- Similarity scores tracked for debug

### 5. Analysis (GPT-4o)
- Construction engineering expert prompt
- Strict JSON schema enforcement
- Grounded findings only (no hallucination)
- Robust parsing with fallbacks

---

## 🚀 Local Setup

### Prerequisites
- Python 3.9+
- OpenAI API key with GPT-4o access

### Installation

```bash
# Clone the repository
git clone https://github.com/ANUVIK2401/Build-Spec-AI.git
cd Build-Spec-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Run Locally

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## ☁️ Streamlit Cloud Deployment

### 1. Fork/Push to GitHub
Ensure your code is pushed to a GitHub repository.

### 2. Connect to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository
4. Set main file path: `app.py`

### 3. Configure Secrets
In Streamlit Cloud settings, add your secrets:

```toml
OPENAI_API_KEY = "your-openai-api-key"
```

### 4. Deploy
Click "Deploy" and your app will be live!

---

## 📸 Example Output

### Findings Dashboard
```
┌────────────────────────────────────────────────────────┐
│  Total: 8  │  High: 2  │  Medium: 4  │  Low: 2       │
└────────────────────────────────────────────────────────┘
```

### Sample Finding
```json
{
  "type": "compliance_gap",
  "severity": "high",
  "discipline": "electrical",
  "confidence": "high",
  "page": 12,
  "title": "Missing emergency lighting specification",
  "description": "The electrical specifications do not include emergency lighting requirements for egress paths, which is required by IBC Section 1008.",
  "evidence": "Section 4.2 Lighting Requirements specifies general illumination but omits emergency fixture placement.",
  "recommended_action": "Add emergency lighting requirements per IBC 1008 with battery backup specifications."
}
```

---

## ⚠️ Limitations

1. **PDF Quality** - Scanned PDFs without OCR will have poor text extraction
2. **Document Length** - Very large documents (500+ pages) may hit API limits
3. **Technical Scope** - Optimized for construction/engineering docs, not general PDFs
4. **False Positives** - Some findings may require human verification
5. **Language** - English documents only

---

## 🗺️ Future Roadmap

- [ ] Multi-document comparison (spec vs. drawing)
- [ ] Code compliance database integration
- [ ] Issue tracking integration (Jira, Asana)
- [ ] Team collaboration features
- [ ] Custom prompt templates
- [ ] Fine-tuned models for specific disciplines
- [ ] OCR integration for scanned documents
- [ ] Bulk document processing

---

## 💡 Founder-Facing Pitch

> **BuildSpec AI** transforms construction document review from a manual, error-prone process into an AI-assisted workflow that catches issues before they become costly field problems.
>
> **The Problem**: Engineering teams spend hundreds of hours reviewing specifications, often missing critical compliance gaps or coordination issues that lead to RFIs, change orders, and project delays.
>
> **Our Solution**: A RAG-powered document review copilot that:
> - Extracts and analyzes technical PDFs in minutes
> - Surfaces compliance gaps, contradictions, and coordination risks
> - Provides page-cited evidence for every finding
> - Exports structured reports for team collaboration
>
> **Traction**: Production-ready implementation demonstrating real RAG retrieval, structured analysis, and premium UX.
>
> **Ask**: Internship opportunity to bring this capability to Structured AI's product suite.

---

## 🔒 Git Hygiene

Before committing, ensure these files are in `.gitignore` and **not** in the public repo:

```gitignore
# Environment variables
.env

# Development prompt files
cursor_prompt.md
claude.md
quick_start_steps.md
buildspec_cursor_prompt.md
buildspec_claude.md
```

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Anuvik Thota**  
Building AI tools for construction engineering workflows.

[GitHub](https://github.com/ANUVIK2401) • [LinkedIn](https://linkedin.com/in/anuvikthota)

---

<p align="center">
  <strong>🏗️ BuildSpec AI</strong><br>
  <em>AI QA/QC Copilot for Construction Engineering Documents</em>
</p>
