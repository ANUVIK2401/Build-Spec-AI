"""
BuildSpec AI - AI QA/QC Copilot for Construction Engineering Documents

A production-ready Streamlit application that reviews technical PDFs for:
- Compliance gaps
- Contradictions  
- Missing sections
- Unclear requirements
- Coordination risks

Features:
- Real RAG pipeline with page-cited evidence
- Multi-pass review for comprehensive analysis
- Focus modes (Compliance, Coordination, Completeness)
- Priority scoring and discipline summaries
- Robust JSON parsing with graceful degradation
- Export to JSON, CSV, Markdown, TXT

Built for construction engineering QA/QC workflows.
"""

import os
import io
import json
import re
import hashlib
import tempfile
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================================
# CONFIGURATION & ENVIRONMENT
# ============================================================================

load_dotenv()

# App configuration
APP_NAME = "BuildSpec AI"
APP_VERSION = "2.0.0"
APP_TAGLINE = "AI QA/QC Copilot for Construction Engineering Documents"
APP_DESCRIPTION = "Review technical PDFs for compliance gaps, contradictions, missing sections, and coordination risks with page-cited RAG evidence."

# Document limits for safety
MAX_PAGES = 200
MAX_CHARS = 500000
MAX_CHUNKS = 500
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class IssueType(str, Enum):
    MISSING_SECTION = "missing_section"
    CONTRADICTION = "contradiction"
    COMPLIANCE_GAP = "compliance_gap"
    UNCLEAR_REQUIREMENT = "unclear_requirement"
    COORDINATION_RISK = "coordination_risk"

class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Discipline(str, Enum):
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    STRUCTURAL = "structural"
    GENERAL = "general"

class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class Priority(str, Enum):
    CRITICAL = "critical"
    IMPORTANT = "important"
    REVIEW = "review"

# Label mappings for display
ISSUE_TYPE_LABELS = {
    IssueType.MISSING_SECTION: "Missing Section",
    IssueType.CONTRADICTION: "Contradiction",
    IssueType.COMPLIANCE_GAP: "Compliance Gap",
    IssueType.UNCLEAR_REQUIREMENT: "Unclear Requirement",
    IssueType.COORDINATION_RISK: "Coordination Risk"
}

SEVERITY_COLORS = {
    Severity.HIGH: "#ef4444",
    Severity.MEDIUM: "#f59e0b",
    Severity.LOW: "#10b981"
}

PRIORITY_COLORS = {
    Priority.CRITICAL: "#dc2626",
    Priority.IMPORTANT: "#ea580c",
    Priority.REVIEW: "#0891b2"
}

# Review mode configurations
REVIEW_MODES = {
    'Quick Review': {
        'description': 'Fast screening with key findings. Best for initial assessment.',
        'top_k': 8,
        'max_issues': 6,
        'passes': 1
    },
    'Standard Review': {
        'description': 'Balanced analysis with comprehensive coverage. Recommended for most documents.',
        'top_k': 15,
        'max_issues': 12,
        'passes': 2
    },
    'Deep Review': {
        'description': 'Thorough multi-pass analysis for critical documents requiring detailed QA/QC.',
        'top_k': 25,
        'max_issues': 20,
        'passes': 3
    }
}

# Focus mode configurations
FOCUS_MODES = {
    'Full Review': {
        'description': 'Comprehensive review across all categories',
        'emphasis': None,
        'icon': '📋'
    },
    'Compliance Focus': {
        'description': 'Prioritize code compliance and regulatory gaps',
        'emphasis': 'compliance',
        'icon': '⚖️'
    },
    'Coordination Focus': {
        'description': 'Focus on cross-discipline coordination risks',
        'emphasis': 'coordination',
        'icon': '🔄'
    },
    'Completeness Focus': {
        'description': 'Identify missing sections and incomplete specs',
        'emphasis': 'completeness',
        'icon': '📑'
    }
}

# Review pass configurations for multi-pass analysis
REVIEW_PASSES = {
    'completeness': {
        'name': 'Completeness Check',
        'query': 'What required sections, specifications, or details appear to be missing or incomplete?',
        'focus_types': [IssueType.MISSING_SECTION, IssueType.UNCLEAR_REQUIREMENT]
    },
    'contradictions': {
        'name': 'Contradiction Analysis', 
        'query': 'Are there any conflicting requirements, contradictions, or ambiguous specifications?',
        'focus_types': [IssueType.CONTRADICTION, IssueType.UNCLEAR_REQUIREMENT]
    },
    'compliance': {
        'name': 'Compliance & Coordination Review',
        'query': 'What compliance gaps or coordination risks exist between disciplines?',
        'focus_types': [IssueType.COMPLIANCE_GAP, IssueType.COORDINATION_RISK]
    }
}

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PageData:
    """Extracted page data with metadata."""
    page_number: int
    text: str
    char_count: int
    preview: str = ""
    sections: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.preview:
            self.preview = self.text[:200] + "..." if len(self.text) > 200 else self.text

@dataclass
class ChunkData:
    """Document chunk with metadata."""
    chunk_id: int
    page_number: int
    text: str
    preview: str = ""
    section: str = ""
    similarity: float = 0.0
    
    def __post_init__(self):
        if not self.preview:
            self.preview = self.text[:150] + "..." if len(self.text) > 150 else self.text

@dataclass
class Finding:
    """Normalized finding with all required fields."""
    id: str
    type: str
    severity: str
    discipline: str
    confidence: str
    priority: str
    page: int
    title: str
    description: str
    evidence: str
    recommended_action: str
    source_chunks: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class DocumentSnapshot:
    """Document intelligence summary."""
    filename: str
    file_size: str
    total_pages: int
    extracted_pages: int
    total_chars: int
    total_chunks: int
    detected_sections: List[str]
    review_mode: str
    focus_mode: str

@dataclass
class ReviewSummary:
    """Executive summary of review results."""
    document: DocumentSnapshot
    total_findings: int
    high_severity: int
    medium_severity: int
    low_severity: int
    critical_priority: int
    discipline_counts: Dict[str, int]
    top_issues: List[str]
    top_actions: List[str]
    generated_at: str

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

def inject_custom_css():
    """Inject premium dark theme CSS."""
    st.markdown("""
<style>
    /* CSS Variables */
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --secondary: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --critical: #dc2626;
        --background: #0f172a;
        --surface: #1e293b;
        --surface-light: #334155;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }
    
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
    }
    
    .hero-subtitle {
        font-size: 1.125rem;
        color: #94a3b8;
        margin-bottom: 0.75rem;
    }
    
    .hero-description {
        font-size: 0.9rem;
        color: #64748b;
        max-width: 700px;
        margin: 0 auto 1rem auto;
    }
    
    /* Feature chips */
    .chip-container {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.75rem;
        background: #334155;
        border: 1px solid #475569;
        border-radius: 9999px;
        font-size: 0.75rem;
        color: #e2e8f0;
    }
    
    /* Cards */
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Document snapshot */
    .snapshot-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 0.75rem;
    }
    
    .snapshot-item {
        background: #0f172a;
        border-radius: 8px;
        padding: 0.75rem;
        text-align: center;
    }
    
    .snapshot-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .snapshot-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    
    /* Metrics row */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    .metric-high { border-left: 4px solid #ef4444; }
    .metric-medium { border-left: 4px solid #f59e0b; }
    .metric-low { border-left: 4px solid #10b981; }
    .metric-total { border-left: 4px solid #6366f1; }
    .metric-critical { border-left: 4px solid #dc2626; }
    
    /* Finding cards */
    .finding-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #6366f1;
    }
    
    .finding-card.high { border-left-color: #ef4444; }
    .finding-card.medium { border-left-color: #f59e0b; }
    .finding-card.low { border-left-color: #10b981; }
    .finding-card.critical-priority { border-left-color: #dc2626; }
    
    .finding-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.5rem;
    }
    
    .finding-title {
        font-size: 1rem;
        font-weight: 600;
        color: #f8fafc;
        flex: 1;
    }
    
    .finding-id {
        font-size: 0.7rem;
        color: #64748b;
        font-family: monospace;
    }
    
    .finding-badges {
        display: flex;
        gap: 0.35rem;
        flex-wrap: wrap;
        margin-bottom: 0.75rem;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }
    
    .badge-type { background: #312e81; color: #a5b4fc; }
    .badge-severity-high { background: #7f1d1d; color: #fca5a5; }
    .badge-severity-medium { background: #78350f; color: #fcd34d; }
    .badge-severity-low { background: #064e3b; color: #6ee7b7; }
    .badge-discipline { background: #1e3a5f; color: #7dd3fc; }
    .badge-confidence { background: #3f3f46; color: #d4d4d8; }
    .badge-page { background: #4a044e; color: #f0abfc; }
    .badge-priority-critical { background: #7f1d1d; color: #fca5a5; }
    .badge-priority-important { background: #7c2d12; color: #fdba74; }
    .badge-priority-review { background: #164e63; color: #67e8f9; }
    
    .finding-description {
        color: #cbd5e1;
        margin-bottom: 0.75rem;
        line-height: 1.5;
        font-size: 0.9rem;
    }
    
    .finding-section {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    
    .finding-section-label {
        font-size: 0.65rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    
    .finding-evidence {
        font-size: 0.85rem;
        color: #94a3b8;
        font-style: italic;
    }
    
    .finding-action {
        background: #0d3320;
        border: 1px solid #166534;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.85rem;
        color: #86efac;
    }
    
    .finding-action-label {
        font-size: 0.65rem;
        color: #4ade80;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    
    /* Summary report */
    .summary-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .summary-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .summary-item {
        background: #334155;
        border-radius: 8px;
        padding: 0.75rem;
    }
    
    .summary-item-label {
        font-size: 0.65rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .summary-item-value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f8fafc;
        margin-top: 0.15rem;
    }
    
    /* Discipline breakdown */
    .discipline-bar {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.75rem;
    }
    
    .discipline-item {
        background: #334155;
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        font-size: 0.8rem;
    }
    
    .discipline-count {
        font-weight: 700;
        color: #f8fafc;
    }
    
    .discipline-name {
        color: #94a3b8;
        margin-left: 0.25rem;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 2rem;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
    }
    
    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 0.75rem;
    }
    
    .empty-state-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }
    
    .empty-state-description {
        color: #94a3b8;
        max-width: 450px;
        margin: 0 auto;
        font-size: 0.9rem;
    }
    
    /* Progress pipeline */
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0;
    }
    
    .pipeline-icon {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        background: #334155;
        color: #94a3b8;
    }
    
    .pipeline-icon.active {
        background: #6366f1;
        color: white;
    }
    
    .pipeline-icon.complete {
        background: #10b981;
        color: white;
    }
    
    .pipeline-text {
        color: #94a3b8;
        font-size: 0.85rem;
    }
    
    .pipeline-text.active {
        color: #f8fafc;
        font-weight: 500;
    }
    
    /* Evidence panel */
    .evidence-chunk {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    
    .evidence-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .evidence-page {
        font-size: 0.75rem;
        color: #a78bfa;
        font-weight: 600;
    }
    
    .evidence-similarity {
        font-size: 0.7rem;
        color: #64748b;
    }
    
    .evidence-text {
        font-size: 0.8rem;
        color: #94a3b8;
        line-height: 1.4;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.25rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #334155;
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #64748b;
        font-size: 0.8rem;
        border-top: 1px solid #334155;
        margin-top: 2rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title { font-size: 1.75rem; }
        .snapshot-grid { grid-template-columns: repeat(2, 1fr); }
        .summary-grid { grid-template-columns: repeat(2, 1fr); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def init_session_state():
    """Initialize all session state variables with defaults."""
    defaults = {
        'findings': [],
        'chunks': [],
        'pages': [],
        'sections': [],
        'file_info': None,
        'document_snapshot': None,
        'analysis_complete': False,
        'analysis_in_progress': False,
        'raw_model_outputs': [],
        'retrieved_chunks': [],
        'chunk_embeddings': None,
        'review_mode': 'Standard Review',
        'focus_mode': 'Full Review',
        'debug_mode': False,
        'current_file': None,
        'current_file_hash': None,
        'analysis_steps': [],
        'error_message': None,
        'grouping_mode': 'severity'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_file_hash(file_content: bytes) -> str:
    """Generate stable hash for file content."""
    return hashlib.md5(file_content).hexdigest()[:12]

def generate_finding_id() -> str:
    """Generate unique finding ID."""
    return f"F-{uuid.uuid4().hex[:8].upper()}"

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r'\s+', ' ', text).strip()

# ============================================================================
# OPENAI CLIENT
# ============================================================================

@st.cache_resource
def get_openai_client() -> Optional[OpenAI]:
    """Get cached OpenAI client."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        try:
            api_key = st.secrets.get('OPENAI_API_KEY')
        except:
            pass
    
    if not api_key:
        return None
    
    return OpenAI(api_key=api_key)

# ============================================================================
# SECTION DETECTION
# ============================================================================

def detect_sections(text: str) -> List[str]:
    """
    Detect likely section headers from document text.
    Uses heuristics for common technical document patterns.
    """
    sections = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 3 or len(line) > 100:
            continue
        
        # Pattern 1: All caps lines (likely headers)
        if line.isupper() and len(line) > 5:
            sections.append(line)
            continue
        
        # Pattern 2: Numbered sections (1.0, 1.1, SECTION 1, etc.)
        if re.match(r'^(?:SECTION\s+)?(\d+\.?\d*)\s+[A-Z]', line):
            sections.append(line)
            continue
        
        # Pattern 3: Article/Part headers
        if re.match(r'^(?:ARTICLE|PART|DIVISION|CHAPTER)\s+[IVXLCDM\d]+', line, re.IGNORECASE):
            sections.append(line)
            continue
        
        # Pattern 4: Lines ending with colon that start with caps
        if line.endswith(':') and line[0].isupper() and len(line) > 10:
            sections.append(line.rstrip(':'))
            continue
    
    # Deduplicate and limit
    seen = set()
    unique_sections = []
    for s in sections:
        s_clean = s.strip()[:60]
        if s_clean.lower() not in seen:
            seen.add(s_clean.lower())
            unique_sections.append(s_clean)
    
    return unique_sections[:30]  # Limit to reasonable number

# ============================================================================
# PDF EXTRACTION
# ============================================================================

def extract_pdf_pages(pdf_file) -> Tuple[List[PageData], List[str]]:
    """
    Extract text from PDF file page by page using PyMuPDF.
    Returns pages and detected sections.
    """
    pages = []
    all_sections = []
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        doc = fitz.open(tmp_path)
        total_pages = len(doc)
        
        # Check page limit
        if total_pages > MAX_PAGES:
            st.warning(f"Document has {total_pages} pages. Processing first {MAX_PAGES} pages only.")
            total_pages = MAX_PAGES
        
        for page_num in range(total_pages):
            try:
                page = doc[page_num]
                text = page.get_text("text")
                
                # Clean text while preserving structure for section detection
                raw_text = text
                text_clean = normalize_whitespace(text)
                
                if len(text_clean) < 30:
                    continue
                
                # Detect sections from raw text (preserves line breaks)
                page_sections = detect_sections(raw_text)
                all_sections.extend(page_sections)
                
                page_data = PageData(
                    page_number=page_num + 1,
                    text=text_clean,
                    char_count=len(text_clean),
                    sections=page_sections
                )
                pages.append(page_data)
                
            except Exception as e:
                if st.session_state.debug_mode:
                    st.warning(f"Could not extract page {page_num + 1}: {str(e)}")
                continue
        
        doc.close()
        os.unlink(tmp_path)
        
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return [], []
    
    # Deduplicate sections
    seen = set()
    unique_sections = []
    for s in all_sections:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique_sections.append(s)
    
    return pages, unique_sections[:25]

# ============================================================================
# CHUNKING
# ============================================================================

def chunk_pages(pages: List[PageData]) -> List[ChunkData]:
    """
    Chunk extracted pages while preserving page metadata.
    Uses RecursiveCharacterTextSplitter for stable chunking.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
    )
    
    chunks = []
    chunk_idx = 0
    
    for page in pages:
        if chunk_idx >= MAX_CHUNKS:
            break
            
        page_chunks = splitter.split_text(page.text)
        
        # Get current section context from page
        current_section = page.sections[0] if page.sections else ""
        
        for chunk_text in page_chunks:
            if chunk_idx >= MAX_CHUNKS:
                break
            if len(chunk_text.strip()) < 30:
                continue
            
            chunk = ChunkData(
                chunk_id=chunk_idx,
                page_number=page.page_number,
                text=chunk_text.strip(),
                section=current_section
            )
            chunks.append(chunk)
            chunk_idx += 1
    
    return chunks

# ============================================================================
# EMBEDDINGS & RETRIEVAL
# ============================================================================

def get_embeddings_batch(texts: List[str], client: OpenAI, batch_size: int = 100) -> np.ndarray:
    """Generate embeddings in batches for efficiency."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return np.array([])
    
    return np.array(all_embeddings)

def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector sets."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return np.dot(a_norm, b_norm.T)

def retrieve_chunks(
    query: str,
    chunks: List[ChunkData],
    chunk_embeddings: np.ndarray,
    client: OpenAI,
    top_k: int = 10
) -> List[ChunkData]:
    """Retrieve most relevant chunks for a query."""
    if not chunks or chunk_embeddings.size == 0:
        return []
    
    query_embedding = get_embeddings_batch([query], client)
    if query_embedding.size == 0:
        return []
    
    similarities = cosine_similarity_matrix(query_embedding, chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    retrieved = []
    for idx in top_indices:
        chunk = ChunkData(
            chunk_id=chunks[idx].chunk_id,
            page_number=chunks[idx].page_number,
            text=chunks[idx].text,
            preview=chunks[idx].preview,
            section=chunks[idx].section,
            similarity=float(similarities[idx])
        )
        retrieved.append(chunk)
    
    return retrieved

def retrieve_diverse_evidence(
    chunks: List[ChunkData],
    chunk_embeddings: np.ndarray,
    client: OpenAI,
    review_mode: str,
    focus_mode: str
) -> List[ChunkData]:
    """
    Retrieve diverse evidence using multiple queries.
    Adapts queries based on focus mode.
    """
    mode_config = REVIEW_MODES[review_mode]
    top_k = mode_config['top_k']
    
    # Base queries for comprehensive retrieval
    base_queries = [
        "What are the key specifications, requirements, and standards mentioned?",
        "What safety requirements and compliance specifications are defined?",
        "What are the mechanical, electrical, and structural requirements?",
    ]
    
    # Add focus-specific queries
    focus_config = FOCUS_MODES[focus_mode]
    if focus_config['emphasis'] == 'compliance':
        base_queries.extend([
            "What building codes, standards, and regulatory requirements apply?",
            "What safety and compliance specifications are mentioned?",
        ])
    elif focus_config['emphasis'] == 'coordination':
        base_queries.extend([
            "What coordination requirements exist between disciplines?",
            "What interfaces and dependencies between systems are specified?",
        ])
    elif focus_config['emphasis'] == 'completeness':
        base_queries.extend([
            "What sections or specifications appear to be referenced but not included?",
            "What scope items and deliverables are defined?",
        ])
    
    all_retrieved = []
    seen_ids = set()
    
    chunks_per_query = max(3, top_k // len(base_queries) + 1)
    
    for query in base_queries:
        retrieved = retrieve_chunks(
            query, chunks, chunk_embeddings, client, top_k=chunks_per_query
        )
        for chunk in retrieved:
            if chunk.chunk_id not in seen_ids:
                all_retrieved.append(chunk)
                seen_ids.add(chunk.chunk_id)
    
    # Sort by similarity and limit
    all_retrieved.sort(key=lambda x: x.similarity, reverse=True)
    return all_retrieved[:top_k]

# ============================================================================
# PROMPT CONSTRUCTION
# ============================================================================

def build_review_prompt(
    retrieved_chunks: List[ChunkData],
    review_mode: str,
    focus_mode: str,
    pass_type: str = None
) -> str:
    """Build analysis prompt with retrieved evidence."""
    mode_config = REVIEW_MODES[review_mode]
    max_issues = mode_config['max_issues']
    
    # Adjust max issues for multi-pass
    if pass_type:
        max_issues = max(3, max_issues // 3)
    
    # Format evidence
    evidence_parts = []
    for c in retrieved_chunks:
        section_info = f" (Section: {c.section})" if c.section else ""
        evidence_parts.append(f"[Page {c.page_number}{section_info}]:\n{c.text}")
    
    evidence_text = "\n\n---\n\n".join(evidence_parts)
    
    # Focus mode instructions
    focus_instructions = ""
    focus_config = FOCUS_MODES[focus_mode]
    if focus_config['emphasis'] == 'compliance':
        focus_instructions = """
## Focus Priority: COMPLIANCE
Prioritize identifying:
- Missing code or standard references
- Incomplete regulatory compliance
- Safety specification gaps
- Permit and approval requirements
"""
    elif focus_config['emphasis'] == 'coordination':
        focus_instructions = """
## Focus Priority: COORDINATION
Prioritize identifying:
- Cross-discipline conflicts
- Interface and handoff issues
- Spatial or routing conflicts
- Sequencing dependencies
"""
    elif focus_config['emphasis'] == 'completeness':
        focus_instructions = """
## Focus Priority: COMPLETENESS
Prioritize identifying:
- Missing sections or specifications
- Referenced but absent documents
- Incomplete scope definitions
- Undefined requirements
"""
    
    # Pass-specific instructions
    pass_instructions = ""
    if pass_type == 'completeness':
        pass_instructions = """
## This Review Pass: COMPLETENESS CHECK
Focus specifically on:
- Missing required sections
- Incomplete specifications
- Undefined terms or requirements
- Referenced but absent content
"""
    elif pass_type == 'contradictions':
        pass_instructions = """
## This Review Pass: CONTRADICTION ANALYSIS
Focus specifically on:
- Conflicting requirements
- Inconsistent specifications
- Ambiguous or vague language
- Contradictory statements
"""
    elif pass_type == 'compliance':
        pass_instructions = """
## This Review Pass: COMPLIANCE & COORDINATION
Focus specifically on:
- Code and standard gaps
- Regulatory compliance issues
- Cross-discipline coordination risks
- Safety specification gaps
"""
    
    prompt = f"""You are an expert construction engineering document QA/QC reviewer specializing in technical specification analysis.

## Context
You are reviewing extracted sections from a construction engineering document to identify potential quality issues that could impact project execution.
{focus_instructions}
{pass_instructions}

## Evidence Retrieved from Document
{evidence_text}

## Issue Types
- missing_section: Required sections, specifications, or details that appear absent
- contradiction: Conflicting requirements or inconsistent specifications
- compliance_gap: Areas not meeting industry standards, codes, or best practices
- unclear_requirement: Vague, ambiguous, or incomplete specifications
- coordination_risk: Potential conflicts between disciplines or systems

## Required Output Format
Return a JSON array. Each finding MUST include ALL these fields:
- type: One of [missing_section, contradiction, compliance_gap, unclear_requirement, coordination_risk]
- severity: One of [high, medium, low]
- discipline: One of [mechanical, electrical, structural, general]
- confidence: One of [high, medium, low]
- page: Page number from evidence (integer)
- title: Clear, specific issue title (50-80 characters, sounds like a real review comment)
- description: Detailed explanation (2-3 sentences, specific to the document)
- evidence: Direct quote or observation from the retrieved text supporting this finding
- recommended_action: Specific, actionable resolution step

## Quality Standards
1. ONLY flag issues clearly supported by the evidence
2. Use specific page citations
3. Write titles like professional review comments: "Missing fire alarm acceptance criteria" not "Issue with fire alarms"
4. Descriptions must explain WHY this is a problem
5. Evidence must quote or reference specific text
6. Actions must be concrete and achievable
7. Return at most {max_issues} findings
8. Return [] if no issues are clearly evident
9. Order by severity (high first)

## Response
Return ONLY a valid JSON array. No markdown fencing, no explanations.

Example:
[{{"type": "compliance_gap", "severity": "high", "discipline": "electrical", "confidence": "high", "page": 12, "title": "Missing emergency lighting coverage requirements", "description": "The electrical specifications do not define emergency lighting coverage for egress paths. This omission could result in code violations and safety issues during emergency evacuations.", "evidence": "Section 4.2 specifies general lighting but makes no mention of emergency fixtures or backup power.", "recommended_action": "Add emergency lighting requirements per IBC Section 1008 with battery backup specifications for all egress routes."}}]
"""
    return prompt

# ============================================================================
# JSON PARSING & NORMALIZATION
# ============================================================================

def repair_json(text: str) -> str:
    """Attempt to repair common JSON issues."""
    # Remove markdown fences
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text)
    
    # Try to extract JSON array
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        text = match.group()
    
    # Fix common issues
    text = re.sub(r',\s*}', '}', text)  # Trailing commas in objects
    text = re.sub(r',\s*]', ']', text)  # Trailing commas in arrays
    text = re.sub(r'"\s*\n\s*"', '", "', text)  # Missing commas between strings
    
    return text

def normalize_enum(value: str, valid_values: set, default: str) -> str:
    """Normalize a value to a valid enum."""
    if not value:
        return default
    normalized = str(value).lower().strip().replace(' ', '_').replace('-', '_')
    return normalized if normalized in valid_values else default

def parse_model_output(raw_output: str) -> List[Dict]:
    """Parse and validate model JSON output with robust error handling."""
    if not raw_output:
        return []
    
    # Repair JSON
    cleaned = repair_json(raw_output)
    
    # Parse
    try:
        findings = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try more aggressive extraction
        try:
            # Find anything that looks like a JSON array of objects
            matches = re.findall(r'\{[^{}]+\}', cleaned)
            if matches:
                findings = [json.loads(m) for m in matches]
            else:
                return []
        except:
            return []
    
    if not isinstance(findings, list):
        findings = [findings] if isinstance(findings, dict) else []
    
    # Valid enum values
    valid_types = {e.value for e in IssueType}
    valid_severities = {e.value for e in Severity}
    valid_disciplines = {e.value for e in Discipline}
    valid_confidence = {e.value for e in Confidence}
    
    validated = []
    seen_titles = set()
    
    for f in findings:
        if not isinstance(f, dict):
            continue
        
        # Normalize enums
        f_type = normalize_enum(f.get('type'), valid_types, IssueType.UNCLEAR_REQUIREMENT.value)
        severity = normalize_enum(f.get('severity'), valid_severities, Severity.MEDIUM.value)
        discipline = normalize_enum(f.get('discipline'), valid_disciplines, Discipline.GENERAL.value)
        confidence = normalize_enum(f.get('confidence'), valid_confidence, Confidence.MEDIUM.value)
        
        # Get title and deduplicate
        title = str(f.get('title', 'Untitled Finding'))[:100].strip()
        if not title or title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        
        # Validate page
        try:
            page = max(1, int(f.get('page', 1)))
        except:
            page = 1
        
        validated.append({
            'type': f_type,
            'severity': severity,
            'discipline': discipline,
            'confidence': confidence,
            'page': page,
            'title': title,
            'description': str(f.get('description', ''))[:600].strip(),
            'evidence': str(f.get('evidence', ''))[:400].strip(),
            'recommended_action': str(f.get('recommended_action', ''))[:400].strip()
        })
    
    return validated

# ============================================================================
# PRIORITY SCORING
# ============================================================================

def calculate_priority(finding: Dict) -> str:
    """
    Calculate priority score based on severity, confidence, and type.
    Returns: critical, important, or review
    """
    score = 0
    
    # Severity weight (0-3)
    severity_weights = {'high': 3, 'medium': 2, 'low': 1}
    score += severity_weights.get(finding['severity'], 2)
    
    # Confidence weight (0-2)
    confidence_weights = {'high': 2, 'medium': 1, 'low': 0}
    score += confidence_weights.get(finding['confidence'], 1)
    
    # Type weight (0-2)
    type_weights = {
        'compliance_gap': 2,
        'coordination_risk': 2,
        'contradiction': 1.5,
        'missing_section': 1,
        'unclear_requirement': 0.5
    }
    score += type_weights.get(finding['type'], 1)
    
    # Calculate priority
    if score >= 6:
        return Priority.CRITICAL.value
    elif score >= 4:
        return Priority.IMPORTANT.value
    else:
        return Priority.REVIEW.value

def normalize_findings(raw_findings: List[Dict]) -> List[Finding]:
    """Normalize raw findings into Finding objects with IDs and priorities."""
    normalized = []
    
    for f in raw_findings:
        priority = calculate_priority(f)
        
        finding = Finding(
            id=generate_finding_id(),
            type=f['type'],
            severity=f['severity'],
            discipline=f['discipline'],
            confidence=f['confidence'],
            priority=priority,
            page=f['page'],
            title=f['title'],
            description=f['description'],
            evidence=f['evidence'],
            recommended_action=f['recommended_action'],
            source_chunks=f.get('source_chunks', [])
        )
        normalized.append(finding)
    
    return normalized

def deduplicate_findings(findings: List[Finding]) -> List[Finding]:
    """Remove near-duplicate findings based on title similarity."""
    if len(findings) <= 1:
        return findings
    
    unique = []
    seen_titles = set()
    
    for f in findings:
        # Simple title normalization for comparison
        normalized_title = re.sub(r'[^a-z0-9]', '', f.title.lower())
        
        # Check for similar titles
        is_duplicate = False
        for seen in seen_titles:
            if len(normalized_title) > 10 and len(seen) > 10:
                # Check if one contains most of the other
                if normalized_title in seen or seen in normalized_title:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            unique.append(f)
            seen_titles.add(normalized_title)
    
    return unique

def sort_findings(findings: List[Finding], sort_by: str = 'severity') -> List[Finding]:
    """Sort findings by specified criteria."""
    if sort_by == 'severity':
        order = {Severity.HIGH.value: 0, Severity.MEDIUM.value: 1, Severity.LOW.value: 2}
        return sorted(findings, key=lambda f: (order.get(f.severity, 1), f.page))
    elif sort_by == 'priority':
        order = {Priority.CRITICAL.value: 0, Priority.IMPORTANT.value: 1, Priority.REVIEW.value: 2}
        return sorted(findings, key=lambda f: (order.get(f.priority, 1), f.page))
    elif sort_by == 'page':
        return sorted(findings, key=lambda f: f.page)
    elif sort_by == 'discipline':
        return sorted(findings, key=lambda f: (f.discipline, f.page))
    elif sort_by == 'type':
        return sorted(findings, key=lambda f: (f.type, f.page))
    return findings

# ============================================================================
# MULTI-PASS ANALYSIS
# ============================================================================

def run_analysis_pass(
    chunks: List[ChunkData],
    chunk_embeddings: np.ndarray,
    client: OpenAI,
    review_mode: str,
    focus_mode: str,
    pass_type: str,
    progress_callback=None
) -> Tuple[List[Dict], List[ChunkData], str]:
    """Run a single analysis pass."""
    mode_config = REVIEW_MODES[review_mode]
    pass_config = REVIEW_PASSES.get(pass_type, {})
    
    # Retrieve evidence with pass-specific query if available
    query = pass_config.get('query', '')
    
    if query:
        retrieved = retrieve_chunks(
            query, chunks, chunk_embeddings, client,
            top_k=mode_config['top_k'] // 2
        )
    else:
        retrieved = retrieve_diverse_evidence(
            chunks, chunk_embeddings, client, review_mode, focus_mode
        )
    
    if not retrieved:
        return [], [], "No relevant chunks retrieved"
    
    # Build and execute prompt
    prompt = build_review_prompt(retrieved, review_mode, focus_mode, pass_type)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert construction engineering QA/QC reviewer. Return only valid JSON arrays with no markdown formatting."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=4000
        )
        raw_output = response.choices[0].message.content
    except Exception as e:
        return [], retrieved, f"API Error: {str(e)}"
    
    findings = parse_model_output(raw_output)
    return findings, retrieved, raw_output

def run_multi_pass_analysis(
    chunks: List[ChunkData],
    chunk_embeddings: np.ndarray,
    client: OpenAI,
    review_mode: str,
    focus_mode: str,
    progress_container
) -> Tuple[List[Finding], List[ChunkData], List[str]]:
    """Run multi-pass analysis pipeline."""
    mode_config = REVIEW_MODES[review_mode]
    num_passes = mode_config['passes']
    
    all_findings = []
    all_retrieved = []
    all_raw_outputs = []
    seen_chunk_ids = set()
    
    passes_to_run = list(REVIEW_PASSES.keys())[:num_passes]
    
    for i, pass_type in enumerate(passes_to_run):
        pass_config = REVIEW_PASSES[pass_type]
        
        with progress_container:
            st.markdown(f"**Pass {i+1}/{num_passes}**: {pass_config['name']}")
        
        findings, retrieved, raw_output = run_analysis_pass(
            chunks, chunk_embeddings, client,
            review_mode, focus_mode, pass_type
        )
        
        all_findings.extend(findings)
        all_raw_outputs.append(raw_output)
        
        # Collect unique retrieved chunks
        for chunk in retrieved:
            if chunk.chunk_id not in seen_chunk_ids:
                all_retrieved.append(chunk)
                seen_chunk_ids.add(chunk.chunk_id)
    
    # Normalize and deduplicate
    normalized = normalize_findings(all_findings)
    deduped = deduplicate_findings(normalized)
    
    # Limit to max issues
    max_issues = mode_config['max_issues']
    final_findings = sort_findings(deduped, 'priority')[:max_issues]
    
    return final_findings, all_retrieved, all_raw_outputs

# ============================================================================
# DOCUMENT ANALYSIS ORCHESTRATION
# ============================================================================

def analyze_document(
    chunks: List[ChunkData],
    client: OpenAI,
    review_mode: str,
    focus_mode: str,
    progress_container
) -> Tuple[List[Finding], List[ChunkData], List[str]]:
    """Full document analysis pipeline."""
    if not chunks:
        return [], [], []
    
    st.session_state.analysis_steps = []
    
    # Step 1: Generate embeddings
    with progress_container:
        st.markdown("**Step 1/4**: Generating embeddings...")
    
    chunk_texts = [c.text for c in chunks]
    chunk_embeddings = get_embeddings_batch(chunk_texts, client)
    
    if chunk_embeddings.size == 0:
        st.error("Failed to generate embeddings")
        return [], [], []
    
    st.session_state.chunk_embeddings = chunk_embeddings
    
    # Step 2: Retrieve evidence
    with progress_container:
        st.markdown("**Step 2/4**: Retrieving relevant evidence...")
    
    # Step 3: Multi-pass analysis
    with progress_container:
        st.markdown("**Step 3/4**: Running multi-pass analysis...")
    
    findings, retrieved, raw_outputs = run_multi_pass_analysis(
        chunks, chunk_embeddings, client,
        review_mode, focus_mode, progress_container
    )
    
    # Step 4: Finalize
    with progress_container:
        st.markdown("**Step 4/4**: Finalizing results...")
    
    return findings, retrieved, raw_outputs

# ============================================================================
# SUMMARY GENERATION
# ============================================================================

def generate_review_summary(
    findings: List[Finding],
    document: DocumentSnapshot
) -> ReviewSummary:
    """Generate executive summary of review results."""
    
    # Count by severity
    high_count = len([f for f in findings if f.severity == Severity.HIGH.value])
    medium_count = len([f for f in findings if f.severity == Severity.MEDIUM.value])
    low_count = len([f for f in findings if f.severity == Severity.LOW.value])
    
    # Count by priority
    critical_count = len([f for f in findings if f.priority == Priority.CRITICAL.value])
    
    # Count by discipline
    discipline_counts = {}
    for f in findings:
        discipline_counts[f.discipline] = discipline_counts.get(f.discipline, 0) + 1
    
    # Top issues and actions
    sorted_findings = sort_findings(findings, 'priority')
    top_issues = [f.title for f in sorted_findings[:5]]
    top_actions = [f.recommended_action for f in sorted_findings[:5] if f.recommended_action]
    
    return ReviewSummary(
        document=document,
        total_findings=len(findings),
        high_severity=high_count,
        medium_severity=medium_count,
        low_severity=low_count,
        critical_priority=critical_count,
        discipline_counts=discipline_counts,
        top_issues=top_issues,
        top_actions=top_actions,
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_json(findings: List[Finding], summary: ReviewSummary) -> str:
    """Export findings as JSON."""
    export_data = {
        'metadata': {
            'app': APP_NAME,
            'version': APP_VERSION,
            'generated_at': summary.generated_at,
            'document': summary.document.filename,
            'review_mode': summary.document.review_mode,
            'focus_mode': summary.document.focus_mode
        },
        'summary': {
            'total_findings': summary.total_findings,
            'high_severity': summary.high_severity,
            'medium_severity': summary.medium_severity,
            'low_severity': summary.low_severity,
            'critical_priority': summary.critical_priority,
            'discipline_counts': summary.discipline_counts
        },
        'findings': [f.to_dict() for f in findings]
    }
    return json.dumps(export_data, indent=2)

def export_csv(findings: List[Finding]) -> str:
    """Export findings as CSV."""
    if not findings:
        return "No findings to export"
    
    rows = []
    for f in findings:
        rows.append({
            'ID': f.id,
            'Title': f.title,
            'Type': f.type.replace('_', ' ').title(),
            'Severity': f.severity.upper(),
            'Priority': f.priority.upper(),
            'Discipline': f.discipline.title(),
            'Confidence': f.confidence.title(),
            'Page': f.page,
            'Description': f.description,
            'Evidence': f.evidence,
            'Recommended Action': f.recommended_action
        })
    
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

def export_markdown(findings: List[Finding], summary: ReviewSummary) -> str:
    """Export findings as Markdown report."""
    md = f"""# {APP_NAME} - Document Review Report

## Document Information
| Field | Value |
|-------|-------|
| Filename | {summary.document.filename} |
| Pages Analyzed | {summary.document.extracted_pages} |
| Review Mode | {summary.document.review_mode} |
| Focus Mode | {summary.document.focus_mode} |
| Generated | {summary.generated_at} |

## Executive Summary

| Metric | Count |
|--------|-------|
| Total Findings | {summary.total_findings} |
| High Severity | {summary.high_severity} |
| Medium Severity | {summary.medium_severity} |
| Low Severity | {summary.low_severity} |
| Critical Priority | {summary.critical_priority} |

### Discipline Breakdown
"""
    for disc, count in summary.discipline_counts.items():
        md += f"- **{disc.title()}**: {count} findings\n"
    
    md += "\n### Top Critical Issues\n"
    for issue in summary.top_issues[:5]:
        md += f"1. {issue}\n"
    
    md += "\n### Top Recommended Actions\n"
    for action in summary.top_actions[:5]:
        md += f"1. {action}\n"
    
    md += "\n---\n\n## Detailed Findings\n\n"
    
    for i, f in enumerate(findings, 1):
        md += f"""### {i}. {f.title}

| Attribute | Value |
|-----------|-------|
| ID | `{f.id}` |
| Type | {f.type.replace('_', ' ').title()} |
| Severity | **{f.severity.upper()}** |
| Priority | {f.priority.upper()} |
| Discipline | {f.discipline.title()} |
| Confidence | {f.confidence.title()} |
| Page | {f.page} |

**Description:** {f.description}

**Evidence:** _{f.evidence}_

**Recommended Action:** {f.recommended_action}

---

"""
    
    md += f"""
## About This Report

This report was generated by **{APP_NAME}** v{APP_VERSION}, an AI-powered QA/QC copilot for construction engineering documents.

The analysis uses a RAG (Retrieval-Augmented Generation) pipeline to ground all findings in evidence from the source document. All page citations refer to the original document.

**Note:** This automated review should be verified by qualified engineering professionals before taking action.
"""
    return md

def export_txt(findings: List[Finding], summary: ReviewSummary) -> str:
    """Export findings as plain text."""
    txt = f"""{APP_NAME.upper()} - DOCUMENT REVIEW REPORT
{'=' * 50}

DOCUMENT: {summary.document.filename}
REVIEW MODE: {summary.document.review_mode}
FOCUS MODE: {summary.document.focus_mode}
GENERATED: {summary.generated_at}

SUMMARY
{'-' * 50}
Total Findings: {summary.total_findings}
High Severity: {summary.high_severity}
Medium Severity: {summary.medium_severity}
Low Severity: {summary.low_severity}
Critical Priority: {summary.critical_priority}

FINDINGS
{'=' * 50}

"""
    for i, f in enumerate(findings, 1):
        txt += f"""
[{i}] {f.title}
{'-' * 40}
ID: {f.id}
Type: {f.type.replace('_', ' ').title()}
Severity: {f.severity.upper()} | Priority: {f.priority.upper()}
Discipline: {f.discipline.title()} | Confidence: {f.confidence.title()}
Page: {f.page}

Description:
{f.description}

Evidence:
"{f.evidence}"

Recommended Action:
{f.recommended_action}

"""
    return txt

# ============================================================================
# UI RENDERING FUNCTIONS
# ============================================================================

def render_hero():
    """Render the hero section."""
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-title">🏗️ {APP_NAME}</div>
        <div class="hero-subtitle">{APP_TAGLINE}</div>
        <div class="hero-description">{APP_DESCRIPTION}</div>
        <div class="chip-container">
            <span class="chip">🔍 RAG-Grounded</span>
            <span class="chip">📄 Page-Cited</span>
            <span class="chip">🔄 Multi-Pass Review</span>
            <span class="chip">⚡ Priority Scoring</span>
            <span class="chip">☁️ Cloud Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with settings and information."""
    with st.sidebar:
        st.markdown("## ⚙️ Review Settings")
        
        # Review mode
        st.markdown("### Review Depth")
        review_mode = st.radio(
            "Select analysis depth:",
            options=list(REVIEW_MODES.keys()),
            index=list(REVIEW_MODES.keys()).index(st.session_state.review_mode),
            help="Controls how thorough the analysis will be"
        )
        st.session_state.review_mode = review_mode
        mode_info = REVIEW_MODES[review_mode]
        st.caption(f"{mode_info['description']} ({mode_info['passes']} pass{'es' if mode_info['passes'] > 1 else ''})")
        
        st.markdown("---")
        
        # Focus mode
        st.markdown("### Review Focus")
        focus_mode = st.radio(
            "Select focus area:",
            options=list(FOCUS_MODES.keys()),
            index=list(FOCUS_MODES.keys()).index(st.session_state.focus_mode),
            format_func=lambda x: f"{FOCUS_MODES[x]['icon']} {x}"
        )
        st.session_state.focus_mode = focus_mode
        st.caption(FOCUS_MODES[focus_mode]['description'])
        
        st.markdown("---")
        
        # Debug mode
        st.markdown("### Developer Options")
        st.session_state.debug_mode = st.toggle(
            "Debug mode",
            value=st.session_state.debug_mode,
            help="Show raw outputs and retrieved evidence"
        )
        
        st.markdown("---")
        
        # What we check
        st.markdown("### 🔎 Issue Types")
        st.markdown("""
        - **Missing Sections** — Absent specs
        - **Contradictions** — Conflicts
        - **Compliance Gaps** — Code issues
        - **Unclear Requirements** — Ambiguity
        - **Coordination Risks** — Cross-discipline
        """)
        
        st.markdown("---")
        
        # Quick start
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. Upload PDF document
        2. Configure review settings
        3. Run analysis
        4. Review & filter findings
        5. Export report
        """)
        
        st.markdown("---")
        st.caption(f"{APP_NAME} v{APP_VERSION}")

def render_empty_state():
    """Render empty state before upload."""
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">📄</div>
        <div class="empty-state-title">Upload a Construction Document</div>
        <div class="empty-state-description">
            Upload technical PDFs such as mechanical specifications, electrical designs, 
            structural plans, or construction specs. BuildSpec AI will identify QA/QC 
            issues with page-cited evidence.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🔴 Compliance")
        st.caption("Code gaps, missing standards, regulatory issues")
    with col2:
        st.markdown("#### 🟡 Coordination")
        st.caption("Cross-discipline conflicts, interface issues")
    with col3:
        st.markdown("#### 🟢 Completeness")
        st.caption("Missing sections, incomplete specs")

def render_document_snapshot(snapshot: DocumentSnapshot):
    """Render document intelligence snapshot."""
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📊 Document Snapshot</div>
        <div class="snapshot-grid">
            <div class="snapshot-item">
                <div class="snapshot-value">{snapshot.extracted_pages}</div>
                <div class="snapshot-label">Pages</div>
            </div>
            <div class="snapshot-item">
                <div class="snapshot-value">{snapshot.total_chars:,}</div>
                <div class="snapshot-label">Characters</div>
            </div>
            <div class="snapshot-item">
                <div class="snapshot-value">{snapshot.total_chunks}</div>
                <div class="snapshot-label">Chunks</div>
            </div>
            <div class="snapshot-item">
                <div class="snapshot-value">{len(snapshot.detected_sections)}</div>
                <div class="snapshot-label">Sections</div>
            </div>
            <div class="snapshot-item">
                <div class="snapshot-value">{snapshot.file_size}</div>
                <div class="snapshot-label">Size</div>
            </div>
            <div class="snapshot-item">
                <div class="snapshot-value">{snapshot.review_mode.split()[0]}</div>
                <div class="snapshot-label">Mode</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show detected sections in expander
    if snapshot.detected_sections and st.session_state.debug_mode:
        with st.expander("📑 Detected Sections"):
            for section in snapshot.detected_sections[:15]:
                st.caption(f"• {section}")

def render_metrics(findings: List[Finding]):
    """Render metrics row."""
    total = len(findings)
    high = len([f for f in findings if f.severity == Severity.HIGH.value])
    medium = len([f for f in findings if f.severity == Severity.MEDIUM.value])
    low = len([f for f in findings if f.severity == Severity.LOW.value])
    critical = len([f for f in findings if f.priority == Priority.CRITICAL.value])
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card metric-total">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Total</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-critical">
            <div class="metric-value">{critical}</div>
            <div class="metric-label">Critical</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card metric-high">
            <div class="metric-value">{high}</div>
            <div class="metric-label">High</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card metric-medium">
            <div class="metric-value">{medium}</div>
            <div class="metric-label">Medium</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card metric-low">
            <div class="metric-value">{low}</div>
            <div class="metric-label">Low</div>
        </div>
        """, unsafe_allow_html=True)

def render_discipline_summary(findings: List[Finding]):
    """Render discipline breakdown."""
    counts = {}
    for f in findings:
        counts[f.discipline] = counts.get(f.discipline, 0) + 1
    
    if not counts:
        return
    
    items_html = ""
    for disc, count in sorted(counts.items(), key=lambda x: -x[1]):
        items_html += f'<span class="discipline-item"><span class="discipline-count">{count}</span><span class="discipline-name">{disc.title()}</span></span>'
    
    st.markdown(f"""
    <div class="discipline-bar">{items_html}</div>
    """, unsafe_allow_html=True)

def render_filters(findings: List[Finding]) -> Tuple[List[Finding], str]:
    """Render filters and return filtered findings."""
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    
    types = sorted(set(f.type for f in findings))
    severities = [s.value for s in Severity]
    priorities = [p.value for p in Priority]
    disciplines = sorted(set(f.discipline for f in findings))
    
    with col1:
        search = st.text_input("🔍 Search", placeholder="Search findings...", label_visibility="collapsed")
    
    with col2:
        selected_types = st.multiselect("Type", types, default=types, label_visibility="collapsed")
    
    with col3:
        selected_severities = st.multiselect("Severity", severities, default=severities, label_visibility="collapsed")
    
    with col4:
        selected_priorities = st.multiselect("Priority", priorities, default=priorities, label_visibility="collapsed")
    
    with col5:
        group_by = st.selectbox(
            "Group by",
            ["severity", "priority", "discipline", "page", "type"],
            index=0,
            label_visibility="collapsed"
        )
    
    # Apply filters
    filtered = []
    for f in findings:
        if f.type not in selected_types:
            continue
        if f.severity not in selected_severities:
            continue
        if f.priority not in selected_priorities:
            continue
        if search and search.lower() not in f.title.lower() and search.lower() not in f.description.lower():
            continue
        filtered.append(f)
    
    return sort_findings(filtered, group_by), group_by

def render_findings_table(findings: List[Finding]):
    """Render findings as a table."""
    if not findings:
        st.info("No findings match the selected filters.")
        return
    
    rows = []
    for f in findings:
        rows.append({
            'ID': f.id,
            'Title': truncate_text(f.title, 50),
            'Type': f.type.replace('_', ' ').title(),
            'Severity': f.severity.upper(),
            'Priority': f.priority.upper(),
            'Discipline': f.discipline.title(),
            'Page': f.page
        })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

def render_finding_card(finding: Finding):
    """Render a single finding card."""
    severity_class = finding.severity
    priority_class = f"critical-priority" if finding.priority == Priority.CRITICAL.value else ""
    
    severity_badge = f"badge-severity-{finding.severity}"
    priority_badge = f"badge-priority-{finding.priority}"
    
    st.markdown(f"""
    <div class="finding-card {severity_class} {priority_class}">
        <div class="finding-header">
            <div class="finding-title">{finding.title}</div>
            <div class="finding-id">{finding.id}</div>
        </div>
        <div class="finding-badges">
            <span class="badge badge-type">{finding.type.replace('_', ' ')}</span>
            <span class="badge {severity_badge}">{finding.severity}</span>
            <span class="badge {priority_badge}">{finding.priority}</span>
            <span class="badge badge-discipline">{finding.discipline}</span>
            <span class="badge badge-confidence">{finding.confidence}</span>
            <span class="badge badge-page">Page {finding.page}</span>
        </div>
        <div class="finding-description">{finding.description}</div>
        <div class="finding-section">
            <div class="finding-section-label">📎 Evidence</div>
            <div class="finding-evidence">{finding.evidence}</div>
        </div>
        <div class="finding-action">
            <div class="finding-action-label">✅ Recommended Action</div>
            {finding.recommended_action}
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_summary_tab(summary: ReviewSummary):
    """Render the summary tab content."""
    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-title">📊 Executive Summary</div>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-item-label">Document</div>
                <div class="summary-item-value">{truncate_text(summary.document.filename, 25)}</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-label">Review Mode</div>
                <div class="summary-item-value">{summary.document.review_mode.split()[0]}</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-label">Focus</div>
                <div class="summary-item-value">{summary.document.focus_mode.split()[0]}</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-label">Total Findings</div>
                <div class="summary-item-value">{summary.total_findings}</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-label">Critical Issues</div>
                <div class="summary-item-value">{summary.critical_priority}</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-label">High Severity</div>
                <div class="summary-item-value">{summary.high_severity}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Discipline breakdown
    st.markdown("#### Discipline Breakdown")
    cols = st.columns(4)
    for i, (disc, count) in enumerate(summary.discipline_counts.items()):
        with cols[i % 4]:
            st.metric(disc.title(), count)
    
    # Top issues
    if summary.top_issues:
        st.markdown("#### 🔴 Top Critical Issues")
        for i, issue in enumerate(summary.top_issues, 1):
            st.markdown(f"{i}. {issue}")
    
    # Top actions
    if summary.top_actions:
        st.markdown("#### ✅ Top Recommended Actions")
        for i, action in enumerate(summary.top_actions, 1):
            st.markdown(f"{i}. {action}")

def render_evidence_tab(retrieved_chunks: List[ChunkData]):
    """Render the evidence tab content."""
    if not retrieved_chunks:
        st.info("No evidence chunks available.")
        return
    
    st.markdown(f"**{len(retrieved_chunks)} chunks retrieved** for analysis")
    
    # Group by page
    by_page = {}
    for chunk in retrieved_chunks:
        page = chunk.page_number
        if page not in by_page:
            by_page[page] = []
        by_page[page].append(chunk)
    
    for page in sorted(by_page.keys()):
        chunks = by_page[page]
        with st.expander(f"📄 Page {page} ({len(chunks)} chunks)"):
            for chunk in chunks:
                st.markdown(f"""
                <div class="evidence-chunk">
                    <div class="evidence-header">
                        <span class="evidence-page">Chunk #{chunk.chunk_id}</span>
                        <span class="evidence-similarity">Similarity: {chunk.similarity:.3f}</span>
                    </div>
                    <div class="evidence-text">{truncate_text(chunk.text, 300)}</div>
                </div>
                """, unsafe_allow_html=True)

def render_export_tab(findings: List[Finding], summary: ReviewSummary):
    """Render the export tab content."""
    st.markdown("### Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Structured Data")
        
        json_data = export_json(findings, summary)
        st.download_button(
            "📋 Download JSON",
            json_data,
            file_name=f"buildspec_findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
        
        csv_data = export_csv(findings)
        st.download_button(
            "📊 Download CSV",
            csv_data,
            file_name=f"buildspec_findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### Reports")
        
        md_data = export_markdown(findings, summary)
        st.download_button(
            "📝 Download Markdown",
            md_data,
            file_name=f"buildspec_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )
        
        txt_data = export_txt(findings, summary)
        st.download_button(
            "📄 Download TXT",
            txt_data,
            file_name=f"buildspec_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def render_debug_tab(raw_outputs: List[str], chunks: List[ChunkData]):
    """Render debug tab content."""
    st.markdown("### Debug Information")
    
    # Raw outputs
    for i, output in enumerate(raw_outputs):
        with st.expander(f"📤 Raw Output - Pass {i+1}"):
            st.code(output or "No output", language="json")
    
    # Chunk info
    with st.expander(f"📑 All Chunks ({len(chunks)})"):
        chunk_data = []
        for c in chunks[:100]:
            chunk_data.append({
                'ID': c.chunk_id,
                'Page': c.page_number,
                'Section': c.section[:30] if c.section else '-',
                'Preview': c.preview[:80]
            })
        st.dataframe(pd.DataFrame(chunk_data), use_container_width=True)

def render_footer():
    """Render footer."""
    st.markdown(f"""
    <div class="footer">
        <p><strong>{APP_NAME}</strong> • {APP_TAGLINE}</p>
        <p>Built with Streamlit • Powered by OpenAI GPT-4o • RAG-Enabled Multi-Pass Analysis</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    # Inject CSS
    inject_custom_css()
    
    # Check API key
    client = get_openai_client()
    if not client:
        st.error("⚠️ OpenAI API key not found. Set OPENAI_API_KEY in .env or Streamlit secrets.")
        st.stop()
    
    # Render sidebar and hero
    render_sidebar()
    render_hero()
    
    # File upload
    st.markdown("### 📁 Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload a construction engineering PDF",
        type=['pdf'],
        help="Technical specifications, engineering documents, construction plans"
    )
    
    if uploaded_file is None:
        render_empty_state()
        render_footer()
        return
    
    # Check for file change
    file_content = uploaded_file.getvalue()
    file_hash = get_file_hash(file_content)
    
    if st.session_state.current_file_hash != file_hash:
        # New file - extract and process
        st.session_state.current_file = uploaded_file.name
        st.session_state.current_file_hash = file_hash
        st.session_state.analysis_complete = False
        st.session_state.findings = []
        
        with st.spinner("Extracting document..."):
            pages, sections = extract_pdf_pages(uploaded_file)
            chunks = chunk_pages(pages)
        
        st.session_state.pages = pages
        st.session_state.sections = sections
        st.session_state.chunks = chunks
    
    pages = st.session_state.pages
    sections = st.session_state.sections
    chunks = st.session_state.chunks
    
    # Validate extraction
    if not pages:
        st.error("Could not extract text from this PDF. It may be scanned or image-based.")
        return
    
    if not chunks:
        st.error("No processable content found in the document.")
        return
    
    # Create document snapshot
    total_chars = sum(p.char_count for p in pages)
    snapshot = DocumentSnapshot(
        filename=uploaded_file.name,
        file_size=format_file_size(len(file_content)),
        total_pages=len(pages) + len([p for p in range(1, 100) if p not in [pg.page_number for pg in pages]]),
        extracted_pages=len(pages),
        total_chars=total_chars,
        total_chunks=len(chunks),
        detected_sections=sections,
        review_mode=st.session_state.review_mode,
        focus_mode=st.session_state.focus_mode
    )
    st.session_state.document_snapshot = snapshot
    
    # Document warnings
    if len(pages) > 100:
        st.warning(f"Large document ({len(pages)} pages). Deep Review may take longer.")
    
    if total_chars > 400000:
        st.warning("Document has significant content. Consider Standard or Quick Review for faster results.")
    
    # Render document snapshot
    render_document_snapshot(snapshot)
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_analysis = st.button(
            f"🚀 Run {st.session_state.review_mode}",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.analysis_in_progress
        )
    
    # Run analysis
    if run_analysis and not st.session_state.analysis_in_progress:
        st.session_state.analysis_in_progress = True
        st.session_state.error_message = None
        
        progress_container = st.empty()
        
        try:
            with progress_container.container():
                st.markdown("### 🔄 Analysis Pipeline")
            
            findings, retrieved, raw_outputs = analyze_document(
                chunks, client,
                st.session_state.review_mode,
                st.session_state.focus_mode,
                progress_container
            )
            
            st.session_state.findings = findings
            st.session_state.retrieved_chunks = retrieved
            st.session_state.raw_model_outputs = raw_outputs
            st.session_state.analysis_complete = True
            
            progress_container.empty()
            
            if findings:
                st.success(f"✅ Analysis complete! Found {len(findings)} issues.")
            else:
                st.info("✅ Analysis complete. No significant issues identified.")
                
        except Exception as e:
            st.session_state.error_message = str(e)
            st.error(f"Analysis failed: {str(e)}")
        finally:
            st.session_state.analysis_in_progress = False
    
    # Display results
    if st.session_state.analysis_complete:
        findings = st.session_state.findings
        retrieved = st.session_state.retrieved_chunks
        raw_outputs = st.session_state.raw_model_outputs
        
        if findings:
            # Generate summary
            summary = generate_review_summary(findings, snapshot)
            
            st.markdown("---")
            
            # Metrics
            render_metrics(findings)
            render_discipline_summary(findings)
            
            st.markdown("---")
            
            # Tabs
            tab_findings, tab_summary, tab_evidence, tab_export = st.tabs([
                "📋 Findings",
                "📊 Summary", 
                "🔍 Evidence",
                "📥 Export"
            ])
            
            if st.session_state.debug_mode:
                tabs = st.tabs(["📋 Findings", "📊 Summary", "🔍 Evidence", "📥 Export", "🔧 Debug"])
                tab_findings, tab_summary, tab_evidence, tab_export, tab_debug = tabs
            else:
                tab_findings, tab_summary, tab_evidence, tab_export = st.tabs([
                    "📋 Findings", "📊 Summary", "🔍 Evidence", "📥 Export"
                ])
            
            with tab_findings:
                st.markdown("### Filter & Browse Findings")
                filtered_findings, group_by = render_filters(findings)
                
                st.markdown(f"**Showing {len(filtered_findings)} of {len(findings)} findings** (grouped by {group_by})")
                
                view_mode = st.radio(
                    "View mode",
                    ["Cards", "Table"],
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
                if view_mode == "Cards":
                    for f in filtered_findings:
                        render_finding_card(f)
                else:
                    render_findings_table(filtered_findings)
            
            with tab_summary:
                render_summary_tab(summary)
            
            with tab_evidence:
                render_evidence_tab(retrieved)
            
            with tab_export:
                render_export_tab(findings, summary)
            
            if st.session_state.debug_mode:
                with tab_debug:
                    render_debug_tab(raw_outputs, chunks)
        
        else:
            # No findings state
            st.markdown("""
            <div class="card">
                <div class="card-title">✅ No Significant Issues Found</div>
                <p style="color: #94a3b8;">
                    The analysis did not identify significant QA/QC issues in this document.
                    This could indicate a well-structured document, or you may want to try 
                    a deeper review mode or different focus area.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    render_footer()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
