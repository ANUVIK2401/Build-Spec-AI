"""
BuildSpec AI - AI QA/QC Copilot for Construction Engineering Documents

A production-ready Streamlit application that reviews technical PDFs for:
- Compliance gaps
- Contradictions
- Missing sections
- Unclear requirements
- Coordination risks

Uses real RAG (Retrieval-Augmented Generation) with page-cited evidence.
"""

import os
import io
import json
import re
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="BuildSpec AI",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium dark theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #6366f1;
        --primary-light: #818cf8;
        --secondary: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --background: #0f172a;
        --surface: #1e293b;
        --surface-light: #334155;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.25rem;
        color: #94a3b8;
        margin-bottom: 1rem;
    }
    
    .hero-description {
        font-size: 1rem;
        color: #64748b;
        max-width: 800px;
        margin: 0 auto 1.5rem auto;
    }
    
    /* Status chips */
    .chip-container {
        display: flex;
        justify-content: center;
        gap: 0.75rem;
        flex-wrap: wrap;
    }
    
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: #334155;
        border: 1px solid #475569;
        border-radius: 9999px;
        font-size: 0.875rem;
        color: #e2e8f0;
    }
    
    .chip-icon {
        font-size: 1rem;
    }
    
    /* Cards */
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 1rem;
    }
    
    /* Metrics */
    .metric-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        flex: 1;
        min-width: 150px;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    .metric-high { border-left: 4px solid #ef4444; }
    .metric-medium { border-left: 4px solid #f59e0b; }
    .metric-low { border-left: 4px solid #10b981; }
    .metric-total { border-left: 4px solid #6366f1; }
    
    /* Finding cards */
    .finding-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #6366f1;
    }
    
    .finding-card.high { border-left-color: #ef4444; }
    .finding-card.medium { border-left-color: #f59e0b; }
    .finding-card.low { border-left-color: #10b981; }
    
    .finding-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.75rem;
    }
    
    .finding-badges {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    .badge-type { background: #312e81; color: #a5b4fc; }
    .badge-severity-high { background: #7f1d1d; color: #fca5a5; }
    .badge-severity-medium { background: #78350f; color: #fcd34d; }
    .badge-severity-low { background: #064e3b; color: #6ee7b7; }
    .badge-discipline { background: #1e3a5f; color: #7dd3fc; }
    .badge-confidence { background: #3f3f46; color: #d4d4d8; }
    .badge-page { background: #4a044e; color: #f0abfc; }
    
    .finding-description {
        color: #cbd5e1;
        margin-bottom: 1rem;
        line-height: 1.6;
    }
    
    .finding-evidence {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        font-size: 0.875rem;
        color: #94a3b8;
        font-style: italic;
    }
    
    .finding-evidence-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .finding-action {
        background: #0d3320;
        border: 1px solid #166534;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.875rem;
        color: #86efac;
    }
    
    .finding-action-label {
        font-size: 0.75rem;
        color: #4ade80;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed #475569;
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        background: #1e293b;
        margin-bottom: 1.5rem;
    }
    
    .upload-area:hover {
        border-color: #6366f1;
        background: #1e293b;
    }
    
    /* Document info */
    .doc-info {
        display: flex;
        gap: 2rem;
        flex-wrap: wrap;
        padding: 1rem;
        background: #0f172a;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    .doc-info-item {
        display: flex;
        flex-direction: column;
    }
    
    .doc-info-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
    }
    
    .doc-info-value {
        font-size: 1rem;
        color: #f8fafc;
        font-weight: 500;
    }
    
    /* Progress */
    .progress-container {
        margin: 1.5rem 0;
    }
    
    /* Summary report */
    .summary-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
    }
    
    .summary-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .summary-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .summary-item {
        background: #334155;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .summary-item-label {
        font-size: 0.75rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .summary-item-value {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f8fafc;
        margin-top: 0.25rem;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .empty-state-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.5rem;
    }
    
    .empty-state-description {
        color: #94a3b8;
        max-width: 500px;
        margin: 0 auto;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .sidebar-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Table styling */
    .dataframe {
        background: #1e293b !important;
        border-radius: 8px !important;
    }
    
    .dataframe th {
        background: #334155 !important;
        color: #f8fafc !important;
    }
    
    .dataframe td {
        background: #1e293b !important;
        color: #cbd5e1 !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.875rem;
        border-top: 1px solid #334155;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'findings': [],
        'chunks': [],
        'pages': [],
        'file_info': None,
        'analysis_complete': False,
        'raw_model_output': None,
        'retrieved_chunks': [],
        'review_mode': 'Standard Review',
        'debug_mode': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# Review mode configurations
REVIEW_MODES = {
    'Quick Review': {
        'description': 'Fast analysis with fewer chunks. Good for initial screening.',
        'top_k': 5,
        'max_issues': 5
    },
    'Standard Review': {
        'description': 'Balanced depth and coverage. Recommended for most documents.',
        'top_k': 10,
        'max_issues': 10
    },
    'Deep Review': {
        'description': 'Comprehensive analysis with detailed findings. Best for critical documents.',
        'top_k': 20,
        'max_issues': 20
    }
}


def get_openai_client() -> Optional[OpenAI]:
    """Get OpenAI client with API key from environment or Streamlit secrets."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        try:
            api_key = st.secrets.get('OPENAI_API_KEY')
        except:
            pass
    
    if not api_key:
        return None
    
    return OpenAI(api_key=api_key)


def extract_pdf_pages(pdf_file) -> List[Dict[str, Any]]:
    """
    Extract text from PDF file page by page using PyMuPDF.
    
    Preserves page numbers and handles extraction errors gracefully.
    """
    pages = []
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_path = tmp_file.name
        
        # Open PDF with PyMuPDF
        doc = fitz.open(tmp_path)
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                text = page.get_text("text")
                
                # Clean up text
                text = text.strip()
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                
                # Skip mostly empty pages
                if len(text) < 50:
                    continue
                
                pages.append({
                    'page_number': page_num + 1,
                    'text': text,
                    'preview': text[:200] + '...' if len(text) > 200 else text,
                    'char_count': len(text)
                })
            except Exception as e:
                # Log but don't crash on individual page errors
                if st.session_state.debug_mode:
                    st.warning(f"Could not extract page {page_num + 1}: {str(e)}")
                continue
        
        doc.close()
        os.unlink(tmp_path)
        
    except Exception as e:
        st.error(f"Error extracting PDF: {str(e)}")
        return []
    
    return pages


def chunk_pages(pages: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Chunk extracted pages while preserving page metadata.
    
    Uses RecursiveCharacterTextSplitter for stable, deployment-safe chunking
    that adapts to technical document structure.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    chunk_idx = 0
    
    for page in pages:
        page_chunks = splitter.split_text(page['text'])
        
        for chunk_text in page_chunks:
            if len(chunk_text.strip()) < 20:
                continue
                
            chunks.append({
                'chunk_id': chunk_idx,
                'page_number': page['page_number'],
                'text': chunk_text.strip(),
                'preview': chunk_text[:150].strip() + '...' if len(chunk_text) > 150 else chunk_text.strip()
            })
            chunk_idx += 1
    
    return chunks


def get_embeddings(texts: List[str], client: OpenAI) -> np.ndarray:
    """Generate embeddings using OpenAI text-embedding-3-small."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return np.array([])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vectors."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def retrieve_relevant_chunks(query: str, chunks: List[Dict], chunk_embeddings: np.ndarray, 
                            client: OpenAI, top_k: int = 10) -> List[Dict]:
    """
    Retrieve the most relevant chunks using cosine similarity.
    
    This is the core RAG retrieval step - finds chunks most relevant to the query.
    """
    if len(chunks) == 0 or chunk_embeddings.size == 0:
        return []
    
    # Get query embedding
    query_embedding = get_embeddings([query], client)
    if query_embedding.size == 0:
        return []
    
    # Compute similarities
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return chunks with similarity scores
    retrieved = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk['similarity'] = float(similarities[idx])
        retrieved.append(chunk)
    
    return retrieved


def build_analysis_prompt(retrieved_chunks: List[Dict], review_mode: str) -> str:
    """Build the analysis prompt with retrieved evidence."""
    
    mode_config = REVIEW_MODES[review_mode]
    max_issues = mode_config['max_issues']
    
    # Format evidence from retrieved chunks
    evidence_text = "\n\n".join([
        f"[Page {c['page_number']}]: {c['text']}"
        for c in retrieved_chunks
    ])
    
    prompt = f"""You are an expert construction engineering document QA/QC reviewer. Your role is to analyze technical specifications, engineering documents, and construction plans to identify potential issues.

## Your Task
Review the following extracted evidence from a construction engineering document and identify quality issues that could impact project success.

## Evidence Retrieved from Document
{evidence_text}

## Issue Types to Identify
- missing_section: Required sections or specifications that appear to be absent
- contradiction: Conflicting requirements or specifications within the document
- compliance_gap: Areas where the document may not meet industry standards or codes
- unclear_requirement: Vague or ambiguous specifications that could lead to misinterpretation
- coordination_risk: Potential conflicts between different engineering disciplines

## Output Requirements
Return a JSON array of findings. Each finding must have:
- type: One of [missing_section, contradiction, compliance_gap, unclear_requirement, coordination_risk]
- severity: One of [high, medium, low]
- discipline: One of [mechanical, electrical, structural, general]
- confidence: One of [high, medium, low] - your confidence in this finding
- page: The page number from the evidence (integer)
- title: A brief descriptive title (max 80 chars)
- description: Detailed explanation of the issue (2-3 sentences)
- evidence: The specific text or observation that supports this finding
- recommended_action: Specific action to resolve the issue

## Critical Rules
1. ONLY identify issues that are clearly supported by the provided evidence
2. DO NOT hallucinate issues not grounded in the evidence
3. Include specific page citations from the evidence
4. Return at most {max_issues} findings
5. Return an empty array [] if no clear issues are found
6. Order findings by severity (high first)

## Response Format
Return ONLY a valid JSON array. No markdown, no explanation, just the JSON array.

Example format:
[
  {{
    "type": "compliance_gap",
    "severity": "high",
    "discipline": "electrical",
    "confidence": "medium",
    "page": 12,
    "title": "Missing emergency lighting requirement",
    "description": "The document does not specify emergency lighting coverage for exit paths, which is typically required by building codes.",
    "evidence": "Section on lighting requirements omits emergency fixtures near egress routes.",
    "recommended_action": "Add emergency lighting requirements aligned with IBC and local codes."
  }}
]
"""
    return prompt


def parse_model_output(raw_output: str) -> List[Dict]:
    """
    Parse and validate model JSON output with robust error handling.
    
    Handles markdown fences, malformed JSON, and normalizes enum values.
    """
    if not raw_output:
        return []
    
    # Strip markdown code fences
    cleaned = raw_output.strip()
    if cleaned.startswith('```'):
        # Remove opening fence
        cleaned = re.sub(r'^```(?:json)?\n?', '', cleaned)
        # Remove closing fence
        cleaned = re.sub(r'\n?```$', '', cleaned)
    
    # Try to parse JSON
    try:
        findings = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON array in the output
        match = re.search(r'\[[\s\S]*\]', cleaned)
        if match:
            try:
                findings = json.loads(match.group())
            except:
                return []
        else:
            return []
    
    if not isinstance(findings, list):
        return []
    
    # Validate and normalize each finding
    valid_types = {'missing_section', 'contradiction', 'compliance_gap', 'unclear_requirement', 'coordination_risk'}
    valid_severities = {'high', 'medium', 'low'}
    valid_disciplines = {'mechanical', 'electrical', 'structural', 'general'}
    valid_confidence = {'high', 'medium', 'low'}
    
    validated = []
    seen_titles = set()
    
    for f in findings:
        if not isinstance(f, dict):
            continue
        
        # Normalize values
        f_type = str(f.get('type', 'unclear_requirement')).lower().replace(' ', '_')
        if f_type not in valid_types:
            f_type = 'unclear_requirement'
        
        severity = str(f.get('severity', 'medium')).lower()
        if severity not in valid_severities:
            severity = 'medium'
        
        discipline = str(f.get('discipline', 'general')).lower()
        if discipline not in valid_disciplines:
            discipline = 'general'
        
        confidence = str(f.get('confidence', 'medium')).lower()
        if confidence not in valid_confidence:
            confidence = 'medium'
        
        title = str(f.get('title', 'Untitled Finding'))[:100]
        
        # Deduplicate
        if title.lower() in seen_titles:
            continue
        seen_titles.add(title.lower())
        
        # Ensure page is an integer
        try:
            page = int(f.get('page', 1))
        except:
            page = 1
        
        validated.append({
            'type': f_type,
            'severity': severity,
            'discipline': discipline,
            'confidence': confidence,
            'page': page,
            'title': title,
            'description': str(f.get('description', ''))[:500],
            'evidence': str(f.get('evidence', ''))[:300],
            'recommended_action': str(f.get('recommended_action', ''))[:300]
        })
    
    # Sort by severity
    severity_order = {'high': 0, 'medium': 1, 'low': 2}
    validated.sort(key=lambda x: severity_order.get(x['severity'], 1))
    
    return validated


def analyze_document(chunks: List[Dict], client: OpenAI, review_mode: str) -> Tuple[List[Dict], List[Dict], str]:
    """
    Run the full RAG analysis pipeline.
    
    1. Generate embeddings for all chunks
    2. Create analysis query
    3. Retrieve relevant chunks
    4. Send to GPT-4o for analysis
    5. Parse and return findings
    """
    if not chunks:
        return [], [], ""
    
    mode_config = REVIEW_MODES[review_mode]
    top_k = mode_config['top_k']
    
    # Step 1: Generate embeddings for all chunks
    with st.spinner("Generating embeddings..."):
        chunk_texts = [c['text'] for c in chunks]
        chunk_embeddings = get_embeddings(chunk_texts, client)
        
        if chunk_embeddings.size == 0:
            return [], [], "Failed to generate embeddings"
    
    # Step 2: Create analysis queries to retrieve diverse evidence
    queries = [
        "What are the safety requirements and compliance specifications?",
        "What are the mechanical, electrical, and structural coordination requirements?",
        "What sections might be missing or incomplete?",
        "Are there any contradictions or conflicting requirements?",
        "What specifications might be unclear or ambiguous?"
    ]
    
    # Step 3: Retrieve relevant chunks for each query
    all_retrieved = []
    seen_chunk_ids = set()
    
    with st.spinner("Retrieving relevant evidence..."):
        for query in queries:
            retrieved = retrieve_relevant_chunks(
                query, chunks, chunk_embeddings, client, top_k=top_k // len(queries) + 2
            )
            for chunk in retrieved:
                if chunk['chunk_id'] not in seen_chunk_ids:
                    all_retrieved.append(chunk)
                    seen_chunk_ids.add(chunk['chunk_id'])
        
        # Limit total retrieved chunks
        all_retrieved = all_retrieved[:top_k]
    
    if not all_retrieved:
        return [], [], "No relevant chunks retrieved"
    
    # Step 4: Build prompt and call GPT-4o
    with st.spinner("Analyzing document with GPT-4o..."):
        prompt = build_analysis_prompt(all_retrieved, review_mode)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert construction engineering document QA/QC reviewer. Return only valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )
            raw_output = response.choices[0].message.content
        except Exception as e:
            return [], all_retrieved, f"API Error: {str(e)}"
    
    # Step 5: Parse findings
    findings = parse_model_output(raw_output)
    
    return findings, all_retrieved, raw_output


def render_hero():
    """Render the hero section."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🏗️ BuildSpec AI</div>
        <div class="hero-subtitle">AI QA/QC Copilot for Construction Engineering Documents</div>
        <div class="hero-description">
            Review technical PDFs for compliance gaps, contradictions, missing sections, 
            and coordination risks with page-cited RAG evidence.
        </div>
        <div class="chip-container">
            <span class="chip"><span class="chip-icon">🔍</span> RAG Enabled</span>
            <span class="chip"><span class="chip-icon">📄</span> Page-Cited Findings</span>
            <span class="chip"><span class="chip-icon">☁️</span> Streamlit Cloud Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with settings and information."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Review mode selector
        st.markdown("### Review Mode")
        review_mode = st.radio(
            "Select analysis depth:",
            options=list(REVIEW_MODES.keys()),
            index=1,
            help="Choose how thorough the analysis should be"
        )
        st.session_state.review_mode = review_mode
        
        mode_info = REVIEW_MODES[review_mode]
        st.caption(mode_info['description'])
        
        st.markdown("---")
        
        # Debug mode toggle
        st.markdown("### Debug Mode")
        st.session_state.debug_mode = st.toggle(
            "Enable debug mode",
            value=st.session_state.debug_mode,
            help="Show raw model output and retrieved chunks"
        )
        
        st.markdown("---")
        
        # What the app checks
        st.markdown("### 🔎 What We Check")
        st.markdown("""
        - **Missing Sections**: Required specs that appear absent
        - **Contradictions**: Conflicting requirements
        - **Compliance Gaps**: Standards/code issues
        - **Unclear Requirements**: Ambiguous specs
        - **Coordination Risks**: Cross-discipline conflicts
        """)
        
        st.markdown("---")
        
        # Quick start
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. Upload a construction PDF
        2. Select review depth
        3. Click "Run Analysis"
        4. Review findings
        5. Export results
        """)
        
        st.markdown("---")
        
        # Tech stack
        st.markdown("### 🛠️ Tech Stack")
        st.caption("OpenAI GPT-4o • RAG Pipeline • PyMuPDF • FAISS/NumPy • Streamlit")


def render_empty_state():
    """Render empty state before file upload."""
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">📄</div>
        <div class="empty-state-title">Upload a Construction Document</div>
        <div class="empty-state-description">
            Upload technical PDFs such as mechanical specifications, electrical designs, 
            structural drawings, or construction plans. BuildSpec AI will identify potential 
            quality issues with page-cited evidence.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Example issues detected
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🔴 Compliance Gaps")
        st.caption("Missing code requirements, incomplete safety specifications")
    
    with col2:
        st.markdown("#### 🟡 Contradictions")
        st.caption("Conflicting requirements between sections or disciplines")
    
    with col3:
        st.markdown("#### 🟢 Coordination Risks")
        st.caption("Potential MEP conflicts, unclear handoffs")


def render_metrics(findings: List[Dict]):
    """Render the metrics row."""
    total = len(findings)
    high = len([f for f in findings if f['severity'] == 'high'])
    medium = len([f for f in findings if f['severity'] == 'medium'])
    low = len([f for f in findings if f['severity'] == 'low'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card metric-total">
            <div class="metric-value">{total}</div>
            <div class="metric-label">Total Findings</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card metric-high">
            <div class="metric-value">{high}</div>
            <div class="metric-label">High Severity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card metric-medium">
            <div class="metric-value">{medium}</div>
            <div class="metric-label">Medium Severity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card metric-low">
            <div class="metric-value">{low}</div>
            <div class="metric-label">Low Severity</div>
        </div>
        """, unsafe_allow_html=True)


def render_filters(findings: List[Dict]) -> List[Dict]:
    """Render filters and return filtered findings."""
    col1, col2, col3, col4 = st.columns(4)
    
    # Get unique values
    types = sorted(set(f['type'] for f in findings))
    severities = ['high', 'medium', 'low']
    disciplines = sorted(set(f['discipline'] for f in findings))
    confidences = ['high', 'medium', 'low']
    
    with col1:
        selected_types = st.multiselect("Issue Type", types, default=types)
    
    with col2:
        selected_severities = st.multiselect("Severity", severities, default=severities)
    
    with col3:
        selected_disciplines = st.multiselect("Discipline", disciplines, default=disciplines)
    
    with col4:
        selected_confidence = st.multiselect("Confidence", confidences, default=confidences)
    
    # Apply filters
    filtered = [
        f for f in findings
        if f['type'] in selected_types
        and f['severity'] in selected_severities
        and f['discipline'] in selected_disciplines
        and f['confidence'] in selected_confidence
    ]
    
    return filtered


def render_findings_table(findings: List[Dict]):
    """Render findings as a table."""
    if not findings:
        st.info("No findings match the selected filters.")
        return
    
    df = pd.DataFrame([
        {
            'Title': f['title'],
            'Type': f['type'].replace('_', ' ').title(),
            'Severity': f['severity'].upper(),
            'Discipline': f['discipline'].title(),
            'Page': f['page'],
            'Confidence': f['confidence'].title()
        }
        for f in findings
    ])
    
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_finding_card(finding: Dict, index: int):
    """Render a single finding card."""
    severity_class = finding['severity']
    
    # Badge colors
    severity_badge_class = f"badge-severity-{finding['severity']}"
    
    st.markdown(f"""
    <div class="finding-card {severity_class}">
        <div class="finding-title">{finding['title']}</div>
        <div class="finding-badges">
            <span class="badge badge-type">{finding['type'].replace('_', ' ').title()}</span>
            <span class="badge {severity_badge_class}">{finding['severity'].upper()}</span>
            <span class="badge badge-discipline">{finding['discipline'].title()}</span>
            <span class="badge badge-confidence">Confidence: {finding['confidence'].title()}</span>
            <span class="badge badge-page">📄 Page {finding['page']}</span>
        </div>
        <div class="finding-description">{finding['description']}</div>
        <div class="finding-evidence">
            <div class="finding-evidence-label">📎 Evidence</div>
            {finding['evidence']}
        </div>
        <div class="finding-action">
            <div class="finding-action-label">✅ Recommended Action</div>
            {finding['recommended_action']}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_summary_report(findings: List[Dict], file_info: Dict, chunks: List[Dict]):
    """Render the summary report card."""
    high_count = len([f for f in findings if f['severity'] == 'high'])
    disciplines = set(f['discipline'] for f in findings)
    
    top_issues = [f['title'] for f in findings[:3]]
    top_actions = [f['recommended_action'] for f in findings[:3] if f['recommended_action']]
    
    st.markdown(f"""
    <div class="summary-card">
        <div class="summary-title">📊 Review Summary Report</div>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-item-label">Document</div>
                <div class="summary-item-value">{file_info['name']}</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-label">Review Mode</div>
                <div class="summary-item-value">{st.session_state.review_mode}</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-label">Pages Analyzed</div>
                <div class="summary-item-value">{file_info['pages']}</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-label">Chunks Created</div>
                <div class="summary-item-value">{len(chunks)}</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-label">Total Findings</div>
                <div class="summary-item-value">{len(findings)}</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-label">High Severity</div>
                <div class="summary-item-value">{high_count}</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-label">Disciplines Affected</div>
                <div class="summary-item-value">{', '.join(d.title() for d in disciplines) if disciplines else 'None'}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top issues
    if top_issues:
        st.markdown("#### 🔴 Top Critical Issues")
        for issue in top_issues:
            st.markdown(f"- {issue}")
    
    # Top actions
    if top_actions:
        st.markdown("#### ✅ Top Recommended Actions")
        for action in top_actions:
            st.markdown(f"- {action}")


def generate_export_json(findings: List[Dict], file_info: Dict) -> str:
    """Generate JSON export."""
    export_data = {
        'document': file_info['name'],
        'review_mode': st.session_state.review_mode,
        'generated_at': datetime.now().isoformat(),
        'total_findings': len(findings),
        'findings': findings
    }
    return json.dumps(export_data, indent=2)


def generate_export_csv(findings: List[Dict]) -> str:
    """Generate CSV export."""
    if not findings:
        return "No findings to export"
    
    df = pd.DataFrame(findings)
    return df.to_csv(index=False)


def generate_export_markdown(findings: List[Dict], file_info: Dict, chunks: List[Dict]) -> str:
    """Generate Markdown export."""
    high_count = len([f for f in findings if f['severity'] == 'high'])
    medium_count = len([f for f in findings if f['severity'] == 'medium'])
    low_count = len([f for f in findings if f['severity'] == 'low'])
    
    md = f"""# BuildSpec AI - Document Review Report

## Document Information
- **Filename**: {file_info['name']}
- **Review Mode**: {st.session_state.review_mode}
- **Pages Analyzed**: {file_info['pages']}
- **Chunks Processed**: {len(chunks)}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- **Total Findings**: {len(findings)}
- **High Severity**: {high_count}
- **Medium Severity**: {medium_count}
- **Low Severity**: {low_count}

## Findings

"""
    
    for i, f in enumerate(findings, 1):
        md += f"""### {i}. {f['title']}

- **Type**: {f['type'].replace('_', ' ').title()}
- **Severity**: {f['severity'].upper()}
- **Discipline**: {f['discipline'].title()}
- **Confidence**: {f['confidence'].title()}
- **Page**: {f['page']}

**Description**: {f['description']}

**Evidence**: {f['evidence']}

**Recommended Action**: {f['recommended_action']}

---

"""
    
    return md


def render_export_section(findings: List[Dict], file_info: Dict, chunks: List[Dict]):
    """Render export buttons."""
    st.markdown("### 📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        json_data = generate_export_json(findings, file_info)
        st.download_button(
            "📋 Download JSON",
            json_data,
            file_name=f"buildspec_findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col2:
        csv_data = generate_export_csv(findings)
        st.download_button(
            "📊 Download CSV",
            csv_data,
            file_name=f"buildspec_findings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        md_data = generate_export_markdown(findings, file_info, chunks)
        st.download_button(
            "📝 Download Markdown",
            md_data,
            file_name=f"buildspec_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )


def render_debug_section(raw_output: str, retrieved_chunks: List[Dict], chunks: List[Dict]):
    """Render debug information when debug mode is enabled."""
    if not st.session_state.debug_mode:
        return
    
    st.markdown("---")
    st.markdown("### 🔧 Debug Information")
    
    with st.expander("📤 Raw Model Output", expanded=False):
        st.code(raw_output or "No output", language="json")
    
    with st.expander(f"🔍 Retrieved Chunks ({len(retrieved_chunks)})", expanded=False):
        for chunk in retrieved_chunks:
            st.markdown(f"**Page {chunk['page_number']}** (Similarity: {chunk.get('similarity', 0):.3f})")
            st.text(chunk['preview'])
            st.markdown("---")
    
    with st.expander(f"📑 All Chunks ({len(chunks)})", expanded=False):
        chunk_df = pd.DataFrame([
            {'ID': c['chunk_id'], 'Page': c['page_number'], 'Preview': c['preview'][:100]}
            for c in chunks[:50]  # Limit to first 50
        ])
        st.dataframe(chunk_df, use_container_width=True)


def render_footer():
    """Render the footer."""
    st.markdown("""
    <div class="footer">
        <p>BuildSpec AI • AI QA/QC Copilot for Construction Engineering Documents</p>
        <p>Built with Streamlit • Powered by OpenAI GPT-4o • RAG-Enabled Analysis</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Check for OpenAI API key
    client = get_openai_client()
    
    if not client:
        st.error("⚠️ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file or Streamlit secrets.")
        st.stop()
    
    # Render sidebar
    render_sidebar()
    
    # Render hero
    render_hero()
    
    # File upload section
    st.markdown("### 📁 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Upload a construction engineering PDF",
        type=['pdf'],
        help="Upload technical specifications, engineering documents, or construction plans"
    )
    
    if uploaded_file is None:
        render_empty_state()
        render_footer()
        return
    
    # Display file info
    file_size = len(uploaded_file.getvalue()) / 1024  # KB
    
    # Extract pages to get page count
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.pages = extract_pdf_pages(uploaded_file)
        st.session_state.chunks = chunk_pages(st.session_state.pages)
        st.session_state.analysis_complete = False
        st.session_state.findings = []
    
    pages = st.session_state.pages
    chunks = st.session_state.chunks
    
    file_info = {
        'name': uploaded_file.name,
        'size': f"{file_size:.1f} KB",
        'pages': len(pages),
        'chunks': len(chunks)
    }
    st.session_state.file_info = file_info
    
    # Document info card
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📄 Document Information</div>
        <div class="doc-info">
            <div class="doc-info-item">
                <span class="doc-info-label">Filename</span>
                <span class="doc-info-value">{file_info['name']}</span>
            </div>
            <div class="doc-info-item">
                <span class="doc-info-label">Size</span>
                <span class="doc-info-value">{file_info['size']}</span>
            </div>
            <div class="doc-info-item">
                <span class="doc-info-label">Pages</span>
                <span class="doc-info-value">{file_info['pages']}</span>
            </div>
            <div class="doc-info-item">
                <span class="doc-info-label">Chunks</span>
                <span class="doc-info-value">{file_info['chunks']}</span>
            </div>
            <div class="doc-info-item">
                <span class="doc-info-label">Review Mode</span>
                <span class="doc-info-value">{st.session_state.review_mode}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Analysis button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    if run_analysis:
        if not chunks:
            st.error("No text could be extracted from the PDF. Please try a different file.")
        else:
            # Run analysis
            findings, retrieved_chunks, raw_output = analyze_document(
                chunks, client, st.session_state.review_mode
            )
            
            st.session_state.findings = findings
            st.session_state.retrieved_chunks = retrieved_chunks
            st.session_state.raw_model_output = raw_output
            st.session_state.analysis_complete = True
            
            if findings:
                st.success(f"✅ Analysis complete! Found {len(findings)} potential issues.")
            else:
                st.info("Analysis complete. No significant issues were identified in the document.")
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.findings:
        findings = st.session_state.findings
        
        st.markdown("---")
        
        # Summary report
        render_summary_report(findings, file_info, chunks)
        
        st.markdown("---")
        
        # Metrics
        st.markdown("### 📊 Findings Overview")
        render_metrics(findings)
        
        # Filters
        st.markdown("### 🔍 Filter Findings")
        filtered_findings = render_filters(findings)
        
        # Tabs for table and cards
        tab1, tab2 = st.tabs(["📋 Table View", "🃏 Card View"])
        
        with tab1:
            render_findings_table(filtered_findings)
        
        with tab2:
            for i, finding in enumerate(filtered_findings):
                render_finding_card(finding, i)
        
        st.markdown("---")
        
        # Export section
        render_export_section(findings, file_info, chunks)
        
        # Debug section
        render_debug_section(
            st.session_state.raw_model_output,
            st.session_state.retrieved_chunks,
            chunks
        )
    
    elif st.session_state.analysis_complete and not st.session_state.findings:
        st.markdown("---")
        st.markdown("""
        <div class="card">
            <div class="card-title">✅ No Issues Found</div>
            <p style="color: #94a3b8;">
                The analysis did not identify any significant QA/QC issues in this document. 
                This could mean the document is well-structured, or you may want to try a 
                deeper review mode for more thorough analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    render_footer()


if __name__ == "__main__":
    main()
