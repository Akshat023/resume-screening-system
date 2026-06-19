# Resume Screening System

An AI-powered Applicant Tracking System (ATS) built in Python. Automatically screens resumes against job descriptions, scores candidates 0-100, and sends rejection/shortlist emails.

## Features
- PDF, DOCX, TXT, MD resume support
- BERT semantic similarity scoring (all-MiniLM-L6-v2)
- Skill taxonomy with aliases and inference (80+ skills, 10 domains)
- Skill gate: if skills don't match, experience is not considered
- Explainability dashboard with score breakdown
- Rule-based feedback: strengths, gaps, improvement suggestions
- AI commentary via Ollama (local LLM, free, no API key)
- Gmail SMTP email integration
- SQLite persistence with audit log
- Analytics: funnel, top skills, score distribution
- CSV export

## Tech Stack
- **UI:** Streamlit
- **NLP:** spaCy (en_core_web_sm)
- **Semantic similarity:** sentence-transformers (all-MiniLM-L6-v2)
- **ML:** scikit-learn (TF-IDF, LinearSVC)
- **File parsing:** pdfminer.six, python-docx
- **Database:** SQLite
- **Email:** smtplib + Gmail SMTP
- **AI commentary:** Ollama (llama3.2:3b)

## Setup

> **Important:** Place project at `C:\Projects\resume_screening` — NOT inside OneDrive.

```bash
# Run setup script
setup.bat

# Configure email (optional)
copy .env.example .env
# Edit .env with your Gmail credentials

# Start the app
streamlit run layer7_app.py
```

## Architecture

```
Layer 1  File Ingestion      PDF/DOCX/TXT → raw text
Layer 2  Preprocessing       Clean + section detection + tokenize
Layer 3  Feature Engineering Skill matching + BERT similarity
Layer 4  Extraction          Name/email/experience/education/certs
Layer 5  Scoring Engine      Skill gate + weighted score + feedback
Layer 6  Database            SQLite persistence
Layer 7  Streamlit UI        HR-facing web interface
Layer 8  Email               Gmail SMTP rejection/shortlist emails
```

## Scoring

| Component | Weight | Method |
|---|---|---|
| Skill Match | 40% | Taxonomy matching with aliases + inference |
| Experience | 30% | Years vs requirement (zeroed if skill gate fails) |
| BERT Similarity | 20% | Semantic similarity via all-MiniLM-L6-v2 |
| Education | 10% | Degree level vs requirement |

**Skill Gate:** If skill match < 30%, experience and education scores are zeroed. If 0% overlap, final score is capped at 25.

## AI Commentary (Optional)
```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2:3b
```
Then use the "Generate AI Commentary" button in the candidate card.

## Environment Variables
```
SENDER_EMAIL=yourname@gmail.com
SENDER_APP_PASSWORD=xxxx xxxx xxxx xxxx
```