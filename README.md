# Resume Screening System

An AI-powered Applicant Tracking System (ATS) that automatically screens resumes against job descriptions, ranks candidates with an explainable fit score, and handles the full HR workflow — from upload to rejection email — through a modern web interface.

---

## Dashboard

<!-- Paste dashboard screenshot here -->
![Dashboard Screenshot](assets/dashboard.png)
![Dashboard Screenshot](assets/result.png)
---

## Architecture

```
Next.js (Vercel)  ──→  Flask REST API (HuggingFace Docker Space)  ──→  Supabase (PostgreSQL)
     ↑                           ↓
  HR Interface            BERT + spaCy + sklearn
```

The frontend and backend are fully decoupled. Next.js on Vercel handles all UI. Flask on HuggingFace handles all processing and exposes a pure JSON API. Supabase provides persistent PostgreSQL storage shared across both.

---

## Features

- **Multi-format resume ingestion** — PDF, DOCX, TXT, MD
- **8-layer NLP pipeline** — extraction, cleaning, tokenization, lemmatization via spaCy
- **Skill taxonomy matching** — 80+ skills across 10 domains with alias inference
- **BERT semantic similarity** — `all-MiniLM-L6-v2` for meaning-level JD matching
- **Skill gate** — if skill overlap is below threshold, experience score is zeroed to prevent cross-domain false positives
- **Weighted ranking engine** — four dimensions (skill, experience, BERT similarity, education) with HR-configurable weights per role
- **Explainability dashboard** — per-candidate score breakdown showing exactly why they ranked where they did
- **Rule-based feedback** — automatic strengths, gaps, and improvement suggestions for every candidate
- **Gmail email integration** — one-click rejection and shortlist emails with HTML templates, duplicate-send protection
- **Analytics** — hiring funnel, top skills found, score distribution across all screened candidates
- **CSV export** — ranked candidate list downloadable for offline use
- **Full persistence** — all screening results, jobs, and email logs stored in PostgreSQL across sessions

---

## Scoring

| Component | Default Weight | Method |
|---|---|---|
| Skill Match | 40% | Taxonomy matching with aliases and inference |
| Experience | 30% | Years vs. requirement (zeroed if skill gate fails) |
| BERT Similarity | 20% | Semantic cosine similarity via `all-MiniLM-L6-v2` |
| Education | 10% | Degree level vs. required (scale 1–5) |

**Skill gate:** if skill match falls below 30%, experience and education scores are zeroed. If overlap is 0%, final score is hard-capped at 25. This prevents a candidate with strong but irrelevant credentials from scoring as a false positive.

Weights are configurable per role from the frontend UI.

**Score labels:**
- **Fit** — score ≥ 70
- **Maybe** — score 45–69
- **Not Fit** — score < 45

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, Tailwind CSS |
| Hosting (frontend) | Vercel |
| Backend | Flask, Gunicorn (1 worker, 4 threads) |
| Hosting (backend) | HuggingFace Docker Space |
| NLP | spaCy `en_core_web_sm` |
| Semantic similarity | sentence-transformers `all-MiniLM-L6-v2` |
| Classic ML | scikit-learn (TF-IDF, LinearSVC) |
| File parsing | pdfminer.six, python-docx |
| Database | PostgreSQL via Supabase |
| Email | smtplib (Gmail SMTP) |
| Containerisation | Docker |

---

## Project Structure

```
backend/                        # Flask REST API
├── app.py                      # All /api/* routes — pure JSON, no templates
├── layer1_ingestion.py         # PDF / DOCX / TXT / MD file reading
├── layer2_preprocessing.py     # spaCy cleaning, tokenization, section detection
├── layer3_features.py          # TF-IDF vectorization + BERT similarity
├── layer4_extraction.py        # Contact, skills, experience, education extraction
├── layer5_model.py             # Weighted ranker + skill gate + feedback engine
├── layer6_database.py          # PostgreSQL (Supabase) — jobs, candidates, email log
├── layer8_email.py             # Gmail SMTP rejection / shortlist emails
├── Dockerfile                  # Python 3.11-slim, CPU torch, spaCy, BERT baked in
├── requirements.txt
└── .env                        # DATABASE_URL, SENDER_EMAIL, FRONTEND_URL (not committed)

frontend/                       # Next.js app
├── app/
│   ├── page.tsx                # Screen resumes — upload form + weight sliders
│   ├── progress/[jobId]/       # Live screening progress with DB polling
│   ├── candidates/[jobId]/     # Ranked results, score breakdown, email actions
│   └── analytics/[jobId]/      # Funnel, top skills, score distribution
├── components/
│   ├── Navbar.tsx
│   └── ui.tsx                  # Shared design system components
├── lib/api.ts                  # Single API layer — swap mock ↔ real with one env var
└── .env.local                  # NEXT_PUBLIC_API_URL (points to HuggingFace Space)
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| GET | `/api/jobs` | List all screening jobs |
| GET | `/api/jobs/<id>` | Get single job |
| POST | `/api/screen` | Upload resumes + JD, start async screening |
| GET | `/api/status/<job_id>` | Poll screening progress |
| GET | `/api/candidates/<job_id>` | Get ranked candidates (supports `?filter=Fit`) |
| GET | `/api/analytics/<job_id>` | Get analytics for a job |
| POST | `/api/send_email/<id>/<type>` | Send rejection or shortlist email |
| POST | `/api/ai_feedback/<id>` | Generate AI commentary (requires Ollama) |
| GET | `/api/export_csv/<job_id>` | Download candidates as CSV |

---

## Local Development

**Backend:**
```bash
cd backend
docker compose up --build
# API running at http://localhost:7860
```

**Frontend:**
```bash
cd frontend
npm install
# Set NEXT_PUBLIC_API_URL=http://localhost:7860 in .env.local
npm run dev
# UI at http://localhost:3000
```

**Environment variables (backend `.env`):**
```
DATABASE_URL=postgresql://postgres.xxx:password@pooler.supabase.com:5432/postgres
SENDER_EMAIL=yourname@gmail.com
SENDER_APP_PASSWORD=xxxx xxxx xxxx xxxx
SECRET_KEY=your-secret-key
FRONTEND_URL=https://your-app.vercel.app
```

---

## Deployment

| Component | Platform | URL |
|---|---|---|
| Frontend | Vercel | `https://resume-screener-mu-ten.vercel.app` |
| Backend API | HuggingFace Docker Space | `https://akay4477-resume-screening.hf.space` |
| Database | Supabase | PostgreSQL (ap-south-1) |

Environment variables for the backend are set as **HuggingFace Space Secrets** (Settings → Variables and Secrets). The Vercel frontend reads `NEXT_PUBLIC_API_URL` set in Vercel's environment variable dashboard.

---

## Known Limitations

- HuggingFace free tier sleeps after 48 hours of inactivity — first request after sleep takes ~60 seconds to cold-start
- BERT scoring adds 2–5 seconds per resume on CPU
- AI commentary requires Ollama running locally and is not available in the hosted version
- Gmail SMTP limited to 500 emails/day on free accounts

---

## License

Internal project — AIML Internship.