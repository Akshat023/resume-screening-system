---
title: Resume Screening
emoji: 📄
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# Resume Screening System

An AI-powered Applicant Tracking System (ATS) built in Python with Flask.

## Features
- PDF, DOCX, TXT, MD resume support
- BERT semantic similarity scoring (all-MiniLM-L6-v2)
- Skill taxonomy with aliases and inference (80+ skills)
- Skill gate: if skills don't match, experience is not considered
- Rule-based feedback: strengths, gaps, improvement suggestions
- Gmail SMTP email integration
- PostgreSQL (Supabase) persistence
- Analytics dashboard
- CSV export

## Tech Stack
Python, Flask, spaCy, sentence-transformers, scikit-learn, PostgreSQL