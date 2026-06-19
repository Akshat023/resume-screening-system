# 🚀 AI-Powered Resume Screening & Candidate Ranking System

An enterprise-style Applicant Tracking System (ATS) that automates resume screening using NLP, semantic matching, candidate ranking, analytics, and email automation.

Built as an end-to-end recruitment solution to help HR teams identify the most relevant candidates faster while maintaining explainability and consistency in hiring decisions.

---

## 📌 Overview

Recruiters often spend hours manually reviewing resumes, comparing qualifications, and communicating with candidates.

This system streamlines the process by:

* Parsing resumes from multiple formats
* Extracting candidate information automatically
* Matching resumes against job descriptions
* Generating candidate fit scores
* Ranking applicants
* Providing explainable feedback
* Automating acceptance and rejection emails

---

## ✨ Key Features

### Resume Processing

* PDF, DOCX, TXT, and Markdown support
* Automated text extraction
* Robust preprocessing pipeline

### NLP & Semantic Matching

* spaCy-based preprocessing
* BERT semantic similarity using `all-MiniLM-L6-v2`
* Skill extraction and normalization
* Domain-aware skill taxonomy

### Intelligent Candidate Scoring

* Skill-based matching
* Experience evaluation
* Education matching
* Semantic similarity scoring
* Weighted ranking system

### Explainable AI

* Score breakdown by component
* Candidate strengths identification
* Skill gap analysis
* Improvement recommendations

### HR Dashboard

* Streamlit-based recruiter interface
* Candidate ranking view
* Detailed candidate profiles
* Search and filtering

### Email Automation

* Acceptance emails
* Rejection emails
* SMTP integration
* Delivery status tracking
* Audit logging

### Analytics & Reporting

* Hiring funnel visualization
* Top skill analysis
* Score distribution insights
* CSV export functionality

---

## 🏗️ System Architecture

The application follows a modular 8-layer architecture.

| Layer   | Component              | Purpose                               |
| ------- | ---------------------- | ------------------------------------- |
| Layer 1 | Resume Ingestion       | Extract text from PDF, DOCX, TXT      |
| Layer 2 | NLP Preprocessing      | Cleaning, tokenization, normalization |
| Layer 3 | Feature Engineering    | Skill matching + BERT embeddings      |
| Layer 4 | Information Extraction | Name, email, education, experience    |
| Layer 5 | Scoring Engine         | Candidate evaluation and ranking      |
| Layer 6 | Database Layer         | SQLite persistence                    |
| Layer 7 | Streamlit Dashboard    | Recruiter-facing application          |
| Layer 8 | Email Automation       | Candidate communication               |

---

## ⚙️ Technology Stack

### Programming Language

* Python

### NLP & Machine Learning

* spaCy
* Sentence Transformers
* BERT (all-MiniLM-L6-v2)
* Scikit-learn

### Data Processing

* Pandas
* NumPy

### Resume Parsing

* pdfminer.six
* python-docx

### Database

* SQLite

### Web Application

* Streamlit

### Email Services

* Gmail SMTP
* smtplib

### Optional AI Commentary

* Ollama
* Llama 3.2

---

## 📊 Candidate Scoring Methodology

Candidates are evaluated using a weighted scoring framework.

| Component                | Weight |
| ------------------------ | ------ |
| Skill Match              | 40%    |
| Experience Match         | 30%    |
| BERT Semantic Similarity | 20%    |
| Education Match          | 10%    |

### Skill Gate Logic

To prevent irrelevant candidates from receiving inflated scores:

* If skill overlap is below threshold, experience scoring is reduced.
* If no meaningful skill overlap exists, the final score is capped.
* Candidates are classified as:

  * Fit
  * Maybe
  * Not Fit

---

## 📂 Project Structure

```text
resume-screening-system/

├── layer1_ingestion.py
├── layer2_preprocessing.py
├── layer3_features.py
├── layer4_extraction.py
├── layer5_model.py
├── layer6_database.py
├── layer7_app.py
├── layer8_email.py

├── requirements.txt
├── setup.bat
├── README.md
└── .gitignore
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/Akshat023/resume-screening-system.git
cd resume-screening-system
```

### Install Dependencies

```bash
setup.bat
```

or

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## ▶️ Running the Application

```bash
streamlit run layer7_app.py
```

Open:

```text
http://localhost:8501
```

---

## 📧 Email Configuration

Create a `.env` file:

```env
SENDER_EMAIL=your_email@gmail.com
SENDER_APP_PASSWORD=your_app_password
```

The system supports:

* Candidate acceptance emails
* Candidate rejection emails
* Delivery logging
* Duplicate-send prevention

---

## 🤖 Optional AI Commentary

Install Ollama:

```bash
ollama pull llama3.2:3b
```

Enable AI-generated candidate insights directly from the dashboard.

---

## 📈 Validation Results

### AI/ML Role Screening

| Candidate   | Score | Status |
| ----------- | ----- | ------ |
| Candidate A | 77.3  | Fit    |
| Candidate B | 70.3  | Fit    |

### Sales Role Screening

| Candidate   | Score | Status  |
| ----------- | ----- | ------- |
| Candidate A | 8.4   | Not Fit |
| Candidate B | 10.9  | Not Fit |

These results demonstrate effective role-specific candidate ranking and filtering.

---

## 🔮 Future Enhancements

* OCR support for scanned resumes
* PostgreSQL integration
* Cloud deployment
* Multi-language resume parsing
* LLM-powered candidate summaries
* Interview question generation
* Recruiter feedback learning loop
* Enterprise ATS integrations

---

## 👨‍💻 Author

**Akshat**

AI/ML Intern

Built as an industry-oriented recruitment automation project demonstrating practical applications of NLP, Machine Learning, Semantic Search, Explainable AI, and Full-Stack Software Engineering.

---

## 📄 License

This project is released under the MIT License.
