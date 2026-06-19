"""
layer4_extraction.py — Layer 4: Information Extraction
Extracts contact, skills, experience, education, certifications from resume text.
"""

import re
import datetime
import spacy
from layer3_features import extract_skills_from_text, get_skill_categories

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Install with:\n"
        "pip install https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
    )

CURRENT_YEAR = datetime.datetime.now().year

DEGREE_LEVELS = {
    "phd": 5, "ph.d": 5, "ph.d.": 5, "doctorate": 5, "doctoral": 5,
    "doctor of philosophy": 5,
    "m.tech": 4, "mtech": 4, "m.e.": 4, "m.e": 4,
    "m.s.": 4, "m.s": 4, "ms": 4,
    "master's": 4, "masters": 4, "master of": 4, "master in": 4,
    "mba": 4, "m.b.a": 4, "m.b.a.": 4,
    "m.sc": 4, "m.sc.": 4, "msc": 4, "m.a.": 4, "m.a": 4,
    "mca": 4, "m.c.a": 4, "m.c.a.": 4,
    "pgdm": 4, "post graduate": 4, "postgraduate": 4,
    "b.tech": 3, "btech": 3, "b.e.": 3, "b.e": 3,
    "b.s.": 3, "b.s": 3, "bs": 3,
    "bachelor's": 3, "bachelors": 3, "bachelor of": 3, "bachelor in": 3,
    "b.sc": 3, "b.sc.": 3, "bsc": 3, "b.a.": 3, "b.a": 3,
    "bca": 3, "b.c.a": 3, "b.c.a.": 3,
    "b.com": 3, "bcom": 3, "b.arch": 3, "b.pharm": 3, "undergraduate": 3,
    "diploma": 2, "associate": 2, "polytechnic": 2,
    "12th": 1, "hsc": 1, "high school": 1, "higher secondary": 1,
    "secondary school": 1, "ssc": 1, "10+2": 1,
}

DEGREE_NAMES = {
    5: "PhD / Doctorate",
    4: "Master's Degree",
    3: "Bachelor's Degree",
    2: "Diploma / Associate",
    1: "High School",
    0: "Not specified",
}

JOB_TITLE_KEYWORDS = [
    "engineer", "developer", "scientist", "analyst", "manager", "lead",
    "architect", "consultant", "specialist", "intern", "associate",
    "director", "head", "officer", "coordinator", "designer", "researcher",
]

CERT_KEYWORDS = [
    "certified", "certification", "certificate", "aws", "gcp", "azure",
    "google", "microsoft", "cisco", "comptia", "pmp", "scrum", "coursera",
    "udemy", "deeplearning.ai", "specialization", "nptel",
]

_NAME_BLOCKLIST = {
    "engineer", "developer", "scientist", "analyst", "manager", "intern",
    "designer", "consultant", "architect", "specialist", "director",
    "machine", "learning", "intelligence", "artificial", "data", "software",
    "computer", "science", "technology", "information", "systems",
    "algorithms", "algorithm", "resume", "curriculum", "vitae",
    "profile", "summary", "objective", "skills", "education", "experience",
}


def extract_resume_info(preprocessed: dict) -> dict:
    raw_text   = preprocessed.get("raw_text", preprocessed.get("clean_text", ""))
    sections   = preprocessed.get("sections", {})
    return {
        "contact":        _extract_contact(raw_text),
        "skills":         _extract_skills(raw_text, sections),
        "experience":     _extract_experience(raw_text, sections),
        "education":      _extract_education(raw_text, sections),
        "certifications": _extract_certifications(raw_text, sections),
    }


def _extract_contact(text: str) -> dict:
    email_match = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,6}", text)
    email = email_match.group(0).rstrip(".,;|>)]") if email_match else None

    phone_match = re.search(
        r"(\+?91[\s\-]?)?[6-9]\d{9}|(\+?1[\s\-]?)?\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4}",
        text
    )
    phone = phone_match.group(0).strip() if phone_match else None

    linkedin_match = re.search(r"linkedin\.com/in/[\w\-]+", text, re.IGNORECASE)
    github_match   = re.search(r"github\.com/[\w\-]+", text, re.IGNORECASE)

    return {
        "name":     _extract_name(text[:500]),
        "email":    email,
        "phone":    phone,
        "linkedin": linkedin_match.group(0) if linkedin_match else None,
        "github":   github_match.group(0) if github_match else None,
    }


def _extract_name(text: str) -> str | None:
    all_lines = [l.strip() for l in text.splitlines() if l.strip()]
    lines = all_lines[:6]

    # Priority 1: single all-caps word (e.g. "AKSHAT")
    for line in lines:
        if (line.isupper() and line.isalpha() and len(line) > 2
                and line.lower() not in _NAME_BLOCKLIST):
            return line.title()

    # Priority 2: all-caps multi-word name (e.g. "PRIYA SHARMA")
    for line in lines:
        words = line.split()
        if (line.isupper() and 2 <= len(words) <= 4
                and all(w.isalpha() for w in words)
                and not any(w.lower() in _NAME_BLOCKLIST for w in words)):
            return line.title()

    # Priority 3: spaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            if not any(w.lower() in _NAME_BLOCKLIST for w in ent.text.split()):
                return ent.text.strip()

    # Priority 4: title-case 2-4 word line
    for line in lines:
        words = line.split()
        if (2 <= len(words) <= 4
                and all(w[0].isupper() for w in words if w)
                and not any(c.isdigit() for c in line)
                and ":" not in line and "|" not in line and "@" not in line
                and not any(w.lower() in _NAME_BLOCKLIST for w in words)):
            return line

    return None


def _extract_skills(text: str, sections: dict) -> dict:
    skills_text    = sections.get("skills", text)
    all_skills     = extract_skills_from_text(text)
    section_skills = extract_skills_from_text(skills_text)
    return {
        "all":          sorted(list(all_skills)),
        "from_section": sorted(list(section_skills)),
        "by_category":  get_skill_categories(all_skills),
        "total_count":  len(all_skills),
    }


def _extract_experience(text: str, sections: dict) -> dict:
    exp_text = sections.get("experience", sections.get("work experience", text))
    years    = _calculate_years_experience(exp_text, text)
    titles   = _extract_job_titles(exp_text)
    return {
        "total_years": years,
        "job_titles":  titles,
        "level":       _experience_level(years),
    }


def _calculate_years_experience(exp_text: str, full_text: str) -> float:
    MONTH_MAP = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    explicit = re.findall(r"(\d+)\+?\s*years?\s+(?:of\s+)?(?:work\s+)?experience", full_text, re.IGNORECASE)
    if explicit:
        return float(max(int(y) for y in explicit))

    explicit_rev = re.findall(r"experience\s+of\s+(\d+)\+?\s*years?", full_text, re.IGNORECASE)
    if explicit_rev:
        return float(max(int(y) for y in explicit_rev))

    _now = datetime.datetime.now()
    normalised = re.sub(
        r"\b(present|current|now|till date|to date)\b",
        f"{_now.strftime('%B')} {CURRENT_YEAR}",
        exp_text, flags=re.IGNORECASE,
    )

    total_months = 0.0
    month_pattern = (
        r"(january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
        r"\s+(20\d{2}|19\d{2})\s*(?:–|-|to)\s*"
        r"(january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
        r"\s+(20\d{2}|19\d{2})"
    )
    for start_mon, start_yr, end_mon, end_yr in re.findall(month_pattern, normalised, re.IGNORECASE):
        sm = MONTH_MAP.get(start_mon.lower(), 1)
        em = MONTH_MAP.get(end_mon.lower(), 1)
        months = (int(end_yr) - int(start_yr)) * 12 + (em - sm)
        if months > 0:
            total_months += min(months, 96)

    for start_str, end_str in re.findall(
        r"(?<!\w)(20\d{2}|19\d{2})\s*(?:–|-|to|till)\s*(20\d{2}|19\d{2})(?!\w)", normalised
    ):
        start, end = int(start_str), int(end_str)
        if end > start:
            total_months += min((end - start) * 12, 96)

    if total_months > 0:
        return round(min(total_months / 12, 40), 1)

    if any(kw in exp_text.lower() for kw in ["intern", "internship", "trainee", "apprentice"]):
        return 0.5

    return 0.0


def _extract_job_titles(text: str) -> list[str]:
    titles = []
    for line in text.splitlines():
        line_lower = line.lower().strip()
        if any(kw in line_lower for kw in JOB_TITLE_KEYWORDS) and len(line.split()) <= 8:
            cleaned = line.strip(" -•|")
            if cleaned:
                titles.append(cleaned)
    seen, unique = set(), []
    for t in titles:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)
    return unique[:5]


def _experience_level(years: float) -> str:
    if years == 0:          return "No experience detected"
    elif years <= 0.5:      return "Fresher / Intern (< 1 yr)"
    elif years < 1:         return "Fresher / Intern"
    elif years < 3:         return "Junior (1-2 yrs)"
    elif years < 6:         return "Mid-level (3-5 yrs)"
    elif years < 10:        return "Senior (6-9 yrs)"
    else:                   return "Lead / Principal (10+ yrs)"


def _extract_education(text: str, sections: dict) -> dict:
    edu_text = sections.get("education", "")
    highest_level, highest_name = _find_highest_degree(edu_text + "\n" + text)

    search_text = edu_text if edu_text else text
    doc = nlp(search_text[:1500])
    institutions = [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ == "ORG" and len(ent.text.split()) >= 2
    ]

    years     = re.findall(r"\b(20\d{2}|19\d{2})\b", search_text)
    grad_year = max([int(y) for y in years]) if years else None

    gpa_match = re.search(r"gpa[:\s]+(\d+\.?\d*)\s*/\s*(\d+\.?\d*)", search_text.lower())
    gpa = f"{gpa_match.group(1)}/{gpa_match.group(2)}" if gpa_match else None

    return {
        "highest_degree":  highest_name,
        "degree_level":    highest_level,
        "institutions":    institutions[:3],
        "graduation_year": grad_year,
        "gpa":             gpa,
    }


def _find_highest_degree(text: str) -> tuple[int, str]:
    text_lower    = text.lower()
    highest_level = 0
    highest_name  = "Not specified"

    for keyword in sorted(DEGREE_LEVELS.keys(), key=len, reverse=True):
        level = DEGREE_LEVELS[keyword]
        if level <= highest_level:
            continue
        pattern = r"\b" + re.escape(keyword) + r"\b" if len(keyword) <= 3 else re.escape(keyword)
        if re.search(pattern, text_lower):
            highest_level = level
            highest_name  = DEGREE_NAMES[level]

    return highest_level, highest_name


def _extract_certifications(text: str, sections: dict) -> list[str]:
    cert_text = sections.get("certifications", "")
    if not cert_text:
        return [
            line.strip(" -•|")
            for line in text.splitlines()
            if any(kw in line.lower() for kw in CERT_KEYWORDS)
            and len(line.strip()) > 5
        ][:8]

    return [
        line.strip(" -•|")
        for line in cert_text.splitlines()
        if line.strip() and len(line.strip()) > 5
    ][:8]