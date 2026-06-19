"""
layer2_preprocessing.py — Layer 2: Preprocessing Pipeline
Cleans raw resume text, detects sections, tokenizes, lemmatizes.
"""

import re
import spacy
from typing import Any

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model not found. Install with:\n"
        "pip install https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
    )

SECTION_KEYWORDS = {
    "experience":     ["experience", "work history", "employment", "career",
                       "internship", "work experience", "professional experience"],
    "education":      ["education", "academic", "qualification", "degree",
                       "university", "college", "academic details"],
    "skills":         ["skills", "technical skills", "technologies", "tools",
                       "competencies", "technical skill", "professional skill",
                       "key skills", "core competencies"],
    "certifications": ["certifications", "certificates", "licenses",
                       "accreditations", "certification"],
    "projects":       ["projects", "portfolio", "work samples", "project"],
    "summary":        ["summary", "objective", "profile", "about me", "overview"],
    "contact":        ["contact", "email", "phone", "address", "linkedin", "github"],
}

# Special-character skills replaced before stripping to survive cleaning
_SPECIAL_SKILL_PLACEHOLDERS = {
    r"\bc\+\+\b":    "cpp",
    r"\bc\#\b":      "csharp",
    r"\b\.net\b":    "dotnet",
    r"\bnode\.js\b": "nodejs",
    r"\bvue\.js\b":  "vuejs",
    r"\breact\.js\b":"reactjs",
}

USEFUL_POS = {"NOUN", "PROPN", "VERB", "ADJ"}

RESUME_STOPWORDS = {
    "resume", "cv", "curriculum", "vitae", "reference", "references",
    "available", "request", "etc", "also", "well", "highly", "strong",
    "excellent", "proven", "demonstrated", "responsible", "ability",
    "work", "working", "worked", "use", "used", "using",
}


def preprocess_resume(raw_text: str) -> dict[str, Any]:
    clean = _clean_text(raw_text)
    sections = _detect_sections(clean)
    doc = nlp(clean)
    tokens, sentences = _extract_tokens_and_sentences(doc)
    return {
        "raw_text":   raw_text,
        "clean_text": clean,
        "tokens":     tokens,
        "sentences":  sentences,
        "sections":   sections,
        "word_count": len(tokens),
    }


def _clean_text(text: str) -> str:
    ligatures = {
        "\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff",
        "\ufb03": "ffi", "\ufb04": "ffl", "\u2019": "'",
        "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
    }
    for bad, good in ligatures.items():
        text = text.replace(bad, good)

    for pattern, placeholder in _SPECIAL_SKILL_PLACEHOLDERS.items():
        text = re.sub(pattern, placeholder, text, flags=re.IGNORECASE)

    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s\.\,\:\;\-\/\+\#\(\)\@\&\_]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


def _detect_sections(text: str) -> dict[str, str]:
    lines = text.splitlines()
    sections: dict[str, list[str]] = {"other": []}
    current_section = "other"

    for line in lines:
        line_lower = line.lower().strip()
        matched = False
        for section_name, keywords in SECTION_KEYWORDS.items():
            if any(kw in line_lower for kw in keywords) and len(line.split()) <= 6:
                current_section = section_name
                if section_name not in sections:
                    sections[section_name] = []
                matched = True
                break
        if not matched:
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(line)

    return {
        k: "\n".join(v).strip()
        for k, v in sections.items()
        if "\n".join(v).strip()
    }


def _extract_tokens_and_sentences(doc: spacy.tokens.Doc) -> tuple[list[str], list[str]]:
    tokens = []
    for token in doc:
        if token.is_punct or token.is_space or token.is_stop or token.like_num:
            continue
        if len(token.text) < 2:
            continue
        if token.lemma_.lower() in RESUME_STOPWORDS:
            continue
        if token.pos_ not in USEFUL_POS:
            continue
        tokens.append(token.lemma_.lower())

    sentences = [
        sent.text.strip()
        for sent in doc.sents
        if len(sent.text.strip()) > 20
    ]
    return tokens, sentences