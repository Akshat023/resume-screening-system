"""
layer5_model.py — Layer 5: Scoring Engine + Feedback Generator
Scores candidates using skill gate + weighted ranking + BERT similarity.
Generates rule-based feedback and optional AI commentary via Ollama.
"""

import numpy as np
import pickle
import os
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

from layer3_features import FeatureEngine, extract_skills_from_text, compute_bert_similarity

DEFAULT_WEIGHTS = {
    "skill_match": 0.40,
    "experience":  0.30,
    "tfidf_sim":   0.20,
    "education":   0.10,
}

EDUCATION_LABELS = {
    0: "No formal education", 1: "High School", 2: "Diploma",
    3: "Bachelor's degree",   4: "Master's degree", 5: "PhD",
}


class ResumeScorer:
    def __init__(self, weights: dict = None):
        self.weights        = weights or DEFAULT_WEIGHTS.copy()
        self.feature_engine = FeatureEngine()
        self.classifier     = None
        self.label_encoder  = LabelEncoder()
        self.is_trained     = False

    def train(self, preprocessed_list: list[dict], labels: list[str]) -> dict:
        texts = [p["clean_text"] for p in preprocessed_list]
        self.feature_engine.fit(texts)
        encoded_labels = self.label_encoder.fit_transform(labels)
        svm = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")
        self.classifier = CalibratedClassifierCV(svm, cv=3)
        X = self.feature_engine.vectorizer.transform(texts)
        raw_svm   = LinearSVC(C=1.0, max_iter=2000, class_weight="balanced")
        cv_scores = cross_val_score(raw_svm, X, encoded_labels, cv=3, scoring="f1_weighted")
        self.classifier.fit(X, encoded_labels)
        self.is_trained = True
        return {
            "cv_f1_mean": round(float(cv_scores.mean()), 3),
            "cv_f1_std":  round(float(cv_scores.std()), 3),
            "classes":    list(self.label_encoder.classes_),
        }

    def save(self, path: str = "model.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "feature_engine": self.feature_engine,
                "classifier":     self.classifier,
                "label_encoder":  self.label_encoder,
                "weights":        self.weights,
                "is_trained":     self.is_trained,
            }, f)

    def load(self, path: str = "model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.feature_engine = state["feature_engine"]
        self.classifier      = state["classifier"]
        self.label_encoder   = state["label_encoder"]
        self.weights         = state["weights"]
        self.is_trained      = state["is_trained"]

    def score(self, preprocessed: dict, extracted_info: dict,
              job_description: str, job_requirements: dict = None) -> dict:
        job_requirements = job_requirements or {}
        weights     = job_requirements.get("weights", self.weights)
        resume_text = preprocessed["clean_text"]

        if not self.feature_engine.is_fitted:
            self.feature_engine.fit([resume_text, job_description])

        features    = self.feature_engine.transform(resume_text, job_description)
        skill_score = features["skill_match_score"]
        bert_score  = compute_bert_similarity(resume_text, job_description)
        exp_score   = _score_experience(
            extracted_info["experience"]["total_years"],
            job_requirements.get("min_years_experience", 0),
        )
        edu_score   = _score_education(
            extracted_info["education"]["degree_level"],
            job_requirements.get("required_degree_level", 0),
        )

        jd_skills = features["jd_skills"]
        matched   = features["matched_skills"]

        # Skill gate: if skills don't match, experience doesn't matter
        if len(jd_skills) >= 3 and skill_score == 0.0:
            exp_score = edu_score = 0.0
            skill_gate_status = "hard_fail"
        elif len(jd_skills) >= 3 and skill_score < 0.30:
            exp_score = edu_score = 0.0
            skill_gate_status = "soft_fail"
        else:
            skill_gate_status = "pass"

        raw_score   = (
            skill_score * weights["skill_match"] +
            exp_score   * weights["experience"]  +
            bert_score  * weights["tfidf_sim"]   +
            edu_score   * weights["education"]
        )
        final_score = round(min(raw_score * 100, 100), 1)
        if skill_gate_status == "hard_fail":
            final_score = min(final_score, 25.0)

        svm_label = svm_confidence = None
        if self.is_trained:
            vec     = self.feature_engine.transform_tfidf(resume_text)
            encoded = self.classifier.predict(vec)[0]
            proba   = self.classifier.predict_proba(vec)[0]
            svm_label      = self.label_encoder.inverse_transform([encoded])[0]
            svm_confidence = round(float(max(proba)) * 100, 1)

        ranking_label = _score_to_label(final_score)
        breakdown     = {
            "skill_match": round(skill_score * 100, 1),
            "experience":  round(exp_score   * 100, 1),
            "tfidf_sim":   round(bert_score  * 100, 1),
            "education":   round(edu_score   * 100, 1),
        }
        skill_gate_reason = {
            "hard_fail": "⛔ No skill overlap — experience not considered",
            "soft_fail": "⚠️ Insufficient skill match (<30%) — experience not considered",
            "pass":      None,
        }.get(skill_gate_status)

        return {
            "final_score":        final_score,
            "ranking_label":      ranking_label,
            "svm_label":          svm_label,
            "svm_confidence":     svm_confidence,
            "breakdown":          breakdown,
            "weights_used":       weights,
            "matched_skills":     features["matched_skills"],
            "missing_skills":     features["missing_skills"],
            "resume_skills":      features["resume_skills"],
            "jd_skills":          features["jd_skills"],
            "candidate_name":     extracted_info["contact"].get("name"),
            "candidate_email":    extracted_info["contact"].get("email"),
            "experience_years":   extracted_info["experience"]["total_years"],
            "experience_level":   extracted_info["experience"]["level"],
            "highest_degree":     extracted_info["education"]["highest_degree"],
            "certifications":     extracted_info["certifications"],
            "_degree_level":      extracted_info["education"]["degree_level"],
            "skill_gate_status":  skill_gate_status,
            "skill_gate_reason":  skill_gate_reason,
        }

    def score_batch(self, candidates: list[dict], job_description: str,
                    job_requirements: dict = None) -> list[dict]:
        all_texts = [c["preprocessed"]["clean_text"] for c in candidates]
        all_texts.append(job_description)
        self.feature_engine.fit(all_texts)

        results = []
        for candidate in candidates:
            try:
                result = self.score(
                    candidate["preprocessed"],
                    candidate["extracted"],
                    job_description,
                    job_requirements,
                )
                result["filename"] = candidate.get("filename", "unknown")
                results.append(result)
                print(f"[✓] {candidate.get('filename')} → {result['final_score']}/100 ({result['ranking_label']})")
            except Exception as e:
                print(f"[✗] Failed '{candidate.get('filename')}': {e}")

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results

    def generate_feedback(self, score_result: dict, job_description: str,
                          job_requirements: dict = None, role_title: str = "this role") -> dict:
        job_requirements = job_requirements or {}
        strengths = []
        gaps      = []
        suggestions = []

        breakdown      = score_result.get("breakdown", {})
        matched_skills = score_result.get("matched_skills", [])
        missing_skills = score_result.get("missing_skills", [])
        exp_years      = score_result.get("experience_years", 0)
        exp_level      = score_result.get("experience_level", "Unknown")
        highest_degree = score_result.get("highest_degree", "Unknown")
        certifications = score_result.get("certifications", [])
        final_score    = score_result.get("final_score", 0)
        ranking_label  = score_result.get("ranking_label", "Not Fit")
        degree_level   = score_result.get("_degree_level", 0)
        min_exp        = job_requirements.get("min_years_experience", 0)
        min_edu        = job_requirements.get("required_degree_level", 0)
        skill_pct      = breakdown.get("skill_match", 0)
        exp_pct        = breakdown.get("experience", 0)
        tfidf_pct      = breakdown.get("tfidf_sim", 0)
        edu_pct        = breakdown.get("education", 0)

        if skill_pct >= 70 and matched_skills:
            strengths.append(f"Strong technical alignment — matched {len(matched_skills)} required skill(s): {', '.join(matched_skills[:6])}.")
        elif skill_pct >= 40 and matched_skills:
            strengths.append(f"Partial skill match — has {len(matched_skills)} relevant skill(s): {', '.join(matched_skills[:4])}.")
        if exp_pct >= 100:
            strengths.append(f"Meets experience requirement — {exp_years} year(s) ({exp_level}).")
        elif exp_pct >= 70:
            strengths.append(f"Experience close to requirement — {exp_years} year(s) ({exp_level}).")
        if edu_pct >= 100 and degree_level > 0:
            strengths.append(f"Education requirement met — holds a {highest_degree}.")
        if certifications:
            strengths.append(f"Holds {len(certifications)} certification(s): {', '.join(certifications[:3])}.")
        if tfidf_pct >= 60:
            strengths.append("Resume language aligns well with the job description.")

        if missing_skills:
            gaps.append(f"Missing {len(missing_skills)} required skill(s): {', '.join(missing_skills[:5])}.")
        if min_exp > 0 and exp_years < min_exp:
            gaps.append(f"Experience shortfall — {exp_years} yr(s) vs {min_exp} required.")
        elif exp_years == 0:
            gaps.append("No verifiable work experience detected.")
        if min_edu > 0 and degree_level < min_edu:
            gaps.append(f"Education gap — holds {EDUCATION_LABELS.get(degree_level, 'unknown')}, but {EDUCATION_LABELS.get(min_edu, 'required degree')} is required.")
        if tfidf_pct < 25:
            gaps.append("Resume language differs significantly from the job description.")
        if skill_pct == 0 and len(score_result.get("jd_skills", [])) >= 3:
            gaps.append("No technical skill overlap — candidate appears to be from a different domain.")

        if missing_skills:
            suggestions.append(f"Develop proficiency in: {', '.join(missing_skills[:3])}.")
        if not certifications and skill_pct >= 30:
            suggestions.append("Adding industry certifications would strengthen the application.")
        if min_exp > 0 and exp_years < min_exp:
            suggestions.append(f"Needs ~{round(min_exp - exp_years, 1)} more year(s) of experience. Open-source projects or internships can help.")
        if tfidf_pct < 40:
            suggestions.append("Resume should be tailored to mirror the language in the job description.")
        if skill_pct >= 50 and final_score < 50:
            suggestions.append("Technical skills are relevant but score is dragged down by experience/education gaps. Consider as a junior candidate.")
        if final_score >= 70 and not suggestions:
            suggestions.append("Strong match. Recommend proceeding to interview.")

        summary = _build_summary(
            ranking_label, final_score, role_title,
            matched_skills, missing_skills, exp_years, min_exp,
            exp_level, highest_degree, certifications, skill_pct, tfidf_pct,
        )

        return {
            "strengths":   strengths   or ["No notable strengths detected against this JD."],
            "gaps":        gaps        or ["No significant gaps identified."],
            "suggestions": suggestions or ["No specific improvements needed — strong match."],
            "summary":     summary,
            "fit_for_role": ranking_label == "Fit",
            "ai_summary":  None,
        }


def generate_ai_feedback(score_result: dict, rule_based_feedback: dict,
                         job_description: str, role_title: str = "this role") -> str:
    try:
        import requests as _requests
    except ImportError:
        return "Install 'requests' to enable AI feedback: pip install requests"

    name      = score_result.get("candidate_name") or "The candidate"
    score     = score_result.get("final_score", 0)
    label     = score_result.get("ranking_label", "Not Fit")
    matched   = score_result.get("matched_skills", [])
    missing   = score_result.get("missing_skills", [])
    certs     = score_result.get("certifications", [])

    prompt = f"""You are an expert HR consultant reviewing a resume for a {role_title} position.

CANDIDATE: {name} | Score: {score}/100 ({label})
Experience: {score_result.get('experience_years', 0)} years ({score_result.get('experience_level', '')})
Education: {score_result.get('highest_degree', '')}
Matched Skills: {', '.join(matched[:8]) if matched else 'None'}
Missing Skills: {', '.join(missing[:6]) if missing else 'None'}
Certifications: {', '.join(certs[:3]) if certs else 'None'}

Strengths: {' '.join(rule_based_feedback.get('strengths', []))[:300]}
Gaps: {' '.join(rule_based_feedback.get('gaps', []))[:300]}

JD excerpt: {job_description[:500]}

Write a 3-4 sentence professional HR commentary. Focus on overall fit, the most important strength or gap, and a specific next step. Plain paragraph, no bullet points."""

    try:
        response = _requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:3b", "prompt": prompt, "stream": False,
                  "options": {"temperature": 0.3, "num_predict": 200}},
            timeout=90,
        )
        if response.status_code == 200:
            ai_text = response.json().get("response", "").strip()
            if ai_text:
                return ai_text
        return f"Ollama returned status {response.status_code}. Run: ollama pull llama3.2:3b"
    except _requests.exceptions.ConnectionError:
        return "AI commentary unavailable — Ollama is not running. Download from https://ollama.com then run: ollama pull llama3.2:3b"
    except _requests.exceptions.Timeout:
        return "AI commentary timed out. Try again or use a smaller model."
    except Exception as e:
        return f"AI commentary error: {str(e)}"


def _build_summary(ranking_label, final_score, role_title, matched_skills,
                   missing_skills, exp_years, min_exp, exp_level,
                   highest_degree, certifications, skill_pct, tfidf_pct) -> str:
    if ranking_label == "Fit":
        opening = f"This candidate is a strong match for the {role_title} role (score: {final_score}/100)."
    elif ranking_label == "Maybe":
        opening = f"This candidate is a partial match for the {role_title} role (score: {final_score}/100) and may be worth considering."
    else:
        opening = f"This candidate does not meet the requirements for the {role_title} role (score: {final_score}/100)."

    if skill_pct >= 70 and matched_skills:
        skill_comment = f"Technical skills are well-aligned, covering {', '.join(matched_skills[:4])}."
    elif skill_pct >= 30 and matched_skills:
        skill_comment = f"Has some relevant skills ({', '.join(matched_skills[:3])}), but is missing {', '.join(missing_skills[:3]) if missing_skills else 'key requirements'}."
    else:
        skill_comment = "Technical skill set does not align well with this role."

    if min_exp > 0 and exp_years >= min_exp:
        exp_comment = f"Meets the experience requirement with {exp_years} year(s) ({exp_level})."
    elif exp_years > 0:
        exp_comment = f"Has {exp_years} year(s) of {exp_level} experience{(', below the ' + str(min_exp) + '-year requirement') if min_exp > 0 else ''}."
    else:
        exp_comment = "No significant work experience detected."

    closing = {
        "Fit":     "Recommend shortlisting for interview.",
        "Maybe":   "Consider for interview if the shortlist is thin.",
        "Not Fit": "Not recommended for this role.",
    }.get(ranking_label, "")

    return " ".join([opening, skill_comment, exp_comment, f"Education: {highest_degree}.", closing])


def _score_experience(candidate_years: float, required_years: float) -> float:
    if required_years == 0:
        return min(candidate_years / 10.0, 1.0)
    if candidate_years >= required_years:
        return 1.0
    if candidate_years == 0:
        return 0.0
    return round(candidate_years / required_years, 3)


def _score_education(candidate_level: int, required_level: int) -> float:
    if required_level == 0:
        return min(candidate_level / 5.0, 1.0)
    if candidate_level >= required_level:
        return 1.0
    if candidate_level == 0:
        return 0.0
    return round(candidate_level / required_level, 3)


def _score_to_label(score: float) -> str:
    if score >= 70:   return "Fit"
    elif score >= 45: return "Maybe"
    else:             return "Not Fit"