"""
layer3_features.py — Layer 3: Feature Engineering
Skill extraction with aliases + inference, TF-IDF vectorization, BERT similarity.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SKILLS_TAXONOMY = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go",
        "rust", "r", "scala", "kotlin", "swift", "php", "ruby", "matlab"
    ],
    "ml_ai": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "tensorflow", "pytorch", "keras", "scikit-learn",
        "xgboost", "lightgbm", "hugging face", "transformers", "bert", "llm",
        "reinforcement learning", "neural network", "random forest", "svm",
        "regression", "classification", "clustering", "opencv", "yolo"
    ],
    "data_engineering": [
        "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
        "spark", "hadoop", "kafka", "airflow", "dbt", "etl", "data pipeline",
        "snowflake", "bigquery", "redshift", "pandas", "numpy"
    ],
    "cloud_devops": [
        "aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ci/cd",
        "jenkins", "github actions", "linux", "bash", "ansible", "helm"
    ],
    "web_backend": [
        "fastapi", "flask", "django", "node.js", "express", "rest api",
        "graphql", "microservices", "spring boot", "react", "vue", "angular"
    ],
    "sales_business": [
        "sales", "business development", "lead generation", "crm",
        "negotiation", "client acquisition", "account management",
        "revenue growth", "market research", "salesforce", "hubspot",
        "digital marketing", "cold calling", "b2b", "b2c",
        "upselling", "cross-selling", "customer success"
    ],
    "marketing": [
        "marketing", "seo", "sem", "content marketing", "social media marketing",
        "email marketing", "brand management", "campaign management",
        "google ads", "facebook ads", "copywriting", "product marketing"
    ],
    "finance_accounting": [
        "financial analysis", "accounting", "budgeting", "forecasting", "excel",
        "financial modeling", "tally", "balance sheet", "p&l", "taxation",
        "audit", "cost accounting", "erp", "sap", "quickbooks"
    ],
    "hr_operations": [
        "recruitment", "talent acquisition", "onboarding", "payroll", "hris",
        "performance management", "employee relations", "training", "hr operations"
    ],
}

SKILL_ALIASES: dict[str, list[str]] = {
    "sql":              ["mysql", "postgresql", "sqlite", "mssql", "ms sql",
                         "sql server", "mariadb", "oracle sql", "pl/sql", "t-sql"],
    "python":           ["python3", "python 3"],
    "aws":              ["amazon web services", "amazon aws"],
    "gcp":              ["google cloud", "google cloud platform"],
    "azure":            ["microsoft azure", "azure cloud"],
    "tensorflow":       ["tf", "tensorflow2", "tensorflow 2"],
    "pytorch":          ["torch", "pytorch lightning"],
    "scikit-learn":     ["sklearn", "scikit learn"],
    "hugging face":     ["huggingface", "hf transformers"],
    "xgboost":          ["xgb"],
    "keras":            ["tf.keras"],
    "nlp":              ["text mining", "text analytics", "text processing"],
    "bert":             ["roberta", "distilbert", "albert", "xlnet"],
    "llm":              ["large language model", "chatgpt", "gpt-4", "openai api", "generative ai"],
    "kubernetes":       ["k8s", "kube"],
    "ci/cd":            ["cicd", "ci cd", "continuous integration", "continuous deployment",
                         "github actions", "gitlab ci"],
    "docker":           ["containerization", "containers"],
    "javascript":       ["js", "es6", "ecmascript"],
    "typescript":       ["ts"],
    "node.js":          ["nodejs", "node js", "node"],
    "react":            ["reactjs", "react.js", "react js"],
    "angular":          ["angularjs", "angular.js"],
    "vue":              ["vuejs", "vue.js"],
    "django":           ["django rest framework", "drf"],
    "flask":            ["flask api"],
    "fastapi":          ["fast api"],
    "pandas":           ["dataframe"],
    "elasticsearch":    ["elastic search", "elk", "opensearch"],
    "mongodb":          ["mongo", "mongo db"],
    "spark":            ["apache spark", "pyspark"],
    "kafka":            ["apache kafka"],
    "airflow":          ["apache airflow"],
    "machine learning": ["ml"],
    "deep learning":    ["dl"],
    "computer vision":  ["cv", "image recognition"],
    "random forest":    ["rf"],
    "c++":              ["cpp", "c plus plus"],
    "c#":               ["csharp", "c sharp", "dotnet", ".net"],
    "excel":            ["microsoft excel", "ms excel"],
    "sap":              ["sap erp", "sap hana"],
    "salesforce":       ["sfdc"],
}

# Skill inference — if candidate has X, automatically credit Y
# Only near-universal relationships (>90% of practitioners)
SKILL_INFERENCE = {
    "python":           ["numpy", "pandas"],
    "machine learning": ["classification", "regression", "clustering"],
    "deep learning":    ["neural network", "classification"],
    "nlp":              ["classification"],
    "tensorflow":       ["deep learning", "neural network", "classification"],
    "pytorch":          ["deep learning", "neural network", "classification"],
    "scikit-learn":     ["machine learning", "classification", "regression", "clustering"],
    "sql":              ["postgresql", "mysql"],
    "spark":            ["etl", "data pipeline"],
    "kubernetes":       ["docker"],
}

# Reverse alias lookup: alias → canonical (all lowercase)
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in SKILL_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias.lower()] = _canonical.lower()

_SORTED_ALIASES = sorted(_ALIAS_TO_CANONICAL.keys(), key=len, reverse=True)

ALL_SKILLS = [
    skill
    for category in SKILLS_TAXONOMY.values()
    for skill in category
]


class FeatureEngine:
    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.is_fitted = False

    def fit(self, texts: list[str]) -> "FeatureEngine":
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform_tfidf(self, text: str) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform_tfidf().")
        return self.vectorizer.transform([text])

    def transform(self, resume_text: str, job_description: str) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transform().")

        resume_vec = self.vectorizer.transform([resume_text])
        jd_vec     = self.vectorizer.transform([job_description])
        tfidf_sim  = float(cosine_similarity(resume_vec, jd_vec)[0][0])

        jd_skills     = extract_skills_from_text(job_description)
        resume_skills = extract_skills_from_text(resume_text)
        matched       = [s for s in jd_skills if s in resume_skills]
        missing       = [s for s in jd_skills if s not in resume_skills]
        skill_score   = len(matched) / len(jd_skills) if jd_skills else 0.0

        return {
            "tfidf_vector":      resume_vec,
            "tfidf_similarity":  round(tfidf_sim, 4),
            "skill_match_score": round(skill_score, 4),
            "matched_skills":    matched,
            "missing_skills":    missing,
            "jd_skills":         jd_skills,
            "resume_skills":     list(resume_skills),
        }


def _normalise_aliases(text: str) -> str:
    for alias in _SORTED_ALIASES:
        canonical = _ALIAS_TO_CANONICAL[alias]
        pattern = r"\b" + re.escape(alias) + r"\b"
        text = re.sub(pattern, canonical, text, flags=re.IGNORECASE)
    return text


def extract_skills_from_text(text: str) -> set[str]:
    normalised = _normalise_aliases(text.lower())
    found = set()
    for skill in ALL_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, normalised):
            found.add(skill)

    # Apply skill inference
    inferred = set()
    for skill in found:
        for implied in SKILL_INFERENCE.get(skill, []):
            if implied not in found:
                inferred.add(implied)
    found.update(inferred)
    return found


def get_skill_categories(skills: set[str]) -> dict[str, list[str]]:
    return {
        category: [s for s in category_skills if s in skills]
        for category, category_skills in SKILLS_TAXONOMY.items()
        if any(s in skills for s in category_skills)
    }


# BERT semantic similarity
_BERT_MODEL = None


def _get_bert_model():
    global _BERT_MODEL
    if _BERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print("[Layer 3] Loading BERT model (all-MiniLM-L6-v2)...")
        _BERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Layer 3] BERT model ready.")
    return _BERT_MODEL


def compute_bert_similarity(text1: str, text2: str) -> float:
    try:
        model = _get_bert_model()
        embeddings = model.encode([text1[:2000], text2[:2000]], convert_to_tensor=False)
        sim = float(cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0])
        return round(max(0.0, min(sim, 1.0)), 4)
    except Exception as e:
        print(f"[Layer 3] BERT similarity failed: {e}. Returning 0.0.")
        return 0.0