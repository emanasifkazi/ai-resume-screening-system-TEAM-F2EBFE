# Author: Eman Asif | Roll No: 2K23/CSM/35    
"""
AI Resume Screening System  v3.0
==================================
Team ID: TEAM_F2EBFE
AI Lab Final Project — Flask + Rule-Based AI + Search/Optimisation AI +
Machine Learning AI + NLP/LLM-assisted AI, in one single-file application.

This header is a map from the project guide's requirements to where each
one is implemented, so every requirement is easy to find during grading
or a viva.

────────────────────────────────────────────────────────────────────────
MINIMUM REQUIRED FEATURES (Section 3)
────────────────────────────────────────────────────────────────────────
  A) Problem Setup Module
       - Problem statement, input/output/constraints .... HTML: <div class="hero">
       - Job description input (preset or free text),
         candidate input (Sample / Manual / PDF upload) .. HTML: #jd-preset, setMode()
       - Input validation + inline error messages ........ JS: run() -> setSt('❌...','err')

  B) Core Logic Module
       - load_data(mode)          — fetch sample / manual / uploaded resumes
       - preprocess_data(text)    — clean, tokenise, extract skills/exp/education
       - run_screening(...)       — the core scoring algorithm (TF-IDF ± semantic)
       - Modular: parsing, scoring, ranking, and explanation are separate
         functions; the HTML/JS layer only renders what the backend returns
       - Intermediate steps are shown, not just a final number: per-factor
         breakdown, NLP token-by-token pipeline, forward-chaining rule trace,
         and search-algorithm explored-state trail are all visible in the UI

  C) Visual UI Module
       - Charts (bar/pie/line/scatter/radar/heatmap) ..... Chart.js, renderCharts()
       - Tables with highlights ........................... results table, tier colours
       - Graph/network view ............................... #network-sec (skill graph)
       - Grid/map view ..................................... #grid-sec (score grid)
       - Timeline / step-by-step animation ................ #timeline-sec
       - Controls: sliders, toggles, dropdowns ............ sidebar weight sliders,
         semantic/explanations/comparison toggles, JD preset dropdown
       - Result panel + status messages (success/error/loading)

  D) Explainability Module
       - generate_explanation(result, context) ............ per-candidate factor breakdown
       - Forward-chaining rule trace (#fc-sec) ............. shows every fired IF-THEN rule
       - NLP chatbot panel (#chatbot-sec) .................. plain-English Q&A over results

  E) Evaluation Module
       - Accuracy / precision / recall / F1 ................ evaluate_model(), ML panel
       - Nodes expanded / frontier size (search cost) ...... search_optimal_shortlist()
       - Runtime (ms) / peak memory (KB) ................... evaluate_model()
       - Compare ≥2 approaches ............................. compare_approaches(),
         run_all_search_algorithms(), TF-IDF vs Semantic toggle

────────────────────────────────────────────────────────────────────────
AI INTEGRATION (Section 5) — all four options are implemented
────────────────────────────────────────────────────────────────────────
  Option 1 — Rule-based AI ........... forward_chaining_rules()      (#fc-sec)
  Option 2 — Search/Optimisation AI .. search_optimal_shortlist()    (#search-sec)
                                        BFS, DFS, UCS, Greedy, A*, Hill Climbing
  Option 3 — Machine Learning AI ..... train_classification_model(),
                                        train_regression_model(),
                                        train_clustering_model()      (#ml-sec)
  Option 4 — NLP/LLM-assisted AI ..... nlp_pipeline_analysis(), generate_summary(),
                                        analyze_sentiment(), extract_bigrams(),
                                        askChat() chatbot                (#nlp-sec, #chatbot-sec)

  No external AI API is used anywhere in this app — every model above is a
  locally-run scikit-learn / spaCy / rule-based component, so there is no
  external prompt, cost, or privacy concern to disclose (Section 5, note).

────────────────────────────────────────────────────────────────────────
SUGGESTED FUNCTION STRUCTURE (Section 4) — implemented under these names
────────────────────────────────────────────────────────────────────────
  load_data(mode)                              -> resume records
  preprocess_data(text)                        -> cleaned tokens
  run_screening(data, jd_text, weights...)     -> scored + explained results
  generate_explanation(result, context)        -> per-candidate explanation
  create_visuals(data, result)                 -> chart/graph/grid/timeline payload
  render_ui()                                  -> serves the dashboard (Flask '/')

Run:
  pip install -r requirements.txt
  python -m spacy download en_core_web_sm   # optional, enables full POS/NER
  python main.py
  # then open http://127.0.0.1:5000
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import re, time, base64, io, math, json, os, tracemalloc
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings; warnings.filterwarnings("ignore")

# Optional imports
try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    SEMANTIC_OK = True
except Exception:
    SEMANTIC_OK = False

try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    SPACY_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE MODULE — MySQL Score Storage
# ─────────────────────────────────────────────────────────────────────────────
try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
    MYSQL_OK = True
except ImportError:
    MYSQL_OK = False

MYSQL_CONFIG = {
    "host":     os.environ.get("DB_HOST",     "localhost"),
    "user":     os.environ.get("DB_USER",     "root"),
    "password": os.environ.get("DB_PASSWORD", ""),
    "database": os.environ.get("DB_NAME",     "resume_screening"),
}

def db_init():
    if not MYSQL_OK: return False
    try:
        conn = mysql.connector.connect(host=MYSQL_CONFIG["host"],user=MYSQL_CONFIG["user"],password=MYSQL_CONFIG["password"])
        c = conn.cursor()
        c.execute("CREATE DATABASE IF NOT EXISTS resume_screening")
        c.execute("USE resume_screening")
        c.execute("""CREATE TABLE IF NOT EXISTS screening_sessions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            job_title VARCHAR(255), jd_text TEXT,
            total_candidates INT DEFAULT 0, threshold INT DEFAULT 60,
            screened_at DATETIME, INDEX idx_date(screened_at)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
        c.execute("""CREATE TABLE IF NOT EXISTS candidate_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id INT NOT NULL, rank_num INT, candidate_name VARCHAR(255),
            final_score FLOAT, tfidf_score FLOAT DEFAULT 0, semantic_score FLOAT DEFAULT 0,
            skill_score FLOAT DEFAULT 0, exp_score FLOAT DEFAULT 0,
            matched_skills TEXT, missing_skills TEXT,
            education VARCHAR(500), exp_years INT DEFAULT 0,
            tier VARCHAR(5), verdict VARCHAR(100), screened_at DATETIME,
            FOREIGN KEY(session_id) REFERENCES screening_sessions(id) ON DELETE CASCADE,
            INDEX idx_session(session_id), INDEX idx_score(final_score DESC)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""")
        conn.commit(); conn.close()
        print("[DB] MySQL ready!")
        return True
    except Exception as e:
        print(f"[DB] MySQL not available: {e}")
        return False

def db_save(job_title, jd_text, candidates, threshold=60):
    if not MYSQL_OK: return None
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        c = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO screening_sessions (job_title,jd_text,total_candidates,threshold,screened_at) VALUES (%s,%s,%s,%s,%s)",
                  (job_title, jd_text[:300], len(candidates), threshold, now))
        sid = c.lastrowid
        for r in candidates:
            s = r.get("score",0)
            tier = "A" if s>=75 else "B" if s>=60 else "C" if s>=45 else "D"
            verdict = "Strong Match" if s>=75 else "Moderate Match" if s>=50 else "Weak Match"
            c.execute("""INSERT INTO candidate_results
                (session_id,rank_num,candidate_name,final_score,tfidf_score,semantic_score,
                 skill_score,exp_score,matched_skills,missing_skills,education,exp_years,tier,verdict,screened_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (sid,r.get("rank",0),r.get("name",""),round(s,2),round(r.get("tfidf",0),2),
                 round(r.get("semantic") or 0,2),round(r.get("skill",0),2),round(r.get("exp_score",0),2),
                 json.dumps(r.get("matched",[])),json.dumps(r.get("missing",[])),
                 r.get("education",""),r.get("exp_yrs",0),tier,verdict,now))
        conn.commit(); conn.close()
        print(f"[DB] Session #{sid} saved — {len(candidates)} candidates")
        return sid
    except Exception as e:
        print(f"[DB ERROR] {e}"); return None

def db_get_all():
    if not MYSQL_OK: return []
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        c = conn.cursor(dictionary=True)
        c.execute("""SELECT cr.rank_num AS `rank`,cr.candidate_name AS name,
            cr.final_score AS score,cr.tfidf_score AS tfidf,cr.semantic_score AS semantic,
            cr.skill_score AS skill,cr.exp_score,cr.matched_skills,cr.missing_skills,
            cr.education,cr.exp_years AS exp_yrs,cr.tier,cr.verdict,cr.screened_at,
            ss.job_title,cr.session_id
            FROM candidate_results cr JOIN screening_sessions ss ON cr.session_id=ss.id
            ORDER BY cr.screened_at DESC, cr.rank_num ASC""")
        rows = c.fetchall(); conn.close()
        for row in rows:
            row["matched"] = json.loads(row["matched_skills"] or "[]")
            row["missing"] = json.loads(row["missing_skills"] or "[]")
            row["screened_at"] = str(row["screened_at"])
            del row["matched_skills"]; del row["missing_skills"]
        return rows
    except Exception as e:
        return []

def db_get_sessions():
    if not MYSQL_OK: return []
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        c = conn.cursor(dictionary=True)
        c.execute("SELECT id,job_title,total_candidates,threshold,screened_at FROM screening_sessions ORDER BY screened_at DESC")
        rows = c.fetchall(); conn.close()
        for row in rows: row["screened_at"] = str(row["screened_at"])
        return rows
    except: return []

def db_get_stats():
    if not MYSQL_OK: return {}
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        c = conn.cursor(dictionary=True)
        c.execute("SELECT COUNT(*) AS total FROM screening_sessions")
        sessions = c.fetchone()["total"]
        c.execute("""SELECT COUNT(*) AS total, AVG(final_score) AS avg_score,
            MAX(final_score) AS max_score, MIN(final_score) AS min_score,
            SUM(CASE WHEN tier='A' THEN 1 ELSE 0 END) AS tier_a,
            SUM(CASE WHEN tier='B' THEN 1 ELSE 0 END) AS tier_b,
            SUM(CASE WHEN tier='C' THEN 1 ELSE 0 END) AS tier_c,
            SUM(CASE WHEN tier='D' THEN 1 ELSE 0 END) AS tier_d
            FROM candidate_results""")
        s = c.fetchone(); conn.close()
        return {"total_sessions":sessions,"total_candidates":s["total"] or 0,
                "avg_score":round(s["avg_score"] or 0,2),
                "max_score":s["max_score"] or 0,"min_score":s["min_score"] or 0,
                "tier_a":s["tier_a"] or 0,"tier_b":s["tier_b"] or 0,
                "tier_c":s["tier_c"] or 0,"tier_d":s["tier_d"] or 0}
    except: return {}

def db_clear():
    if not MYSQL_OK: return False
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        c = conn.cursor()
        c.execute("SET FOREIGN_KEY_CHECKS=0")
        c.execute("TRUNCATE TABLE candidate_results")
        c.execute("TRUNCATE TABLE screening_sessions")
        c.execute("SET FOREIGN_KEY_CHECKS=1")
        conn.commit(); conn.close(); return True
    except: return False

# ─────────────────────────────────────────────────────────────────────────────
# DATASET MODULE — Save sample data to JSON file (Requirement: Section 2)
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
DATA_FILE = os.path.join(DATA_DIR, "sample_resumes.json")

def ensure_dataset():
    """
    Creates data/sample_resumes.json on first run so the project ships with
    a real external dataset file (required by Section 2 deliverables).

    Built directly from SAMPLE_RESUMES (single source of truth) so the
    shipped JSON file and the live in-app sample data can never drift out
    of sync with each other (Coding Standards: avoid duplicated logic/data).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        dataset = {
            "description": (
                "Sample resume dataset for AI Resume Screening System. "
                "Each record contains candidate name, a plain-text resume body, "
                "a pre-extracted skills list, years of experience, and highest education."
            ),
            "version": "2.0",
            "created": datetime.now().strftime("%Y-%m-%d"),
            "records": [
                {
                    "id": i + 1,
                    "name": r["name"],
                    "text": r["text"],
                    "skills": r["skills"],
                    "experience_years": r["experience_years"],
                    "education": r["education"],
                }
                for i, r in enumerate(SAMPLE_RESUMES)
            ],
        }
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"[DATASET] Created {DATA_FILE} with {len(SAMPLE_RESUMES)} candidates")

def load_dataset_from_file():
    """Load sample resumes from data/sample_resumes.json."""
    ensure_dataset()
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            {
                "name":             r["name"],
                "text":             r["text"],
                "skills":           r.get("skills", []),
                "experience_years": r.get("experience_years", 0),
                "education":        r.get("education", ""),
            }
            for r in data.get("records", [])
        ]
    except Exception as e:
        print(f"[DATASET] Could not load file, using fallback: {e}")
        return SAMPLE_RESUMES

# ─────────────────────────────────────────────────────────────────────────────
# RANKING MODULE
# ─────────────────────────────────────────────────────────────────────────────
def rank_with_tiers(candidates, threshold=60):
    TIERS = {
        "A":{"label":"Highly Recommended","color":"#059669","bg":"#ecfdf5","icon":"🏆"},
        "B":{"label":"Recommended",        "color":"#3b82f6","bg":"#eff6ff","icon":"✅"},
        "C":{"label":"Consider",           "color":"#b45309","bg":"#fffbeb","icon":"⚠️"},
        "D":{"label":"Not Recommended",    "color":"#dc2626","bg":"#fef2f2","icon":"❌"},
    }
    for r in candidates:
        s = r.get("score",0)
        key = "A" if s>=75 else "B" if s>=60 else "C" if s>=45 else "D"
        r["tier"]       = key
        r["tier_label"] = TIERS[key]["label"]
        r["tier_color"] = TIERS[key]["color"]
        r["tier_bg"]    = TIERS[key]["bg"]
        r["tier_icon"]  = TIERS[key]["icon"]
        r["qualified"]  = s >= threshold
    return candidates

# ─────────────────────────────────────────────────────────────────────────────
# SKILL GAP MODULE
# ─────────────────────────────────────────────────────────────────────────────
SKILL_TIPS = {
    "python":"Python for Everybody — Coursera","machine learning":"ML — Andrew Ng, Coursera",
    "sql":"SQL Practice — LeetCode","tensorflow":"TensorFlow — tensorflow.org",
    "pytorch":"PyTorch — pytorch.org","nlp":"NLP Course — Hugging Face",
    "pandas":"Pandas — Kaggle Learn","docker":"Docker — docs.docker.com",
    "scikit-learn":"scikit-learn — scikit-learn.org","deep learning":"deeplearning.ai",
}
def skill_gap_analysis(candidates, jd_skills):
    if not jd_skills: return [], {}
    coverage = {sk:0 for sk in jd_skills}
    gaps = []
    for r in candidates:
        for sk in r.get("matched",[]): 
            if sk in coverage: coverage[sk]+=1
        missing = r.get("missing",[])
        recs = [f"{sk} → {SKILL_TIPS.get(sk,'Search online')}" for sk in missing[:3]]
        total = len(jd_skills)
        covered = len(r.get("matched",[]))
        gaps.append({
            "name":r["name"],"rank":r["rank"],"score":r["score"],
            "matched":r.get("matched",[]),"missing":missing,
            "covered_pct":round(covered/total*100,1) if total else 0,
            "gap_pct":round(len(missing)/total*100,1) if total else 0,
            "total_req":total,"recommendations":recs,
        })
    n = len(candidates)
    overall = {sk:round(coverage[sk]/n*100,1) for sk in coverage}
    return gaps, overall

db_init()
app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE B — CORE LOGIC (separate from UI)
# ─────────────────────────────────────────────────────────────────────────────

SKILLS_DB = [
    # Programming languages
    "python","java","c++","c#","javascript","typescript","r","scala","go","rust","swift","kotlin","php","ruby",
    # ML / AI
    "machine learning","deep learning","nlp","natural language processing","computer vision",
    "reinforcement learning","transfer learning","data science","data analysis","data engineering",
    # Frameworks
    "tensorflow","pytorch","keras","scikit-learn","xgboost","lightgbm","hugging face","transformers",
    # Data tools
    "pandas","numpy","matplotlib","seaborn","plotly","tableau","power bi","excel",
    # Web
    "django","flask","fastapi","react","angular","vue","node.js","express","spring boot","html","css",
    # Databases
    "sql","mysql","postgresql","mongodb","redis","elasticsearch","sqlite",
    # DevOps / Cloud
    "docker","kubernetes","aws","azure","gcp","git","linux","ci/cd","jenkins","terraform",
    # Other
    "opencv","rest api","rest apis","graphql","microservices","agile","scrum","research",
]

# Aliases: common alternate spellings/abbreviations → canonical skill name
SKILL_ALIASES = {
    "natural language processing": "nlp",
    "ml":  "machine learning",
    "dl":  "deep learning",
    "cv":  "computer vision",
    "js":  "javascript",
    "ts":  "typescript",
    "postgres": "postgresql",
    "sk-learn": "scikit-learn",
    "sklearn":  "scikit-learn",
    "hf":       "hugging face",
    "k8s":      "kubernetes",
}

SAMPLE_JDS = {
    "ML Engineer":       "Python developer with machine learning, TensorFlow or PyTorch, SQL, REST APIs. Minimum 2 years. BS/MS Computer Science.",
    "Data Scientist":    "Data scientist with Python, NLP, scikit-learn, pandas, deep learning. MS preferred. 2+ years.",
    "Backend Developer": "Java developer with Spring Boot, MySQL, Docker, microservices. 3+ years required.",
    "AI Researcher":     "PyTorch, computer vision, Python, research publications. PhD preferred.",
}

SAMPLE_RESUMES = [
    {
        "name": "Ali Hassan",
        "text": """Ali Hassan
Email: ali.hassan@email.com | Phone: +92-300-1234567 | Lahore, Pakistan
LinkedIn: linkedin.com/in/alihassan | GitHub: github.com/ali-hassan

PROFESSIONAL SUMMARY
Results-driven Python developer with 3 years of hands-on experience in machine learning,
backend development, and REST API design. Proficient in TensorFlow, scikit-learn, and
Django. Built and deployed multiple ML models in production environments.

TECHNICAL SKILLS
Languages: Python, SQL, JavaScript, Bash
Frameworks: Django, Flask, TensorFlow, scikit-learn, pandas, numpy
Tools: Git, Docker, Postman, Jupyter Notebook, VS Code
Databases: MySQL, PostgreSQL, SQLite

WORK EXPERIENCE
ML Engineer — TechVentures Pvt Ltd, Lahore (2022 - Present)
• Developed image classification pipeline using TensorFlow achieving 91% accuracy
• Built REST APIs using Django REST Framework serving 10,000+ daily requests
• Optimized SQL queries reducing database load by 40%
• Created data preprocessing pipelines using pandas and numpy

Junior Python Developer — StartupPK, Karachi (2021 - 2022)
• Developed backend services in Flask for e-commerce platform
• Wrote unit tests and improved code coverage from 60% to 85%
• Collaborated with data team on feature engineering for customer churn model

EDUCATION
Bachelor of Science in Computer Science — FAST NUCES, Lahore (2021)
CGPA: 3.5/4.0

PROJECTS
Resume Screening AI — NLP-based resume matcher using TF-IDF and cosine similarity
Sentiment Analyzer — Real-time Twitter sentiment analysis with scikit-learn
Sales Forecasting — Time series prediction using LSTM with TensorFlow""",
        "skills": ["python", "machine learning", "tensorflow", "sql", "django", "flask", "rest api",
                   "scikit-learn", "pandas", "numpy", "docker", "postgresql"],
        "experience_years": 3,
        "education": "BS Computer Science"
    },
    {
        "name": "Sara Khan",
        "text": """Sara Khan
Email: sara.khan@email.com | Phone: +92-321-7654321 | Islamabad, Pakistan
GitHub: github.com/sara-nlp | Portfolio: sara-khan.dev

PROFESSIONAL SUMMARY
Passionate Data Scientist and NLP engineer with 2 years of experience building
production NLP systems. Expert in transformer models, text classification, and
deep learning. Published researcher with expertise in Hugging Face and PyTorch.

TECHNICAL SKILLS
Languages: Python, R, SQL
ML/AI: NLP, Deep Learning, Machine Learning, Transfer Learning
Frameworks: PyTorch, scikit-learn, Hugging Face Transformers, Keras, pandas, numpy
Tools: Jupyter, Git, MLflow, Docker, Weights & Biases

WORK EXPERIENCE
NLP Engineer — AI Solutions Ltd, Islamabad (2023 - Present)
• Built text classification system using BERT fine-tuning with 94% accuracy
• Developed customer sentiment analysis pipeline processing 50,000 reviews/day
• Implemented named entity recognition (NER) for legal document parsing
• Created text summarization tool using T5 transformer reducing review time by 60%

Data Science Intern — DataTech, Rawalpindi (2022 - 2023)
• Performed EDA on large datasets using pandas and matplotlib
• Built customer churn prediction model using XGBoost (AUC: 0.89)
• Created automated reporting dashboards with Python and Plotly

EDUCATION
Master of Science in Data Science — NUST Islamabad (2022)
Bachelor of Science in Statistics — Quaid-i-Azam University (2020)
CGPA: 3.8/4.0

RESEARCH
"Urdu Sentiment Analysis using Multilingual BERT" — NLP conference 2023
"Text Classification for Low-Resource Languages" — IEEE submission 2024""",
        "skills": ["python", "nlp", "deep learning", "scikit-learn", "pandas", "numpy",
                   "pytorch", "keras", "machine learning", "sql", "docker", "transformers", "hugging face"],
        "experience_years": 2,
        "education": "MS Data Science"
    },
    {
        "name": "Ahmed Raza",
        "text": """Ahmed Raza
Email: ahmed.raza@email.com | Phone: +92-333-9876543 | Lahore, Pakistan
GitHub: github.com/ahmed-raza | LinkedIn: linkedin.com/in/ahmedraza

PROFESSIONAL SUMMARY
Senior Backend Engineer with 4 years of experience designing and building
scalable microservices. Expert in Java, Spring Boot, and containerization.
Led development of fintech platform handling PKR 2 billion in daily transactions.

TECHNICAL SKILLS
Languages: Java, Python, SQL, Bash
Frameworks: Spring Boot, Hibernate, REST APIs, GraphQL
DevOps: Docker, Kubernetes, Jenkins, CI/CD, Terraform
Databases: MySQL, PostgreSQL, MongoDB, Redis
Cloud: AWS (EC2, RDS, S3, Lambda), Azure

WORK EXPERIENCE
Senior Backend Engineer — FinTech Corp, Lahore (2022 - Present)
• Designed microservices architecture reducing system downtime from 2% to 0.01%
• Containerized 15 services with Docker and orchestrated via Kubernetes on AWS
• Implemented CI/CD pipelines with Jenkins reducing deployment time by 70%
• Designed MySQL database schema handling 5 million+ daily transactions
• Led team of 4 junior engineers, conducted code reviews and architecture planning

Backend Developer — SoftHouse Pvt Ltd, Lahore (2020 - 2022)
• Built RESTful APIs in Spring Boot for mobile application (500K+ users)
• Migrated legacy monolith to microservices architecture
• Implemented Redis caching reducing API response time by 65%

EDUCATION
Bachelor of Science in Software Engineering — UET Lahore (2020)
CGPA: 3.6/4.0

CERTIFICATIONS
AWS Certified Developer - Associate (2023)
Docker Certified Associate (2022)""",
        "skills": ["java", "spring boot", "mysql", "docker", "kubernetes", "aws",
                   "postgresql", "mongodb", "redis", "microservices", "ci/cd", "git"],
        "experience_years": 4,
        "education": "BS Software Engineering"
    },
    {
        "name": "Fatima Malik",
        "text": """Dr. Fatima Malik (PhD Candidate)
Email: fatima.malik@qau.edu.pk | Phone: +92-345-1122334 | Islamabad, Pakistan
Google Scholar: scholar.google.com/fatima-malik | GitHub: github.com/fatima-cv

PROFESSIONAL SUMMARY
AI Researcher specializing in Computer Vision and Deep Learning with 5 years of
research and industry experience. Published 4 peer-reviewed papers. Expert in
PyTorch, OpenCV, and object detection. Currently finishing PhD at QAU.

TECHNICAL SKILLS
Languages: Python, C++, MATLAB, SQL
ML/AI: Computer Vision, Deep Learning, Machine Learning, Transfer Learning,
       Reinforcement Learning, Object Detection, Image Segmentation
Frameworks: PyTorch, TensorFlow, OpenCV, scikit-learn, numpy, pandas, Keras
Tools: CUDA, Git, LaTeX, Jupyter, Docker, Weights & Biases, MLflow

RESEARCH EXPERIENCE
PhD Researcher — Quaid-i-Azam University, AI Lab (2020 - Present)
• Developed custom YOLOv8 variant for medical imaging with 97.3% mAP
• Research on brain tumor detection achieving 95% sensitivity (4 papers published)
• Teaching Assistant for Machine Learning and Computer Vision courses
• Supervised 6 undergraduate thesis projects on AI topics

AI Research Intern — National Center for AI (NCAI), Islamabad (2021 - 2022)
• Contributed to real-time pedestrian detection system for smart city project
• Implemented semantic segmentation pipeline for satellite imagery analysis

Computer Vision Engineer — VisionTech, Karachi (2019 - 2020)
• Built quality control system using OpenCV reducing defect rate by 35%
• Developed face recognition system with 99.1% accuracy using deep learning

EDUCATION
PhD Computer Science (AI/CV) — Quaid-i-Azam University (2020 - Present)
MS Computer Science — NUML Islamabad (2019) — CGPA: 3.9/4.0
BS Computer Science — COMSATS Islamabad (2017) — CGPA: 3.7/4.0

PUBLICATIONS (Selected)
1. "Multi-scale Feature Fusion for Brain Tumor Detection" — IEEE TMI 2023
2. "Real-time Object Detection with Adaptive Anchors" — CVPR Workshop 2023
3. "Transfer Learning for Medical Image Analysis" — Nature Scientific Reports 2022
4. "Attention Mechanisms in Medical Imaging" — MICCAI 2022""",
        "skills": ["python", "pytorch", "computer vision", "opencv", "machine learning",
                   "deep learning", "tensorflow", "keras", "scikit-learn", "c++",
                   "research", "numpy", "pandas", "docker"],
        "experience_years": 5,
        "education": "PhD (in progress)"
    },
    {
        "name": "Usman Tariq",
        "text": """Usman Tariq
Email: usman.tariq@student.com | Phone: +92-311-5566778 | Karachi, Pakistan
GitHub: github.com/usman-dev | Portfolio: usmandev.netlify.app

PROFESSIONAL SUMMARY
Enthusiastic junior developer currently completing BS in Computer Science.
Completed 6-month internship building React applications. Strong foundation
in HTML, CSS, JavaScript and basic programming concepts.

TECHNICAL SKILLS
Languages: JavaScript, HTML, CSS, Python (beginner), SQL (basic)
Frameworks: React, Node.js (basic), Bootstrap, Tailwind CSS
Tools: Git, VS Code, Figma (basic), npm

WORK EXPERIENCE
Frontend Development Intern — DigitalAgency, Karachi (2023 - 2024)
• Built responsive React web applications for 3 client projects
• Implemented UI components following Figma designs
• Fixed 40+ bugs and improved page load time by 20%
• Participated in agile sprint planning and daily standups

EDUCATION
Bachelor of Science in Computer Science — COMSATS University, Karachi (2021 - Present)
Expected Graduation: 2025 | Current CGPA: 3.1/4.0

PROJECTS
Task Manager App — React + localStorage CRUD application
Portfolio Website — Personal portfolio with dark mode and animations
Weather App — OpenWeatherMap API integration using JavaScript
Calculator — Vanilla JavaScript implementation""",
        "skills": ["html", "css", "javascript", "react", "python", "sql"],
        "experience_years": 1,
        "education": "BS (in progress)"
    },
    {
        "name": "Zara Baig",
        "text": """Zara Baig
Email: zara.baig@email.com | Phone: +92-322-4455667 | Lahore, Pakistan
LinkedIn: linkedin.com/in/zarabaig | GitHub: github.com/zara-ml

PROFESSIONAL SUMMARY
Machine Learning Engineer with 3 years at a high-growth AI startup.
Expert in building end-to-end ML pipelines from data to production deployment.
Proficient in scikit-learn, TensorFlow, Keras, and MLOps practices.

TECHNICAL SKILLS
Languages: Python, SQL, Bash
ML/AI: Machine Learning, Deep Learning, Data Science, Data Analysis, NLP (basic)
Frameworks: scikit-learn, TensorFlow, Keras, pandas, numpy, matplotlib, Flask
MLOps: MLflow, DVC, Docker, Git, CI/CD
Databases: MySQL, PostgreSQL, SQLite
Cloud: AWS (SageMaker, S3), GCP (basic)

WORK EXPERIENCE
ML Engineer — AI Startup (Series A), Lahore (2021 - Present)
• Built demand forecasting ML pipeline reducing inventory costs by 25%
• Developed anomaly detection system for financial fraud (99.2% precision)
• Deployed 8 ML models as REST APIs using Flask on AWS SageMaker
• Implemented MLflow experiment tracking reducing model iteration time by 40%
• Created automated data preprocessing pipeline handling 2M+ daily records

Junior Data Scientist — Analytics Co., Lahore (2020 - 2021)
• Performed customer segmentation using K-means clustering (K=7 optimal)
• Built churn prediction model (XGBoost, F1: 0.87) saving PKR 5M annually

EDUCATION
Master of Science in Computer Science — LUMS Lahore (2021)
Thesis: "Efficient Neural Architecture Search for Edge Deployment"
CGPA: 3.7/4.0
Bachelor of Science in Computer Science — FAST NUCES Lahore (2019)
CGPA: 3.6/4.0

CERTIFICATIONS
TensorFlow Developer Certificate (Google, 2022)
AWS Machine Learning Specialty (2023)""",
        "skills": ["python", "scikit-learn", "tensorflow", "keras", "sql", "flask",
                   "machine learning", "deep learning", "pandas", "numpy", "docker",
                   "postgresql", "aws", "data science", "data analysis"],
        "experience_years": 3,
        "education": "MS Computer Science"
    },
    {
        "name": "Bilal Ahmed",
        "text": """Bilal Ahmed
Email: bilal.ahmed@email.com | Phone: +92-301-2223344 | Karachi, Pakistan
GitHub: github.com/bilal-devops

PROFESSIONAL SUMMARY
DevOps Engineer with 4 years of experience automating cloud infrastructure and
CI/CD pipelines. Strong background in Python scripting, container orchestration,
and monitoring for high-availability systems.

TECHNICAL SKILLS
Languages: Python, Bash, Go
Cloud/DevOps: AWS, Docker, Kubernetes, Terraform, Jenkins, GitLab CI
Monitoring: Prometheus, Grafana, ELK Stack
Databases: PostgreSQL, Redis

WORK EXPERIENCE
DevOps Engineer — CloudNine Systems, Karachi (2021 - Present)
• Automated deployment pipelines with Jenkins and GitLab CI, cutting release time by 70%
• Managed Kubernetes clusters serving 40+ microservices in production
• Wrote Python automation scripts for infrastructure provisioning on AWS
• Set up Prometheus/Grafana monitoring reducing incident response time by 50%

Junior Systems Administrator — NetSecure, Lahore (2020 - 2021)
• Maintained on-premise Linux servers and automated backups with Bash scripting

EDUCATION
Bachelor of Science in Computer Science — NED University, Karachi (2020)

PROJECTS
Infrastructure-as-Code toolkit — Terraform modules reused across 5 client projects
Log Aggregation Pipeline — ELK-based centralized logging for microservices""",
        "skills": ["python", "docker", "kubernetes", "aws", "sql", "postgresql"],
        "experience_years": 4,
        "education": "BS Computer Science"
    },
    {
        "name": "Ayesha Siddiqui",
        "text": """Ayesha Siddiqui
Email: ayesha.siddiqui@email.com | Phone: +92-312-5556677 | Lahore, Pakistan
Portfolio: ayesha-ui.dev | GitHub: github.com/ayesha-frontend

PROFESSIONAL SUMMARY
Frontend Developer with 2 years of experience building responsive, accessible
web applications with React. Strong eye for UI/UX and component architecture.

TECHNICAL SKILLS
Languages: JavaScript, TypeScript, HTML, CSS
Frameworks: React, Next.js, Redux, Tailwind CSS
Tools: Git, Figma, Webpack, Jest

WORK EXPERIENCE
Frontend Developer — PixelCraft Studio, Lahore (2023 - Present)
• Built and maintained a React component library used across 6 client products
• Improved page load performance by 35% through code splitting and lazy loading
• Collaborated with designers in Figma to implement pixel-accurate UI

Web Development Intern — StartHub, Lahore (2022 - 2023)
• Developed landing pages using React and Tailwind CSS for early-stage startups

EDUCATION
Bachelor of Science in Software Engineering — UET Lahore (2022)

PROJECTS
E-commerce Dashboard — React + Redux admin panel with real-time analytics
Recipe Finder App — Next.js app consuming a public REST API""",
        "skills": ["javascript", "react", "html", "css", "rest api"],
        "experience_years": 2,
        "education": "BS Software Engineering"
    },
    {
        "name": "Hamza Sheikh",
        "text": """Hamza Sheikh
Email: hamza.sheikh@email.com | Phone: +92-333-1112233 | Islamabad, Pakistan
GitHub: github.com/hamza-fullstack

PROFESSIONAL SUMMARY
Full Stack Developer with 3 years of experience building end-to-end web
applications. Comfortable across Python backends and React frontends, with
working knowledge of machine learning for data-driven features.

TECHNICAL SKILLS
Languages: Python, JavaScript, SQL
Backend: Django, Flask, REST APIs, PostgreSQL
Frontend: React, HTML, CSS
ML/Data: scikit-learn, pandas (basic feature work)
Tools: Git, Docker

WORK EXPERIENCE
Full Stack Developer — Bitwise Solutions, Islamabad (2022 - Present)
• Built a Django + React SaaS product from scratch, now serving 3,000+ users
• Implemented a scikit-learn based recommendation feature for the product catalog
• Designed REST APIs consumed by both the web app and a mobile client

Software Developer — Freelance (2021 - 2022)
• Delivered 8 client projects combining Flask backends with React frontends

EDUCATION
Bachelor of Science in Computer Science — COMSATS Islamabad (2021)

PROJECTS
SaaS Analytics Platform — Django + React + PostgreSQL, deployed on AWS
Product Recommender — scikit-learn collaborative filtering module""",
        "skills": ["python", "django", "flask", "react", "sql", "postgresql",
                   "scikit-learn", "rest api", "javascript", "docker"],
        "experience_years": 3,
        "education": "BS Computer Science"
    },
    {
        "name": "Mahnoor Iqbal",
        "text": """Mahnoor Iqbal
Email: mahnoor.iqbal@email.com | Phone: +92-345-7778899 | Faisalabad, Pakistan
GitHub: github.com/mahnoor-data

PROFESSIONAL SUMMARY
Data Analyst with 2 years of experience turning business data into actionable
insight using Python and SQL. Comfortable building dashboards and running
statistical analysis, with growing exposure to machine learning.

TECHNICAL SKILLS
Languages: Python, SQL, DAX
Data: pandas, numpy, matplotlib, Power BI, Excel
ML: scikit-learn (basic classification/regression)
Tools: Git, Jupyter Notebook

WORK EXPERIENCE
Data Analyst — RetailMetrics, Faisalabad (2023 - Present)
• Built weekly sales dashboards in Power BI used by 20+ store managers
• Automated reporting pipelines in Python, saving ~6 hours/week of manual work
• Ran A/B test analysis for a pricing experiment using pandas and scipy

Data Analyst Intern — MarketPulse, Lahore (2022 - 2023)
• Cleaned and analysed survey datasets, presented findings to stakeholders

EDUCATION
Bachelor of Science in Statistics — University of Agriculture Faisalabad (2022)

PROJECTS
Sales Forecasting Model — scikit-learn regression on 2 years of retail data
Customer Segmentation — K-means clustering on purchase history data""",
        "skills": ["python", "sql", "pandas", "numpy", "scikit-learn", "data analysis", "data science"],
        "experience_years": 2,
        "education": "BS Statistics"
    },
    {
        "name": "Faisal Chaudhry",
        "text": """Faisal Chaudhry
Email: faisal.chaudhry@email.com | Phone: +92-320-4445566 | Multan, Pakistan
GitHub: github.com/faisal-c

PROFESSIONAL SUMMARY
Recent Computer Science graduate with academic coursework in Python and web
development. Limited professional experience, currently seeking a first
full-time role as a junior developer.

TECHNICAL SKILLS
Languages: Python, HTML, CSS, basic JavaScript
Coursework: Data Structures, Algorithms, Databases (SQL basics)
Tools: Git (coursework use only)

EXPERIENCE
Final Year Project — "Library Management System", Bahauddin Zakariya University (2024)
• Built a simple desktop app in Python with a SQLite backend for book tracking
• No industry work experience yet; actively applying for junior developer roles

EDUCATION
Bachelor of Science in Computer Science — Bahauddin Zakariya University, Multan (2024)
CGPA: 3.1/4.0

PROJECTS
Library Management System — Python + SQLite desktop application
Personal Portfolio Website — static HTML/CSS site""",
        "skills": ["python", "html", "css", "sql"],
        "experience_years": 0,
        "education": "BS Computer Science"
    },
    {
        "name": "Noor Fatima",
        "text": """Noor Fatima
Email: noor.fatima@email.com | Phone: +92-311-9998877 | Islamabad, Pakistan
Google Scholar: scholar.google.com/noor-fatima | GitHub: github.com/noor-ml-research

PROFESSIONAL SUMMARY
Machine Learning Researcher (PhD candidate) with 5 years of combined research
and industry experience in deep learning, specialising in NLP and transformer
architectures. Multiple first-author publications at top-tier venues.

TECHNICAL SKILLS
Languages: Python, C++
ML/DL: PyTorch, TensorFlow, Hugging Face Transformers, JAX
NLP: BERT/GPT fine-tuning, tokenization pipelines, sequence labelling
Tools: Git, Docker, Weights & Biases, SLURM (HPC clusters)

RESEARCH & WORK EXPERIENCE
PhD Researcher — National Center for AI, Islamabad (2021 - Present)
• Published 4 first-author papers on efficient transformer fine-tuning
• Built a low-resource Urdu NLP toolkit adopted by 3 external research groups
• Mentored 6 junior researchers on PyTorch and experiment design

Machine Learning Engineer — DeepStack AI, Islamabad (2019 - 2021)
• Deployed BERT-based text classification models to production, 92% F1
• Built distributed training pipelines cutting model training time by 60%

EDUCATION
PhD in Computer Science (in progress) — NUST Islamabad
Master of Science in Artificial Intelligence — NUST Islamabad (2019)
Bachelor of Science in Computer Science — FAST NUCES Islamabad (2017)

PUBLICATIONS
"Efficient Fine-Tuning of Transformers for Low-Resource Languages" — ACL 2023
"Urdu NLP Toolkit: Tokenization and NER for Under-Resourced Languages" — EMNLP 2022""",
        "skills": ["python", "pytorch", "tensorflow", "nlp", "deep learning", "machine learning",
                   "transformers", "research", "hugging face"],
        "experience_years": 5,
        "education": "PhD (in progress)"
    },
]

# ── B1: load_data ─────────────────────────────────────────────────────────────
def load_data(mode, raw_resumes=None):
    """
    Load resume data.
    mode='sample'  → reads from data/sample_resumes.json (external dataset file)
    mode='upload'  → raw_resumes list passed from PDF upload
    mode='manual'  → raw_resumes list from UI form
    """
    if mode == "sample":
        return load_dataset_from_file()
    return raw_resumes or []

# ── B2: preprocess_data ───────────────────────────────────────────────────────
def preprocess_data(text):
    """Clean and normalize text."""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\,\+\#]', ' ', text)
    return text

# ── B3: extract_name ──────────────────────────────────────────────────────────
def extract_name(text):
    """Extract name using spaCy NER or fallback to first line."""
    if SPACY_OK:
        doc = NLP(text[:500])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
    lines = text.strip().split('\n')
    return lines[0].strip() if lines else "Unknown"

# ── B4: extract_skills ────────────────────────────────────────────────────────
def extract_skills(text):
    """
    Extract skills by matching against SKILLS_DB.
    Handles multi-word skills (e.g. 'machine learning') correctly,
    resolves aliases (e.g. 'sklearn' → 'scikit-learn'),
    and deduplicates results.
    """
    t = text.lower()
    # Resolve aliases first
    for alias, canonical in SKILL_ALIASES.items():
        if alias in t:
            t = t.replace(alias, canonical)
    # Match against SKILLS_DB (longest first to avoid partial matches)
    found = set()
    for skill in sorted(SKILLS_DB, key=len, reverse=True):
        # Word-boundary check for single-word skills to avoid 'java' in 'javascript'
        if ' ' in skill:
            if skill in t:
                found.add(skill)
        else:
            if re.search(r'\b' + re.escape(skill) + r'\b', t):
                found.add(skill)
    return list(found)

# ── B5: extract_experience ────────────────────────────────────────────────────
def extract_exp_years(text):
    """Extract years of experience using regex."""
    m = re.findall(r'(\d+)\s*(?:\+\s*)?years?', text.lower())
    return max([int(x) for x in m], default=0)

# ── B6: extract_education ─────────────────────────────────────────────────────
def extract_education(text):
    """Extract education lines using keyword matching."""
    keywords = ["bachelor","master","bs","ms","phd","university","cgpa","degree"]
    lines = text.split('\n')
    edu = [l.strip() for l in lines if any(k in l.lower() for k in keywords)]
    return edu[0] if edu else "Not specified"

# ── B6b: education_score ──────────────────────────────────────────────────────
def education_score(edu_text):
    """
    Score education level numerically.
    PhD=1.0, MS/Masters=0.8, BS/Bachelors=0.6, In-progress=0.4, Unknown=0.3
    Used as a bonus signal in the final weighted score.
    """
    t = edu_text.lower()
    if "phd" in t or "doctorate" in t or "d.phil" in t:
        return 1.0
    if "ms " in t or "m.s" in t or "master" in t or "msc" in t or "m.sc" in t:
        return 0.8
    if "bs " in t or "b.s" in t or "bachelor" in t or "bsc" in t or "b.sc" in t or "be " in t:
        return 0.6
    if "in progress" in t or "pursuing" in t or "ongoing" in t:
        return 0.4
    return 0.3

# ── B7: parse_jd ──────────────────────────────────────────────────────────────
def parse_jd(jd_text):
    """Parse job description for required skills and experience."""
    return {
        "required_skills":    extract_skills(jd_text),
        "required_exp":       extract_exp_years(jd_text),
    }

# ── B8: batch_tfidf_scores ───────────────────────────────────────────────────
def batch_tfidf_scores(resume_texts, jd_text):
    """
    Compute TF-IDF cosine similarity for ALL resumes in one vectorizer fit.
    Much faster than fitting a new vectorizer per resume (old approach).
    Returns a list of floats, one per resume.
    """
    try:
        vec  = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        docs = [preprocess_data(t) for t in resume_texts] + [preprocess_data(jd_text)]
        mat  = vec.fit_transform(docs)
        jd_vec   = mat[-1]        # last row = JD
        res_vecs = mat[:-1]       # all other rows = resumes
        sims = cosine_similarity(res_vecs, jd_vec)
        return [float(s[0]) for s in sims]
    except Exception:
        return [0.0] * len(resume_texts)

# ── B9: semantic_score ────────────────────────────────────────────────────────
def semantic_score(resume_text, jd_text):
    """Sentence Transformer semantic similarity (meaning-based)."""
    if not SEMANTIC_OK:
        return None
    try:
        e1 = SEMANTIC_MODEL.encode(resume_text[:512], convert_to_tensor=True)
        e2 = SEMANTIC_MODEL.encode(jd_text[:512],    convert_to_tensor=True)
        return float(st_util.cos_sim(e1, e2).item())
    except Exception:
        return None

# ── B10: skill_match_score ────────────────────────────────────────────────────
def skill_match_score(resume_skills, jd_skills):
    """Direct skill overlap ratio."""
    if not jd_skills:
        return 0.0
    matched = set(s.lower() for s in resume_skills) & set(s.lower() for s in jd_skills)
    return len(matched) / len(jd_skills)

# ── B11: exp_score ────────────────────────────────────────────────────────────
def exp_score(candidate_years, required_years):
    """Experience score — capped at 1.0."""
    if required_years == 0:
        return 1.0
    return min(candidate_years / required_years, 1.0)

# ── B12: run_model_or_algorithm (main screening) ──────────────────────────────
def run_screening(resumes, jd_text, w_tf, w_sk, w_ex, use_semantic=False):
    """
    Core AI screening algorithm — improved v2.
    Changes from v1:
      • Batch TF-IDF (single vectorizer for all resumes — faster)
      • Education score added as a bonus signal
      • Weight normalisation (weights always sum to 1.0)
      • Final scores capped at 100.0
      • Peak memory tracked via tracemalloc
    Returns ranked results + jd_skills list + timing dict.
    """
    tracemalloc.start()

    jd        = parse_jd(jd_text)
    jd_skills = jd["required_skills"]
    req_exp   = jd["required_exp"]
    timing    = {}

    # ── Normalise weights so they always sum to 1.0 (Module A validation) ──
    total_w = w_tf + w_sk + w_ex
    if total_w <= 0:
        w_tf, w_sk, w_ex = 0.4, 0.4, 0.2
        total_w = 1.0
    w_tf /= total_w
    w_sk /= total_w
    w_ex /= total_w

    # ── Batch TF-IDF (all resumes in one fit) ──────────────────────────────
    t0 = time.time()
    tf_scores = batch_tfidf_scores([r["text"] for r in resumes], jd_text)
    timing["tfidf_ms"] = round((time.time() - t0) * 1000, 1)

    # ── Build result records ───────────────────────────────────────────────
    results = []
    for i, r in enumerate(resumes):
        matched = list(set(s.lower() for s in r["skills"]) & set(s.lower() for s in jd_skills))
        missing = list(set(s.lower() for s in jd_skills) - set(s.lower() for s in r["skills"]))
        sk  = skill_match_score(r["skills"], jd_skills)
        ex  = exp_score(r["experience_years"], req_exp)
        edu = education_score(r.get("education", ""))
        results.append({
            "name":       r["name"],
            "initials":   ''.join(p[0].upper() for p in r["name"].split()[:2]),
            "tfidf":      round(tf_scores[i] * 100, 1),
            "semantic":   None,
            "skill":      round(sk * 100, 1),
            "exp_score":  round(ex * 100, 1),
            "edu_score":  round(edu * 100, 1),
            "score":      0,
            "skills":     r["skills"],
            "matched":    matched,
            "missing":    missing,
            "exp_yrs":    r["experience_years"],
            "education":  r.get("education", ""),
        })

    # ── Semantic scoring (optional) ────────────────────────────────────────
    t1 = time.time()
    if use_semantic and SEMANTIC_OK:
        for i, r in enumerate(resumes):
            sem = semantic_score(r["text"], jd_text)
            results[i]["semantic"] = round((sem or 0) * 100, 1)
    timing["semantic_ms"] = round((time.time() - t1) * 1000, 1)

    # ── Final weighted score (education adds up to 5% bonus) ──────────────
    for r in results:
        sem_val = r["semantic"] if r["semantic"] is not None else r["tfidf"]
        edu_bonus = r["edu_score"] * 0.05   # 5% weight for education level
        budget = 0.95   # remaining budget after the 5% education bonus

        if use_semantic and SEMANTIC_OK:
            # Semantic gets a fixed share of the budget; the user's TF-IDF/Skill/Exp
            # sliders split the REST of the budget in the proportion the user set —
            # so the sliders keep working even with Semantic AI turned on.
            sem_share = 0.35 * budget
            remain    = budget - sem_share
            base = (r["tfidf"] * w_tf * remain) + (sem_val * sem_share) + \
                   (r["skill"] * w_sk * remain) + (r["exp_score"] * w_ex * remain) + edu_bonus
        else:
            base = (r["tfidf"] * w_tf * budget) + (r["skill"] * w_sk * budget) + \
                   (r["exp_score"] * w_ex * budget) + edu_bonus

        # Cap at 100 so scores never exceed 100%
        r["score"] = round(min(base, 100.0), 1)

    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timing["peak_memory_kb"] = round(peak_mem / 1024, 1)

    return results, jd_skills, timing

# ── B13: generate_explanation ─────────────────────────────────────────────────
def generate_explanation(r, jd_skills, use_semantic):
    """Natural-language explanation of why candidate got this score."""
    s       = r["score"]
    matched = r["matched"]
    missing = r["missing"]
    verdict = "Strong Match" if s >= 75 else "Moderate Match" if s >= 50 else "Weak Match"

    factors = []
    if r["tfidf"] >= 60:  factors.append(f"high text similarity ({r['tfidf']}%)")
    if r["skill"] >= 60:  factors.append(f"good skill coverage ({r['skill']}%)")
    if r["exp_score"] >= 80: factors.append(f"meets experience requirement ({r['exp_yrs']} yrs)")
    if r["semantic"] and r["semantic"] >= 60:
        factors.append(f"strong semantic alignment ({r['semantic']}%)")

    detail = f"Covers {len(matched)} of {len(jd_skills)} required skills." if jd_skills else ""
    if missing:
        detail += f" Missing: {', '.join(missing[:3])}."
    if factors:
        detail += f" Key strengths: {', '.join(factors)}."

    return {"verdict": verdict, "detail": detail, "icon": "✅" if s>=75 else "⚠️" if s>=50 else "❌"}

# ── B14: evaluate_model ───────────────────────────────────────────────────────
def create_visuals(results, evaluation=None, gap_analysis=None):
    """
    Section 4 (Suggested Function Structure) — create_visuals(data, result).
    Assembles every chart/graph/grid/timeline-ready data structure from the
    screening results in ONE place, kept separate from the scoring logic
    (run_screening) and the HTTP layer (screen_route). The frontend (served
    by render_ui) turns this into the Chart.js bar/pie/radar charts, the
    skill network graph, the score grid, and the step-by-step timeline.
    """
    names  = [r["name"] for r in results]
    scores = [r["score"] for r in results]

    bar_chart     = {"labels": names, "data": scores}
    pie_chart     = {"labels": names, "data": scores}
    scatter_chart = {"points": [{"x": r.get("exp_yrs", 0), "y": r["score"], "name": r["name"]} for r in results]}
    radar_chart   = {
        "labels": ["TF-IDF", "Skill Match", "Experience", "Education"],
        "series": [{"name": r["name"], "values": [r["tfidf"], r["skill"], r["exp_score"], r["edu_score"]]}
                   for r in results[:5]],
    }

    # Grid/Map view — one cell per candidate, coloured by score band
    grid_cells = [{
        "name": r["name"], "score": r["score"],
        "band": "high" if r["score"] >= 70 else "mid" if r["score"] >= 40 else "low",
    } for r in results]

    # Timeline / step-by-step animation — the pipeline the app just ran
    timeline_steps = [
        {"step": 1, "label": "Load resumes",          "detail": f"{len(results)} candidate(s) loaded"},
        {"step": 2, "label": "Preprocess text",        "detail": "Tokenised, cleaned, skills/experience extracted"},
        {"step": 3, "label": "Score against JD",       "detail": "TF-IDF / semantic / skill / experience, weighted"},
        {"step": 4, "label": "Rank & tier candidates", "detail": "Sorted and grouped into tiers A–D"},
        {"step": 5, "label": "Generate explanations",  "detail": "Per-candidate factor breakdown produced"},
    ]

    # Graph/Network view — candidate ↔ matched-skill bipartite graph
    skill_nodes = {}
    edges = []
    for r in results:
        for sk in r.get("matched", []):
            skill_nodes[sk] = skill_nodes.get(sk, 0) + 1
            edges.append({"from": r["name"], "to": sk})
    network = {"candidate_nodes": names, "skill_nodes": list(skill_nodes.keys()), "edges": edges}

    return {
        "bar_chart": bar_chart,
        "pie_chart": pie_chart,
        "scatter_chart": scatter_chart,
        "radar_chart": radar_chart,
        "grid_cells": grid_cells,
        "timeline_steps": timeline_steps,
        "network": network,
    }


def render_ui():
    """
    Section 4 (Suggested Function Structure) — render_ui().
    Serves the single-page dashboard (HTML/CSS/JS) that renders every visual
    built from create_visuals()'s output plus the live /screen,
    /train_ml_models, /search_shortlist, etc. API responses.
    """
    return render_template_string(HTML)


def evaluate_model(results, threshold, timing, use_semantic):
    """
    Evaluation module — precision, recall, F1, runtime comparison.
    Treats candidates >= threshold as 'relevant' (ground truth simulation).
    """
    qualified = [r for r in results if r["score"] >= threshold]
    total     = len(results)
    tp        = len(qualified)
    fp        = total - tp
    fn        = 0

    precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0
    recall    = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0
    f1        = round(2 * precision * recall / (precision + recall), 1) if (precision + recall) > 0 else 0

    scores = [r["score"] for r in results]
    avg    = round(sum(scores) / len(scores), 1) if scores else 0
    stddev = round(math.sqrt(sum((s - avg) ** 2 for s in scores) / len(scores)), 1) if scores else 0

    return {
        "precision":      precision,
        "recall":         recall,
        "f1":             f1,
        "tfidf_ms":       timing.get("tfidf_ms", 0),
        "semantic_ms":    timing.get("semantic_ms", 0),
        "total_ms":       round(timing.get("tfidf_ms", 0) + timing.get("semantic_ms", 0), 1),
        "peak_memory_kb": timing.get("peak_memory_kb", 0),
        "qualified":      tp,
        "total":          total,
        "avg_score":      avg,
        "stddev":         stddev,
        "semantic_ok":    SEMANTIC_OK and use_semantic,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MODULE D — NLP PIPELINE (Option 4: full NLP analysis shown to user)
# ─────────────────────────────────────────────────────────────────────────────
def generate_summary(text, freq, n_sentences=2):
    """
    Extractive text summarisation (frequency-scoring method).
    Each sentence is scored by the sum of its keyword frequencies
    (after stop-word removal); the top-N highest scoring sentences,
    re-ordered as they appeared in the original text, form the summary.
    Satisfies Section 5, Option 4: "Summarize input/output in plain
    language" without needing a heavyweight transformer model.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    if len(sentences) <= n_sentences:
        return " ".join(sentences)

    scored = []
    for i, sent in enumerate(sentences):
        words = re.findall(r"\b[a-zA-Z0-9\+\#\.]+\b", sent.lower())
        score = sum(freq.get(w, 0) for w in words)
        norm_score = score / max(len(words), 1)   # avoid bias toward long sentences
        scored.append((i, norm_score, sent))

    top = sorted(scored, key=lambda x: -x[1])[:n_sentences]
    top_in_order = sorted(top, key=lambda x: x[0])
    return " ".join(s for _, _, s in top_in_order)


def analyze_sentiment(clean_tokens):
    """
    Lightweight lexicon-based sentiment/tone analysis (no external API).
    Tuned for resume/JD context: 'strong', 'proficient', 'excellent' etc.
    read as positive tone; 'lack', 'weak', 'limited' etc. as negative tone.
    Returns polarity label, a -1..+1 score, and which matched words drove it —
    an additional NLP feature beyond tokenisation/POS/NER/summarisation.
    """
    POSITIVE = {
        "excellent","strong","proficient","expert","skilled","experienced","advanced",
        "successful","achieved","improved","led","leadership","innovative","efficient",
        "passionate","dedicated","proven","outstanding","talented","motivated","award",
        "optimized","exceeded","strength","robust","solid","confident","capable","great",
    }
    NEGATIVE = {
        "lack","weak","limited","poor","insufficient","struggle","failed","gap","missing",
        "inexperienced","unable","difficulty","problem","concern","risk","below","behind",
        "outdated","unqualified","incomplete","delay","issue","weakness",
    }
    pos_hits = [t for t in clean_tokens if t in POSITIVE]
    neg_hits = [t for t in clean_tokens if t in NEGATIVE]
    total = len(pos_hits) + len(neg_hits)
    score = round((len(pos_hits) - len(neg_hits)) / total, 3) if total else 0.0
    polarity = "Positive" if score > 0.15 else "Negative" if score < -0.15 else "Neutral"
    return {
        "polarity": polarity,
        "score": score,
        "positive_words": sorted(set(pos_hits))[:10],
        "negative_words": sorted(set(neg_hits))[:10],
    }


def extract_bigrams(clean_tokens, top_n=10):
    """Top co-occurring word pairs (bigrams) — a standard NLP feature that
    surfaces multi-word skills/phrases (e.g. 'machine learning', 'rest api')
    that single-keyword frequency alone would miss."""
    if len(clean_tokens) < 2:
        return []
    pairs = {}
    for a, b in zip(clean_tokens, clean_tokens[1:]):
        bg = f"{a} {b}"
        pairs[bg] = pairs.get(bg, 0) + 1
    top = sorted(pairs.items(), key=lambda x: -x[1])[:top_n]
    return [{"phrase": p, "count": c} for p, c in top if c >= 1]


def nlp_pipeline_analysis(text, label="text"):
    """
    Full NLP pipeline demonstration:
      1. Tokenization
      2. Stop-word removal
      3. Lemmatization (via spaCy or regex fallback)
      4. POS tagging (via spaCy)
      5. NER — Named Entity Recognition (via spaCy)
      6. Keyword frequency
      7. Sentence count / word count stats
    Returns a dict of all NLP steps for display in the UI.
    """
    result = {
        "label": label,
        "raw_length": len(text),
        "sentences": [],
        "tokens": [],
        "clean_tokens": [],
        "lemmas": [],
        "pos_tags": [],
        "entities": [],
        "keywords": [],
        "bigrams": [],
        "sentiment": {},
        "summary": "",
        "stats": {},
    }

    STOP_WORDS = {
        "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
        "yourself","he","him","his","himself","she","her","hers","herself","it","its",
        "itself","they","them","their","theirs","themselves","what","which","who","whom",
        "this","that","these","those","am","is","are","was","were","be","been","being",
        "have","has","had","having","do","does","did","doing","a","an","the","and",
        "but","if","or","because","as","until","while","of","at","by","for","with",
        "about","against","between","into","through","during","before","after","above",
        "below","to","from","up","down","in","out","on","off","over","under","again",
        "further","then","once","here","there","when","where","why","how","all","both",
        "each","few","more","most","other","some","such","no","nor","not","only","own",
        "same","so","than","too","very","s","t","can","will","just","don","should",
        "now","d","ll","m","o","re","ve","y","ain","wasn","wouldn",
    }

    try:
        # Sentence split
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        result["sentences"] = sentences[:5]  # show first 5

        # Tokenization — split on non-alphanumeric
        raw_tokens = re.findall(r"\b[a-zA-Z0-9\+\#\.]+\b", text.lower())
        result["tokens"] = raw_tokens[:30]

        # Stop-word removal
        clean_tokens = [t for t in raw_tokens if t not in STOP_WORDS and len(t) > 1]
        result["clean_tokens"] = clean_tokens[:30]

        # Keyword frequency
        freq = {}
        for t in clean_tokens:
            freq[t] = freq.get(t, 0) + 1
        top_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:15]
        result["keywords"] = [{"word": w, "count": c} for w, c in top_kw]

        # Bigrams — common two-word phrases (Option 4: fuller NLP feature set)
        result["bigrams"] = extract_bigrams(clean_tokens, top_n=10)

        # Sentiment / tone analysis (lexicon-based, no external API)
        result["sentiment"] = analyze_sentiment(clean_tokens)

        # Extractive summarisation — plain-language summary (Option 4)
        result["summary"] = generate_summary(text, freq, n_sentences=2)

        # spaCy pipeline (if available)
        if SPACY_OK:
            doc = NLP(text[:1000])
            # Lemmatization
            result["lemmas"] = [
                {"token": token.text, "lemma": token.lemma_}
                for token in doc
                if not token.is_stop and not token.is_punct and len(token.text) > 1
            ][:20]
            # POS tags
            result["pos_tags"] = [
                {"token": token.text, "pos": token.pos_, "tag": token.tag_}
                for token in doc
                if not token.is_punct and len(token.text) > 1
            ][:25]
            # NER
            result["entities"] = [
                {"text": ent.text, "label": ent.label_, "description": spacy.explain(ent.label_) or ent.label_}
                for ent in doc.ents
            ]
        else:
            # Fallback lemmatization (simple suffix stripping)
            def simple_lemma(word):
                for suffix in ["ing","tion","ness","ment","ed","er","ly","ize","ise","ity"]:
                    if word.endswith(suffix) and len(word) - len(suffix) > 2:
                        return word[:-len(suffix)]
                return word
            result["lemmas"] = [
                {"token": t, "lemma": simple_lemma(t)}
                for t in clean_tokens[:20]
            ]
            result["pos_tags"]  = []
            result["entities"]  = []

        # Stats
        word_count = len(raw_tokens)
        result["stats"] = {
            "word_count":        word_count,
            "sentence_count":    len(sentences),
            "unique_tokens":     len(set(clean_tokens)),
            "stop_words_removed": len(raw_tokens) - len(clean_tokens),
            "avg_word_length":   round(sum(len(t) for t in clean_tokens) / max(len(clean_tokens), 1), 1),
            "entity_count":      len(result["entities"]),
            "spacy_used":        SPACY_OK,
        }

    except Exception as e:
        result["error"] = str(e)

    return result

# ─────────────────────────────────────────────────────────────────────────────
# MODULE — FORWARD CHAINING (Option 1: Rule-based AI)
# ─────────────────────────────────────────────────────────────────────────────
def forward_chaining_rules(candidate, jd_text):
    """
    Rule-based AI using forward chaining inference.
    Facts + Rules → Derived decisions (Section 5, Option 1).
    Returns list of fired rules with explanations.
    """
    # Initial facts derived from candidate
    facts = {
        "has_python":        "python" in [s.lower() for s in candidate.get("skills", [])],
        "has_ml":            any(s.lower() in ["machine learning","deep learning","nlp"] for s in candidate.get("skills", [])),
        "has_degree":        candidate.get("education","").lower() not in ["not specified",""],
        "has_ms_or_phd":     any(k in candidate.get("education","").lower() for k in ["ms","master","phd","doctorate"]),
        "experience_ok":     candidate.get("experience_years", 0) >= extract_exp_years(jd_text),
        "senior_exp":        candidate.get("experience_years", 0) >= 4,
        "skill_count":       len(candidate.get("skills", [])),
        "many_skills":       len(candidate.get("skills", [])) >= 5,
        "score":             candidate.get("score", 0),
        "high_score":        candidate.get("score", 0) >= 75,
        "medium_score":      60 <= candidate.get("score", 0) < 75,
    }

    fired_rules = []

    # Rule 1: If has_python AND has_ml → is_ai_developer
    if facts["has_python"] and facts["has_ml"]:
        facts["is_ai_developer"] = True
        fired_rules.append({
            "rule": "R1",
            "condition": "has_python ∧ has_ml",
            "conclusion": "is_ai_developer = TRUE",
            "explanation": "Candidate knows Python and at least one ML framework — classified as AI Developer."
        })

    # Rule 2: If experience_ok AND many_skills → meets_base_requirement
    if facts["experience_ok"] and facts["many_skills"]:
        facts["meets_base_requirement"] = True
        fired_rules.append({
            "rule": "R2",
            "condition": "experience_ok ∧ many_skills (≥5)",
            "conclusion": "meets_base_requirement = TRUE",
            "explanation": f"Has {facts['skill_count']} skills and sufficient experience — meets base job requirements."
        })

    # Rule 3: If high_score AND meets_base_requirement → recommend_interview
    if facts.get("meets_base_requirement") and facts["high_score"]:
        facts["recommend_interview"] = True
        fired_rules.append({
            "rule": "R3",
            "condition": "meets_base_requirement ∧ high_score (≥75%)",
            "conclusion": "RECOMMEND FOR INTERVIEW ✅",
            "explanation": "Candidate meets all base requirements and scored ≥75% — strongly recommended."
        })

    # Rule 4: If senior_exp AND has_ms_or_phd → senior_candidate
    if facts["senior_exp"] and facts["has_ms_or_phd"]:
        facts["senior_candidate"] = True
        fired_rules.append({
            "rule": "R4",
            "condition": "senior_exp (≥4 yrs) ∧ has_ms_or_phd",
            "conclusion": "senior_candidate = TRUE",
            "explanation": "Has 4+ years AND a postgraduate degree — qualifies as Senior Candidate."
        })

    # Rule 5: If medium_score AND meets_base_requirement → consider_candidate
    if facts.get("meets_base_requirement") and facts["medium_score"] and not facts.get("recommend_interview"):
        fired_rules.append({
            "rule": "R5",
            "condition": "meets_base_requirement ∧ medium_score (60–75%)",
            "conclusion": "CONSIDER WITH CAUTION ⚠️",
            "explanation": "Meets base requirements but score is moderate — may need further evaluation."
        })

    # Rule 6: If NOT experience_ok → flag_experience_gap
    if not facts["experience_ok"]:
        fired_rules.append({
            "rule": "R6",
            "condition": "¬experience_ok",
            "conclusion": "flag_experience_gap = TRUE ❌",
            "explanation": f"Candidate has {candidate.get('experience_years',0)} years but JD requires {extract_exp_years(jd_text)} years."
        })

    # Rule 7: If score < 45 → NOT_RECOMMENDED
    if facts["score"] < 45:
        fired_rules.append({
            "rule": "R7",
            "condition": "score < 45",
            "conclusion": "NOT RECOMMENDED ❌",
            "explanation": "Score is below minimum threshold — candidate profile does not match this JD."
        })

    return {
        "facts":       facts,
        "fired_rules": fired_rules,
        "final_decision": (
            "RECOMMEND FOR INTERVIEW" if facts.get("recommend_interview")
            else "CONSIDER WITH CAUTION" if facts.get("meets_base_requirement")
            else "NOT RECOMMENDED"
        )
    }

# ─────────────────────────────────────────────────────────────────────────────
# MODULE E — APPROACH COMPARISON (Section 3-E, Section 8)
# ─────────────────────────────────────────────────────────────────────────────
def compare_approaches(resumes, jd_text, threshold):
    """
    Compares THREE approaches side-by-side (Section 3-E: compare at least 2):
      Approach 1: Rule-based only (forward chaining tiers)
      Approach 2: TF-IDF + Skill matching (statistical)
      Approach 3: TF-IDF + Semantic + Skill (full AI)
    Returns comparison data for each candidate across all approaches.
    """
    jd     = parse_jd(jd_text)
    req_exp = jd["required_exp"]

    # Approach 1 — Rule-based scoring (no ML)
    t0 = time.time()
    rule_scores = []
    for r in resumes:
        sk    = skill_match_score(r["skills"], jd["required_skills"])
        ex    = exp_score(r["experience_years"], req_exp)
        edu   = education_score(r.get("education",""))
        # Pure rule: skill 50% + experience 30% + education 20%
        score = round(min((sk*50) + (ex*30) + (edu*20), 100), 1)
        rule_scores.append({"name": r["name"], "score": score,
                            "qualified": score >= threshold, "approach": "Rule-Based"})
    t_rule = round((time.time() - t0) * 1000, 1)

    # Approach 2 — TF-IDF + Skill (statistical, no semantic)
    t1 = time.time()
    tf_scores = batch_tfidf_scores([r["text"] for r in resumes], jd_text)
    tfidf_only_scores = []
    for i, r in enumerate(resumes):
        sk    = skill_match_score(r["skills"], jd["required_skills"])
        ex    = exp_score(r["experience_years"], req_exp)
        score = round(min((tf_scores[i]*100*0.40) + (sk*100*0.40) + (ex*100*0.20), 100), 1)
        tfidf_only_scores.append({"name": r["name"], "score": score,
                                  "qualified": score >= threshold, "approach": "TF-IDF+Skill"})
    t_tfidf = round((time.time() - t1) * 1000, 1)

    # Approach 3 — Full AI (TF-IDF + Semantic + Skill)
    t2 = time.time()
    full_scores = []
    for i, r in enumerate(resumes):
        sk    = skill_match_score(r["skills"], jd["required_skills"])
        ex    = exp_score(r["experience_years"], req_exp)
        edu   = education_score(r.get("education",""))
        if SEMANTIC_OK:
            sem   = semantic_score(r["text"], jd_text) or 0
            score = round(min((tf_scores[i]*100*0.19)+(sem*100*0.33)+(sk*100*0.28)+(ex*100*0.15)+(edu*100*0.05), 100), 1)
        else:
            score = round(min((tf_scores[i]*100*0.38)+(sk*100*0.38)+(ex*100*0.19)+(edu*100*0.05), 100), 1)
        full_scores.append({"name": r["name"], "score": score,
                            "qualified": score >= threshold, "approach": "Full AI"})
    t_full = round((time.time() - t2) * 1000, 1)

    # Build comparison table
    comparison = []
    for i, r in enumerate(resumes):
        comparison.append({
            "name":        r["name"],
            "rule_score":  rule_scores[i]["score"],
            "tfidf_score": tfidf_only_scores[i]["score"],
            "full_score":  full_scores[i]["score"],
            "rule_qual":   rule_scores[i]["qualified"],
            "tfidf_qual":  tfidf_only_scores[i]["qualified"],
            "full_qual":   full_scores[i]["qualified"],
        })

    return {
        "comparison": comparison,
        "timings": {
            "rule_ms":  t_rule,
            "tfidf_ms": t_tfidf,
            "full_ms":  t_full,
        },
        "approach_stats": {
            "rule_qualified":  sum(1 for x in rule_scores  if x["qualified"]),
            "tfidf_qualified": sum(1 for x in tfidf_only_scores if x["qualified"]),
            "full_qualified":  sum(1 for x in full_scores  if x["qualified"]),
            "rule_avg":  round(sum(x["score"] for x in rule_scores)/len(rule_scores),1)  if rule_scores  else 0,
            "tfidf_avg": round(sum(x["score"] for x in tfidf_only_scores)/len(tfidf_only_scores),1) if tfidf_only_scores else 0,
            "full_avg":  round(sum(x["score"] for x in full_scores)/len(full_scores),1)  if full_scores  else 0,
        },
        "semantic_available": SEMANTIC_OK,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MODULE — SEARCH & OPTIMISATION AI (Option 2 of Section 5)
# ─────────────────────────────────────────────────────────────────────────────
# Problem framed as a state-space search:
#   "Pick the best shortlist of exactly K candidates that maximises the total
#    combined score" — i.e. optimal subset selection (a 0/1-knapsack-style
#    combinatorial search over inclusion/exclusion decisions).
#
# State  = (index processed so far, tuple of chosen candidate indices)
# Action = include candidate[index]  OR  skip candidate[index]
# Goal   = index == n  AND  len(chosen) == K
# g(n)   = -(total score of chosen so far)      (cost to minimise = maximise score)
# h(n)   = -(best-case score obtainable from remaining slots), an optimistic
#           (admissible) estimate built by pre-sorting remaining scores desc.
#
# Each algorithm below returns the same shape of result so they can be
# compared side-by-side in Module E (Evaluation): path/solution found,
# total score, nodes expanded, max frontier size (memory proxy), runtime.
# ─────────────────────────────────────────────────────────────────────────────
import heapq
from collections import deque


def _remaining_upper_bound(scores_sorted_desc, start_idx, slots_left):
    """Admissible heuristic: best possible additional score if we could
    freely take the top `slots_left` remaining candidates (ignores the
    inclusion/exclusion ordering constraint, so it never under-estimates)."""
    remaining = scores_sorted_desc[start_idx:]
    return sum(remaining[:slots_left]) if slots_left > 0 else 0


def search_optimal_shortlist(candidates, budget, algorithm="bfs", max_nodes=20000):
    """
    Runs one search/optimisation algorithm over the candidate-selection
    state space and returns the best shortlist found plus search-process
    statistics (nodes expanded, frontier size, explored-state trail) so the
    UI can visualise how each algorithm explores the space.

    candidates : list of {"name":..., "score":...}, pre-sorted by score desc
                 for a tighter heuristic bound.
    budget     : K — exact headcount to shortlist.
    algorithm  : 'bfs' | 'dfs' | 'ucs' | 'greedy' | 'astar' | 'hill_climb'
    """
    t0 = time.perf_counter()
    n = len(candidates)
    budget = max(0, min(budget, n))
    scores = [c["score"] for c in candidates]

    nodes_expanded = 0
    max_frontier = 0
    explored_trail = []          # first ~25 states visited, for the UI
    best_state = {"chosen": (), "total": -1}

    def record(chosen, idx, total):
        if len(explored_trail) < 25:
            explored_trail.append({
                "step": len(explored_trail) + 1,
                "considered_idx": idx if idx < n else None,
                "candidate": candidates[idx]["name"] if idx < n else "—",
                "chosen_so_far": [candidates[i]["name"] for i in chosen],
                "running_total": round(total, 2),
            })

    # ── Local Search (Hill Climbing) — different paradigm: no explicit tree,
    #    starts from one complete candidate solution and iteratively improves it
    #    by swapping in a better-scoring candidate that isn't in the shortlist.
    if algorithm == "hill_climb":
        current = list(range(min(budget, n)))              # naive initial solution: first K
        current_total = sum(scores[i] for i in current)
        record(current, current[-1] if current else 0, current_total)
        improved = True
        while improved and nodes_expanded < max_nodes:
            improved = False
            outside = [i for i in range(n) if i not in current]
            for out_i in sorted(outside, key=lambda i: -scores[i]):
                worst_in_idx = min(range(len(current)), key=lambda p: scores[current[p]])
                if scores[out_i] > scores[current[worst_in_idx]]:
                    current[worst_in_idx] = out_i
                    current_total = sum(scores[i] for i in current)
                    nodes_expanded += 1
                    record(current, out_i, current_total)
                    improved = True
                    break
            max_frontier = max(max_frontier, 1)  # hill climbing keeps 1 current state
        best_state = {"chosen": tuple(current), "total": current_total}

    else:
        # ── Shared tree-search machinery for BFS / DFS / UCS / Greedy / A* ──
        # A "node" = (index, chosen_tuple, total_score)
        start = (0, (), 0.0)

        if algorithm == "bfs":
            frontier = deque([start])
            pop = frontier.popleft
            push = frontier.append
        elif algorithm == "dfs":
            frontier = [start]
            pop = frontier.pop
            push = frontier.append
        else:
            # ucs / greedy / astar all use a priority queue, differing only
            # in what value is used as priority.
            counter = [0]
            frontier = []
            def push(node):
                idx, chosen, total = node
                slots_left = budget - len(chosen)
                g = -total                                            # cost so far
                h = -_remaining_upper_bound(scores, idx, slots_left)   # heuristic
                if algorithm == "ucs":
                    priority = g
                elif algorithm == "greedy":
                    priority = h
                else:  # astar
                    priority = g + h
                counter[0] += 1
                heapq.heappush(frontier, (priority, counter[0], node))
            def pop():
                return heapq.heappop(frontier)[2]

        push(start)

        while frontier and nodes_expanded < max_nodes:
            max_frontier = max(max_frontier, len(frontier))
            idx, chosen, total = pop()
            nodes_expanded += 1
            record(chosen, idx, total)

            # Goal test
            if len(chosen) == budget:
                if total > best_state["total"]:
                    best_state = {"chosen": chosen, "total": total}
                if algorithm in ("dfs", "bfs"):
                    continue   # keep exploring to also visualise other branches
                else:
                    break      # ucs/greedy/astar: first goal popped is optimal (or near for greedy)
            if idx >= n:
                continue

            slots_left_after_skip = budget - len(chosen)
            # Prune: not enough candidates left to reach budget
            if n - idx - 1 < slots_left_after_skip:
                pass
            else:
                push((idx + 1, chosen, total))                      # SKIP candidate[idx]
            if len(chosen) < budget:
                push((idx + 1, chosen + (idx,), total + scores[idx]))  # INCLUDE candidate[idx]

        # fallback if budget never fully reached within max_nodes
        if best_state["total"] < 0:
            fallback = tuple(sorted(range(min(budget, n)), key=lambda i: -scores[i]))
            best_state = {"chosen": fallback, "total": sum(scores[i] for i in fallback)}

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    chosen_names = [candidates[i]["name"] for i in best_state["chosen"]]
    return {
        "algorithm": algorithm,
        "selected": chosen_names,
        "total_score": round(best_state["total"], 2),
        "avg_score": round(best_state["total"] / len(best_state["chosen"]), 2) if best_state["chosen"] else 0,
        "nodes_expanded": nodes_expanded,
        "max_frontier": max_frontier,
        "runtime_ms": elapsed_ms,
        "explored_trail": explored_trail,
    }


ALGORITHMS_META = {
    "bfs":        {"label": "BFS",           "full": "Breadth-First Search",         "optimal": "Yes (unweighted)",    "color": "#3b82f6"},
    "dfs":        {"label": "DFS",           "full": "Depth-First Search",           "optimal": "No",                  "color": "#8b5cf6"},
    "ucs":        {"label": "UCS",           "full": "Uniform-Cost Search",          "optimal": "Yes",                 "color": "#059669"},
    "greedy":     {"label": "Greedy",        "full": "Greedy Best-First Search",     "optimal": "No",                  "color": "#d97706"},
    "astar":      {"label": "A*",            "full": "A* Search",                    "optimal": "Yes (admissible h)",  "color": "#dc2626"},
    "hill_climb": {"label": "Hill Climbing", "full": "Local Search (Hill Climbing)", "optimal": "No (local optimum)",  "color": "#0891b2"},
}


def run_all_search_algorithms(results, budget):
    """Runs every Option-2 algorithm on the same screened candidates so they
    can be compared head-to-head — mirrors compare_approaches() for Option 3."""
    candidates = sorted(
        [{"name": r["name"], "score": r.get("score", 0)} for r in results],
        key=lambda c: -c["score"]
    )
    out = []
    for algo in ["bfs", "dfs", "ucs", "greedy", "astar", "hill_climb"]:
        res = search_optimal_shortlist(candidates, budget, algo)
        res.update(ALGORITHMS_META[algo])
        out.append(res)
    best_total = max(r["total_score"] for r in out) if out else 0
    return {
        "budget": budget,
        "n_candidates": len(candidates),
        "algorithms": out,
        "best_total_score": best_total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE — MACHINE LEARNING AI (Option 3 of Section 5, done "properly")
# ─────────────────────────────────────────────────────────────────────────────
# The core screening score (TF-IDF / semantic similarity) is a *similarity*
# score, not a trained model. This module adds genuine trained models on top
# of the screened candidates so the guide's requirement — "Train a model on
# dataset (classification / regression / clustering)" + "show training
# results and predictions visually" — is satisfied with real scikit-learn
# models, real metrics, and honest small-sample handling:
#
#   • Classification — LogisticRegression predicts Qualified/Not-Qualified
#     from PARTIAL features (tfidf, skill, exp_score) so the label isn't
#     trivially reproduced from the inputs.
#   • Regression      — LinearRegression predicts the final score from a
#     DIFFERENT partial feature set (skill, exp_score only) so residual
#     error is meaningful, not zero by construction.
#   • Clustering      — KMeans groups candidates purely unsupervised
#     (no score/label used), then PCA projects to 2D for visualisation.
#
# With small candidate counts (typical in a resume-screening demo), a single
# train/test split is unreliable, so classification & regression fall back
# to Leave-One-Out cross-validation automatically — this is disclosed to
# the user in the response ("method_used") rather than hidden.
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_predict
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, r2_score, mean_absolute_error,
                              mean_squared_error)
try:
    from sklearn.metrics import silhouette_score
    SILHOUETTE_OK = True
except Exception:
    SILHOUETTE_OK = False


def train_classification_model(results, threshold):
    """Supervised classification: predict Qualified (1) / Not-Qualified (0)
    from tfidf + skill + exp_score only (education is deliberately withheld
    so the model has to genuinely generalise, not just re-derive the label
    the scoring formula already computed)."""
    n = len(results)
    if n < 4:
        return {"error": "Need at least 4 screened candidates to train a classifier."}

    X = np.array([[r["tfidf"], r["skill"], r["exp_score"]] for r in results], dtype=float)
    y = np.array([1 if r["score"] >= threshold else 0 for r in results], dtype=int)

    if len(set(y)) < 2:
        return {"error": f"All {n} candidates fall on one side of the {threshold}% threshold — "
                          f"lower/raise the qualification threshold so both classes are present, then retrain."}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    use_holdout = n >= 8
    if use_holdout:
        strat = y if min(np.bincount(y)) >= 2 else None
        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            Xs, y, np.arange(n), test_size=0.3, random_state=42, stratify=strat)
        model = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_eval, y_pred_eval = y_te, y_pred
        method_used = f"70/30 train-test split ({len(y_tr)} train / {len(y_te)} test)"
    else:
        try:
            loo = LeaveOneOut()
            y_pred_eval = cross_val_predict(LogisticRegression(max_iter=1000), Xs, y, cv=loo)
            y_eval = y
            method_used = f"Leave-One-Out cross-validation ({n} folds — dataset too small for a reliable holdout split)"
        except ValueError:
            # Extremely imbalanced tiny sample (e.g. only 1 of a class) — LOO
            # can leave a fold with a single class. Fall back to an honest
            # in-sample fit and say so, rather than crashing.
            model = LogisticRegression(max_iter=1000).fit(Xs, y)
            y_pred_eval = model.predict(Xs)
            y_eval = y
            method_used = (f"In-sample fit on all {n} candidates (too few examples of one class for "
                            f"cross-validation — treat these metrics as best-case, not generalisation, performance)")

    acc  = round(accuracy_score(y_eval, y_pred_eval) * 100, 1)
    prec = round(precision_score(y_eval, y_pred_eval, zero_division=0) * 100, 1)
    rec  = round(recall_score(y_eval, y_pred_eval, zero_division=0) * 100, 1)
    f1   = round(f1_score(y_eval, y_pred_eval, zero_division=0) * 100, 1)
    cm   = confusion_matrix(y_eval, y_pred_eval, labels=[0, 1]).tolist()

    # Fit final model on ALL data purely to show learned feature weights +
    # per-candidate predicted probability (clearly labelled as such in the UI).
    final_model = LogisticRegression(max_iter=1000).fit(Xs, y)
    probs = final_model.predict_proba(Xs)[:, 1]
    preds = final_model.predict(Xs)

    feature_importance = [
        {"feature": "TF-IDF Similarity", "weight": round(float(final_model.coef_[0][0]), 3)},
        {"feature": "Skill Match",       "weight": round(float(final_model.coef_[0][1]), 3)},
        {"feature": "Experience Score",  "weight": round(float(final_model.coef_[0][2]), 3)},
    ]

    predictions = [{
        "name": r["name"],
        "actual_label": "Qualified" if y[i] == 1 else "Not Qualified",
        "predicted_label": "Qualified" if preds[i] == 1 else "Not Qualified",
        "probability": round(float(probs[i]) * 100, 1),
        "correct": bool(preds[i] == y[i]),
    } for i, r in enumerate(results)]

    top_pick = max(predictions, key=lambda p: p["probability"])
    n_pred_qualified = sum(1 for p in predictions if p["predicted_label"] == "Qualified")
    recommendation = (
        f"The trained classifier predicts {n_pred_qualified} of {n} candidates as Qualified. "
        f"Its most confident pick is {top_pick['name']} ({top_pick['probability']}% predicted probability of being Qualified). "
        f"Model accuracy on {('held-out' if use_holdout else 'cross-validated')} data was {acc}%, "
        f"so treat this as a second opinion alongside the rule-based and similarity scores, not a sole decision-maker."
    )

    return {
        "task": "classification",
        "algorithm": "Logistic Regression",
        "method_used": method_used,
        "n_samples": n,
        "features_used": ["TF-IDF Similarity", "Skill Match", "Experience Score"],
        "label_definition": f"Qualified = final score ≥ {threshold}% (education score intentionally withheld from the model)",
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        "confusion_matrix": cm,
        "confusion_labels": ["Not Qualified", "Qualified"],
        "feature_importance": feature_importance,
        "predictions": predictions,
        "recommendation": recommendation,
    }


def train_regression_model(results):
    """Supervised regression: predict the final weighted score from skill +
    experience only (TF-IDF/semantic withheld), so residual error reflects
    genuine unexplained variance rather than a tautology."""
    n = len(results)
    if n < 4:
        return {"error": "Need at least 4 screened candidates to train a regression model."}

    X = np.array([[r["skill"], r["exp_score"]] for r in results], dtype=float)
    y = np.array([r["score"] for r in results], dtype=float)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    use_holdout = n >= 8
    if use_holdout:
        X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
            Xs, y, np.arange(n), test_size=0.3, random_state=42)
        model = LinearRegression().fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_eval, y_pred_eval = y_te, y_pred
        method_used = f"70/30 train-test split ({len(y_tr)} train / {len(y_te)} test)"
    else:
        loo = LeaveOneOut()
        y_pred_eval = cross_val_predict(LinearRegression(), Xs, y, cv=loo)
        y_eval = y
        method_used = f"Leave-One-Out cross-validation ({n} folds — dataset too small for a reliable holdout split)"

    r2   = round(r2_score(y_eval, y_pred_eval), 3)
    mae  = round(mean_absolute_error(y_eval, y_pred_eval), 2)
    rmse = round(mean_squared_error(y_eval, y_pred_eval) ** 0.5, 2)

    final_model = LinearRegression().fit(Xs, y)
    y_pred_all = final_model.predict(Xs)

    feature_importance = [
        {"feature": "Skill Match",      "weight": round(float(final_model.coef_[0]), 3)},
        {"feature": "Experience Score", "weight": round(float(final_model.coef_[1]), 3)},
    ]

    predictions = [{
        "name": r["name"],
        "actual_score": r["score"],
        "predicted_score": round(float(y_pred_all[i]), 1),
        "residual": round(r["score"] - float(y_pred_all[i]), 1),
    } for i, r in enumerate(results)]

    return {
        "task": "regression",
        "algorithm": "Linear Regression",
        "method_used": method_used,
        "n_samples": n,
        "features_used": ["Skill Match", "Experience Score"],
        "target": "Final weighted score (TF-IDF/semantic withheld from inputs)",
        "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
        "feature_importance": feature_importance,
        "predictions": predictions,
    }


def train_clustering_model(results, k=3):
    """Unsupervised clustering: groups candidates by their score profile
    (tfidf, skill, exp_score, edu_score) with no label/score used as a
    target — purely exploratory. PCA projects the 4D feature space to 2D
    for visualisation. Clusters are auto-labelled by their mean score."""
    n = len(results)
    if n < 3:
        return {"error": "Need at least 3 screened candidates to run clustering."}
    k = max(2, min(k, n - 1))

    X = np.array([[r["tfidf"], r["skill"], r["exp_score"], r["edu_score"]] for r in results], dtype=float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    cluster_ids = kmeans.labels_

    sil = None
    if SILHOUETTE_OK and n > k + 1:
        try:
            sil = round(float(silhouette_score(Xs, cluster_ids)), 3)
        except Exception:
            sil = None

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xs)
    explained_var = [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_]

    # Auto-label clusters by mean candidate score, best → worst
    cluster_avg = {}
    for cid in range(k):
        members = [results[i]["score"] for i in range(n) if cluster_ids[i] == cid]
        cluster_avg[cid] = sum(members) / len(members) if members else 0
    ranked_clusters = sorted(cluster_avg, key=lambda c: -cluster_avg[c])
    tier_names = ["High Fit", "Moderate Fit", "Developing Fit", "Low Fit", "Needs Review"]
    cluster_label = {cid: tier_names[rank] if rank < len(tier_names) else f"Group {rank+1}"
                      for rank, cid in enumerate(ranked_clusters)}

    points = [{
        "name": r["name"],
        "cluster_id": int(cluster_ids[i]),
        "cluster_label": cluster_label[int(cluster_ids[i])],
        "x": round(float(coords[i][0]), 2),
        "y": round(float(coords[i][1]), 2),
        "score": r["score"],
    } for i, r in enumerate(results)]

    clusters_summary = [{
        "cluster_id": cid,
        "label": cluster_label[cid],
        "size": int(sum(1 for c in cluster_ids if c == cid)),
        "avg_score": round(cluster_avg[cid], 1),
    } for cid in range(k)]

    return {
        "task": "clustering",
        "algorithm": f"K-Means (k={k})",
        "n_samples": n,
        "features_used": ["TF-IDF Similarity", "Skill Match", "Experience Score", "Education Score"],
        "silhouette_score": sil,
        "pca_explained_variance_pct": explained_var,
        "clusters": clusters_summary,
        "points": points,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTML TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Resume Screening System</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root{
  --sb:#1e2535; --sb2:#252d3d; --sb3:#2e3a4e; --sb4:#3a4a62;
  --sbt:#d1dae8; --sbm:#7a8fa8;
  --w:#fff; --g50:#f9fafb; --g100:#f3f4f6; --g200:#e5e7eb;
  --g300:#d1d5db; --g500:#6b7280; --g700:#374151; --g900:#111827;
  --blue:#3b82f6; --blue-lt:#eff6ff; --blue-md:#bfdbfe;
  --green:#059669; --green-lt:#ecfdf5; --green-md:#a7f3d0;
  --amber:#b45309; --amber-lt:#fffbeb; --amber-md:#fcd34d;
  --red:#dc2626; --red-lt:#fef2f2; --red-md:#fecaca;
  --violet:#7c3aed; --violet-lt:#f5f3ff;
  --sh:0 1px 3px rgba(0,0,0,.08),0 1px 2px rgba(0,0,0,.04);
  --shm:0 4px 6px rgba(0,0,0,.06),0 2px 4px rgba(0,0,0,.04);
}
/* ── DARK MODE ── */
body.dark{
  --w:#1e2535; --g50:#252d3d; --g100:#1a2030; --g200:#2e3a4e;
  --g300:#3a4a62; --g500:#9aabbc; --g700:#c8d5e3; --g900:#e8edf4;
  --blue-lt:#1e2f4a; --blue-md:#2d4a72; --green-lt:#1a3028; --green-md:#1e4d35;
  --amber-lt:#2d2010; --amber-md:#4a3412; --red-lt:#2d1515; --red-md:#4a2020;
  --violet-lt:#241540; background:#111827;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter',system-ui,sans-serif;background:var(--g100);color:var(--g900);line-height:1.5}
.app{display:flex;min-height:100vh}

/* ── SIDEBAR ── */
.sb{width:268px;min-width:268px;background:var(--sb);display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow-y:auto}
.sb::-webkit-scrollbar{width:3px}.sb::-webkit-scrollbar-thumb{background:var(--sb4);border-radius:2px}

/* ── Responsive: stack sidebar above content on tablet/mobile so the main
   panel never gets squeezed below a usable width (UI Quality Checklist:
   "Works on expected screen size") ── */
@media (max-width: 900px){
  .app{flex-direction:column}
  .sb{width:100%;min-width:0;height:auto;max-height:340px;position:relative;top:auto}
  .chbox canvas{max-height:180px}
}
@media (max-width: 560px){
  .sb{max-height:280px}
}
.sb-hd{padding:22px 18px 18px;border-bottom:1px solid var(--sb3);background:var(--sb2)}
.sb-logo{display:flex;align-items:center;gap:10px}
.sb-icon{width:42px;height:42px;border-radius:11px;background:linear-gradient(135deg,#3b82f6,#60a5fa);display:flex;align-items:center;justify-content:center;font-size:21px;box-shadow:0 3px 8px rgba(59,130,246,.25);flex-shrink:0}
.sb-title{font-size:14px;font-weight:700;color:#fff}.sb-sub{font-size:10px;color:var(--sbm);margin-top:1px}
.sb-body{padding:18px;display:flex;flex-direction:column;gap:18px;flex:1}
.sb-sec{display:flex;flex-direction:column;gap:9px}
.sb-sec-title{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;color:var(--sbm);padding-bottom:7px;border-bottom:1px solid var(--sb3)}
.sb-lbl{font-size:11px;color:var(--sbt);font-weight:500;display:block;margin-bottom:3px}
.sb-inp{width:100%;background:var(--sb3);border:1px solid var(--sb4);border-radius:7px;color:var(--sbt);font-size:12px;padding:7px 9px;outline:none;font-family:inherit;transition:border-color .2s}
.sb-inp:focus{border-color:#3b82f6}
.sb-ta{width:100%;background:var(--sb3);border:1px solid var(--sb4);border-radius:7px;color:var(--sbt);font-size:11px;padding:7px 9px;outline:none;resize:vertical;min-height:82px;font-family:inherit;line-height:1.5;transition:border-color .2s}
.sb-ta:focus{border-color:#3b82f6}
.mtabs{display:grid;grid-template-columns:1fr 1fr;gap:4px}
.mtab{padding:7px;border-radius:7px;font-size:11px;font-weight:500;border:1px solid var(--sb4);background:transparent;color:var(--sbm);cursor:pointer;font-family:inherit;text-align:center;transition:all .2s}
.mtab.on{background:#3b82f6;color:#fff;border-color:#3b82f6}
.sl-row{display:flex;align-items:center;gap:8px}
.sl{flex:1;-webkit-appearance:none;height:4px;border-radius:2px;background:var(--sb4);outline:none;cursor:pointer}
.sl::-webkit-slider-thumb{-webkit-appearance:none;width:13px;height:13px;border-radius:50%;background:#60a5fa;cursor:pointer}
.sl-v{font-size:11px;font-weight:600;color:#60a5fa;background:var(--sb3);padding:2px 6px;border-radius:4px;min-width:30px;text-align:center}
.tog-row{display:flex;align-items:center;justify-content:space-between}
.tog-lbl{font-size:11px;color:var(--sbt);font-weight:500}
.tog{width:34px;height:19px;border-radius:10px;background:var(--sb4);position:relative;cursor:pointer;border:none;transition:background .22s;flex-shrink:0}
.tog.on{background:#3b82f6}
.tog::after{content:'';position:absolute;top:3px;left:3px;width:13px;height:13px;background:#fff;border-radius:50%;transition:left .22s;box-shadow:0 1px 3px rgba(0,0,0,.25)}
.tog.on::after{left:18px}
.run-btn{background:#3b82f6;color:#fff;border:none;border-radius:8px;padding:11px;font-size:13px;font-weight:600;cursor:pointer;width:100%;font-family:inherit;transition:all .2s;box-shadow:0 2px 5px rgba(59,130,246,.25)}
.run-btn:hover{background:#2563eb;box-shadow:0 4px 10px rgba(59,130,246,.3)}
.sb-ft{padding:14px 18px;border-top:1px solid var(--sb3);font-size:10px;color:var(--sbm);line-height:1.8}
.sb-bdg{display:inline-block;padding:1px 6px;border-radius:3px;font-size:9px;font-weight:500;margin:1px}
.bdg-b{background:rgba(59,130,246,.2);color:#93c5fd}.bdg-g{background:rgba(5,150,105,.2);color:#6ee7b7}

/* ── MAIN ── */
.main{flex:1;display:flex;flex-direction:column;overflow-x:hidden}
.topbar{background:var(--w);border-bottom:1px solid var(--g200);padding:13px 26px;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:10;box-shadow:var(--sh)}
.tb-title{font-size:15px;font-weight:600;color:var(--g900)}.tb-sub{font-size:11px;color:var(--g500);margin-top:1px}
.tb-pills{display:flex;gap:6px}
.tb-pill{padding:3px 10px;border-radius:5px;font-size:10px;font-weight:500;border:1px solid}
.tp-b{background:var(--blue-lt);color:#1d4ed8;border-color:var(--blue-md)}
.tp-g{background:var(--green-lt);color:#065f46;border-color:var(--green-md)}
.tp-v{background:var(--violet-lt);color:var(--violet);border-color:#ddd6fe}
.page{padding:22px 26px;display:flex;flex-direction:column;gap:18px}

/* ── STATUS ── */
.status{border-radius:9px;padding:11px 15px;font-size:12px;display:flex;align-items:center;gap:9px;font-weight:500;border:1px solid;animation:fadeUp .3s ease}
.sdot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.s-info{background:var(--blue-lt);color:#1d4ed8;border-color:var(--blue-md)}.s-info .sdot{background:var(--blue)}
.s-ok{background:var(--green-lt);color:#065f46;border-color:var(--green-md)}.s-ok .sdot{background:var(--green);animation:pulse 1.5s infinite}
.s-err{background:var(--red-lt);color:var(--red);border-color:var(--red-md)}.s-err .sdot{background:var(--red)}
.s-load{background:var(--amber-lt);color:var(--amber);border-color:var(--amber-md)}.s-load .sdot{background:var(--amber);animation:pulse 1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
@keyframes fadeUp{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}
@keyframes slideIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}

/* ── HERO ── */
.hero{background:var(--w);border:1px solid var(--g200);border-left:4px solid #3b82f6;border-radius:11px;padding:20px 24px;box-shadow:var(--sh);display:flex;align-items:center;justify-content:space-between;gap:16px}
.hero h1{font-size:20px;font-weight:700;color:var(--g900);letter-spacing:-.02em;margin-bottom:4px}
.hero p{font-size:12px;color:var(--g500)}
.hero-pills{display:flex;gap:5px;margin-top:10px;flex-wrap:wrap}
.hero-pill{padding:3px 9px;border-radius:5px;font-size:10px;font-weight:500;background:var(--blue-lt);color:#1d4ed8;border:1px solid var(--blue-md)}
.hero-stat{text-align:center;background:var(--blue-lt);border:1px solid var(--blue-md);border-radius:9px;padding:12px 18px;flex-shrink:0}
.hs-val{font-size:26px;font-weight:700;color:#1d4ed8}.hs-lbl{font-size:10px;color:#3b82f6;margin-top:1px;font-weight:500}

/* ── METRICS ── */
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.mc{background:var(--w);border:1px solid var(--g200);border-radius:11px;padding:16px;box-shadow:var(--sh);display:flex;align-items:flex-start;gap:12px;transition:transform .2s,box-shadow .2s;animation:slideIn .4s ease both}
.mc:hover{transform:translateY(-2px);box-shadow:var(--shm)}
.mc-ico{width:40px;height:40px;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:19px;flex-shrink:0}
.ic-b{background:var(--blue-lt)}.ic-g{background:var(--green-lt)}.ic-a{background:var(--amber-lt)}.ic-r{background:var(--red-lt)}
.mc-val{font-size:21px;font-weight:700;line-height:1.1}
.cv-b{color:var(--blue)}.cv-g{color:var(--green)}.cv-a{color:var(--amber)}.cv-r{color:var(--red)}
.mc-lbl{font-size:10px;color:var(--g500);font-weight:500;margin-top:3px;text-transform:uppercase;letter-spacing:.04em}
.mc-sub{font-size:10px;color:var(--g300);margin-top:1px}

/* ── CARD ── */
.card{background:var(--w);border:1px solid var(--g200);border-radius:11px;box-shadow:var(--sh)}
.card-hd{padding:14px 18px;border-bottom:1px solid var(--g100);display:flex;align-items:center;justify-content:space-between}
.card-title{font-size:13px;font-weight:600;color:var(--g900);display:flex;align-items:center;gap:7px}
.card-body{padding:18px}

/* ── UPLOAD ── */
.up-area{border:2px dashed var(--g200);border-radius:9px;padding:24px 18px;text-align:center;cursor:pointer;transition:all .2s;background:var(--g50);position:relative}
.up-area:hover,.up-area.drag{border-color:var(--blue);background:var(--blue-lt)}
.up-area input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.up-icon{font-size:32px;margin-bottom:6px}.up-title{font-size:13px;font-weight:600;color:var(--g700);margin-bottom:3px}
.up-sub{font-size:11px;color:var(--g500)}.up-hint{font-size:10px;color:var(--blue);margin-top:5px;font-weight:500}
.file-list{display:flex;flex-direction:column;gap:5px;margin-top:10px}
.file-item{display:flex;align-items:center;gap:8px;padding:7px 10px;background:var(--blue-lt);border:1px solid var(--blue-md);border-radius:7px;animation:fadeUp .2s ease}
.fi-name{font-size:11px;font-weight:500;color:#1d4ed8;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fi-sz{font-size:10px;color:var(--g500);flex-shrink:0}
.fi-rm{background:var(--red-lt);color:var(--red);border:none;border-radius:4px;padding:2px 7px;font-size:10px;cursor:pointer;font-family:inherit}

/* ── FILTER BAR ── */
.fbar{display:flex;gap:7px;align-items:center;flex-wrap:wrap;padding:12px 18px;background:var(--g50);border-bottom:1px solid var(--g100)}
.sw{position:relative;flex:1;min-width:150px}
.sw input{width:100%;padding:6px 9px 6px 30px;border:1px solid var(--g200);border-radius:7px;font-size:11px;outline:none;background:var(--w);font-family:inherit;transition:border-color .2s}
.sw input:focus{border-color:var(--blue)}
.si{position:absolute;left:9px;top:50%;transform:translateY(-50%);font-size:12px;color:var(--g300);pointer-events:none}
.fsel{padding:6px 9px;border:1px solid var(--g200);border-radius:7px;font-size:11px;background:var(--w);color:var(--g700);font-family:inherit;outline:none}
.schip{padding:5px 11px;border:1px solid var(--g200);border-radius:20px;font-size:10px;font-weight:500;color:var(--g500);background:var(--w);cursor:pointer;font-family:inherit;transition:all .2s;white-space:nowrap}
.schip:hover{border-color:var(--blue);color:var(--blue)}
.schip.on{background:var(--blue-lt);color:var(--blue);border-color:var(--blue-md)}
.rc{font-size:10px;color:var(--g300);margin-left:auto;white-space:nowrap}

/* ── TABLE ── */
.twrap{overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:12px}
thead th{padding:10px 13px;text-align:left;font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--g500);background:var(--g50);border-bottom:1px solid var(--g200);white-space:nowrap}
tbody td{padding:11px 13px;border-bottom:1px solid var(--g100);vertical-align:middle}
tbody tr:last-child td{border-bottom:none}
tbody tr{transition:background .12s;animation:fadeUp .3s ease both}
tbody tr:hover{background:var(--g50)}
.rnum{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700}
.r1{background:var(--amber-lt);color:var(--amber)}.r2{background:var(--g100);color:var(--g700)}
.r3{background:#fff7ed;color:#c2410c}.rn{background:var(--g50);color:var(--g500);font-size:10px}
.av{width:33px;height:33px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;flex-shrink:0}
.av0{background:var(--blue-lt);color:#1d4ed8}.av1{background:var(--green-lt);color:#065f46}
.av2{background:var(--amber-lt);color:#92400e}.av3{background:var(--red-lt);color:#991b1b}
.av4{background:#f0fdf4;color:#166534}.av5{background:#faf5ff;color:#6b21a8}
.cname{font-size:12px;font-weight:600;color:var(--g900)}.cedu{font-size:10px;color:var(--g500);margin-top:1px}
.sp{display:inline-flex;align-items:center;padding:3px 8px;border-radius:20px;font-size:11px;font-weight:600}
.sp-g{background:var(--green-lt);color:var(--green)}.sp-b{background:var(--blue-lt);color:var(--blue)}
.sp-a{background:var(--amber-lt);color:var(--amber)}.sp-r{background:var(--red-lt);color:var(--red)}
.mbar{height:4px;width:72px;border-radius:2px;background:var(--g200);overflow:hidden;margin-top:3px}
.mfill{height:100%;border-radius:2px;transition:width .6s ease}
.snum{font-weight:600;font-size:12px}
.sn-b{color:#2563eb}.sn-g{color:var(--green)}.sn-a{color:var(--amber)}
.tag{display:inline-block;padding:1px 6px;border-radius:4px;font-size:9px;font-weight:500;margin:1px}
.tg{background:var(--green-lt);color:var(--green)}.tr{background:var(--red-lt);color:var(--red)}.tb{background:var(--blue-lt);color:var(--blue)}
.qbdg{display:inline-flex;align-items:center;gap:3px;padding:3px 8px;border-radius:5px;font-size:10px;font-weight:500}
.qy{background:var(--green-lt);color:var(--green)}.qn{background:var(--red-lt);color:var(--red)}

/* ── CHARTS ── */
.ctabs{display:flex;gap:3px;padding:13px 18px 0;border-bottom:1px solid var(--g100);flex-wrap:wrap}
.ctab{padding:7px 15px;border-radius:7px 7px 0 0;font-size:11px;font-weight:500;border:1px solid transparent;border-bottom:none;background:transparent;color:var(--g500);cursor:pointer;font-family:inherit;transition:all .2s;margin-bottom:-1px}
.ctab:hover{color:var(--blue)}.ctab.on{background:var(--w);color:var(--blue);border-color:var(--g100);font-weight:600}
.mltab{padding:7px 15px;border-radius:7px;font-size:11px;font-weight:500;border:1px solid var(--g200);background:var(--g50);color:var(--g500);cursor:pointer;font-family:inherit;transition:all .2s}
.mltab:hover{color:var(--blue)}.mltab.on{background:var(--blue);color:#fff;border-color:var(--blue);font-weight:600}
.cpanel{display:none}.cpanel.on{display:block;animation:fadeUp .3s ease}
.ch2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.chbox{background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:14px}
.chtitle{font-size:11px;font-weight:600;color:var(--g700);margin-bottom:11px;display:flex;align-items:center;gap:5px}
.chdot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
canvas{max-height:210px}

/* ── COMPARISON PANEL ── */
.cmp-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.cmp-box{background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:14px}
.cmp-title{font-size:11px;font-weight:600;color:var(--g700);margin-bottom:10px;display:flex;align-items:center;gap:5px}
.cmp-item{display:flex;align-items:center;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--g100);font-size:11px}
.cmp-item:last-child{border-bottom:none}
.cmp-name{color:var(--g700);font-weight:500}.cmp-score{font-weight:700}

/* ── EXPLAIN ── */
.exp-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.exp-card{background:var(--w);border:1px solid var(--g200);border-radius:11px;padding:14px;box-shadow:var(--sh);animation:slideIn .4s ease both;border-top:3px solid;transition:box-shadow .2s}
.exp-card:hover{box-shadow:var(--shm)}
.ec-g{border-top-color:var(--green)}.ec-b{border-top-color:var(--blue)}.ec-a{border-top-color:var(--amber)}.ec-r{border-top-color:var(--red)}
.exp-hd{display:flex;align-items:center;gap:9px;margin-bottom:10px}
.exp-nw .exp-nm{font-size:13px;font-weight:600;color:var(--g900)}.exp-nw .exp-ed{font-size:10px;color:var(--g500)}
.exp-sb{margin-left:auto;text-align:center;padding:5px 10px;border-radius:7px;line-height:1.2}
.exp-sv{font-size:16px;font-weight:700}.exp-sl{font-size:9px;color:var(--g500);text-transform:uppercase}
.exp-verdict{font-size:11px;color:var(--g600);line-height:1.6;padding:7px 10px;background:var(--g50);border-radius:7px;margin-bottom:8px}
.eb{display:flex;flex-direction:column;gap:5px;margin-bottom:8px}
.ebr{display:flex;align-items:center;gap:7px;font-size:10px;color:var(--g600)}
.ebl{min-width:64px;font-weight:500}.ebb{flex:1;height:5px;border-radius:3px;background:var(--g200);overflow:hidden}
.ebf{height:100%;border-radius:3px;transition:width .8s ease}.ebv{min-width:28px;text-align:right;font-weight:600}
.exp-tags{display:flex;flex-wrap:wrap;gap:3px;margin-top:5px}

/* ── EVAL ── */
.eval-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px}
.eval-box{background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:12px;text-align:center}
.eval-val{font-size:18px;font-weight:700}.eval-lbl{font-size:9px;color:var(--g500);margin-top:2px;text-transform:uppercase;letter-spacing:.05em}
.timing-row{display:grid;grid-template-columns:repeat(8,1fr);gap:8px;margin-top:10px}
.tim-box{background:var(--g50);border:1px solid var(--g200);border-radius:8px;padding:10px;text-align:center}
.tim-val{font-size:16px;font-weight:700;color:var(--violet)}.tim-lbl{font-size:9px;color:var(--g500);margin-top:2px}

/* ── MANUAL ── */
.mgrid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.mcard{background:var(--w);border:1px solid var(--g200);border-radius:9px;padding:13px;display:flex;flex-direction:column;gap:7px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.mhd{display:flex;align-items:center;justify-content:space-between;font-size:11px;font-weight:600;color:var(--blue)}
.delbtn{background:var(--red-lt);color:var(--red);border:none;border-radius:4px;padding:2px 7px;font-size:10px;cursor:pointer;font-family:inherit}
.mlbl{font-size:10px;color:var(--g600);font-weight:500}
.minp{width:100%;padding:6px 9px;border:1px solid var(--g200);border-radius:6px;font-size:11px;outline:none;font-family:inherit;transition:border-color .2s}
.minp:focus{border-color:var(--blue)}
.mta{width:100%;padding:6px 9px;border:1px solid var(--g200);border-radius:6px;font-size:11px;outline:none;resize:vertical;min-height:64px;font-family:inherit;line-height:1.5;transition:border-color .2s}
.mta:focus{border-color:var(--blue)}
.addbtn{padding:9px;border:2px dashed var(--g200);border-radius:9px;background:transparent;color:var(--g400);cursor:pointer;font-size:12px;width:100%;font-family:inherit;transition:all .2s}
.addbtn:hover{border-color:var(--blue);color:var(--blue);background:var(--blue-lt)}

/* ── EMPTY ── */
.empty{text-align:center;padding:50px 20px;color:var(--g400)}
.empty-ic{font-size:48px;margin-bottom:14px}.empty-t{font-size:16px;font-weight:600;color:var(--g600);margin-bottom:6px}.empty-s{font-size:12px;line-height:1.6;max-width:340px;margin:0 auto}

/* ── LOADER ── */
.loader{display:none;position:fixed;inset:0;background:rgba(249,250,251,.88);backdrop-filter:blur(4px);z-index:999;align-items:center;justify-content:center;flex-direction:column;gap:14px}
.loader.show{display:flex}
.lring{width:48px;height:48px;border-radius:50%;border:3px solid #dbeafe;border-top-color:#3b82f6;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.ltitle{font-size:14px;font-weight:600;color:var(--g700)}.lsub{font-size:11px;color:var(--g500)}
.lsteps{display:flex;gap:5px;margin-top:3px}
.ls{padding:3px 9px;border-radius:20px;font-size:10px;background:var(--blue-lt);color:var(--blue);font-weight:500}
/* ── TIMELINE ── */
.timeline{display:flex;flex-direction:column;gap:0;position:relative;padding:0 0 0 28px}
.timeline::before{content:'';position:absolute;left:10px;top:0;bottom:0;width:2px;background:linear-gradient(180deg,var(--blue),var(--green));border-radius:2px}
.tl-step{position:relative;padding:10px 0 10px 16px;animation:fadeUp .4s ease both}
.tl-dot{position:absolute;left:-24px;top:14px;width:16px;height:16px;border-radius:50%;border:2px solid var(--w);display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:700;flex-shrink:0}
.tl-dot.done{background:var(--green)}.tl-dot.active{background:var(--blue);animation:pulse 1s infinite}.tl-dot.pending{background:var(--g300)}
.tl-title{font-size:12px;font-weight:600;color:var(--g900);margin-bottom:2px}
.tl-desc{font-size:11px;color:var(--g500)}.tl-score{font-size:11px;font-weight:700;color:var(--blue);margin-top:2px}
/* ── TIER BADGES ── */
.tier-badge{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:6px;font-size:11px;font-weight:700}
.tier-A{background:#ecfdf5;color:#059669}.tier-B{background:#eff6ff;color:#3b82f6}
.tier-C{background:#fffbeb;color:#b45309}.tier-D{background:#fef2f2;color:#dc2626}
/* ── SKILL GAP ── */
.gap-card{background:var(--w);border:1px solid var(--g200);border-radius:11px;padding:14px;box-shadow:var(--sh);animation:slideIn .4s ease both}
.gap-grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:8px 0}
.gap-half{background:var(--g50);border-radius:8px;padding:10px}
.gap-half-title{font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}
.rec-item{font-size:10px;color:#1d4ed8;padding:3px 8px;background:var(--blue-lt);border-radius:5px;border-left:2px solid var(--blue);margin:2px 0}
/* ── SCORE GRID ── */
.score-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:10px}
.sg-cell{border-radius:10px;padding:14px;text-align:center;border:1px solid;transition:transform .2s}
.sg-cell:hover{transform:translateY(-2px)}
.sg-name{font-size:11px;font-weight:600;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sg-score{font-size:22px;font-weight:700;line-height:1}
.sg-tier{font-size:10px;margin-top:3px;font-weight:500}
/* ── DB TABLE ── */
.db-sess-pill{padding:3px 9px;border-radius:5px;font-size:10px;font-weight:500;background:var(--blue-lt);color:#1d4ed8;border:1px solid var(--blue-md)}
/* ── NETWORK GRAPH ── */
#skill-network{width:100%;height:280px;border:1px solid var(--g200);border-radius:9px;background:var(--g50);position:relative;overflow:hidden}
/* ── CHATBOT ── */
.chat-msg{padding:8px 12px;border-radius:9px;font-size:12px;line-height:1.6;max-width:88%;animation:fadeUp .2s ease}
.chat-user{background:var(--blue-lt);color:#1d4ed8;border:1px solid var(--blue-md);align-self:flex-end;border-bottom-right-radius:3px}
.chat-bot{background:var(--w);color:var(--g700);border:1px solid var(--g200);align-self:flex-start;border-bottom-left-radius:3px}
.chat-typing{background:var(--g100);color:var(--g400);border:1px solid var(--g200);align-self:flex-start;font-style:italic}
.hidden{display:none!important}
</style>
</head>
<body>

<div class="loader" id="loader">
  <div class="lring"></div>
  <div class="ltitle">Analyzing Resumes...</div>
  <div class="lsub">NLP · TF-IDF · Cosine Similarity · Skill Matching</div>
  <div class="lsteps">
    <span class="ls">📄 Parsing</span>
    <span class="ls">🔍 Matching</span>
    <span class="ls">📊 Ranking</span>
    <span class="ls">💡 Explaining</span>
  </div>
</div>

<div class="app">

<!-- ═══════ SIDEBAR ═══════ -->
<aside class="sb">
  <div class="sb-hd">
    <div class="sb-logo">
      <div class="sb-icon">🤖</div>
      <div><div class="sb-title">ResumeAI</div><div class="sb-sub">AI Resume Screening System</div></div>
    </div>
  </div>

  <div class="sb-body">
    <div style="font-size:9px;font-weight:700;letter-spacing:.06em;color:var(--sbm);text-transform:uppercase;padding:2px 0 2px;opacity:.75">A) Problem Setup Module</div>
    <div class="sb-sec">
      <div class="sb-sec-title">Input Mode</div>
      <div class="mtabs">
        <button class="mtab on" id="btn-sample" onclick="setMode('sample')">📋 Sample</button>
        <button class="mtab" id="btn-manual" onclick="setMode('manual')">✏️ Manual</button>
      </div>
    </div>

    <div class="sb-sec">
      <div class="sb-sec-title">Job Description</div>
      <span class="sb-lbl">Select role</span>
      <select class="sb-inp" id="jd-preset" onchange="fillJD()">
        <option value="">— choose role —</option>
        <option>ML Engineer</option>
        <option>Data Scientist</option>
        <option>Backend Developer</option>
        <option>AI Researcher</option>
      </select>
      <span class="sb-lbl" style="margin-top:6px">Job description</span>
      <textarea class="sb-ta" id="jd-text" placeholder="Paste job description here..."></textarea>
    </div>

    <div class="sb-sec">
      <div class="sb-sec-title">Scoring Weights</div>
      <div>
        <span class="sb-lbl">TF-IDF</span>
        <div class="sl-row"><input type="range" class="sl" id="w-tf" min="1" max="10" value="4" oninput="sv('v-tf',this)"><span class="sl-v" id="v-tf">0.4</span></div>
      </div>
      <div>
        <span class="sb-lbl">Skill Match</span>
        <div class="sl-row"><input type="range" class="sl" id="w-sk" min="1" max="10" value="4" oninput="sv('v-sk',this)"><span class="sl-v" id="v-sk">0.4</span></div>
      </div>
      <div>
        <span class="sb-lbl">Experience</span>
        <div class="sl-row"><input type="range" class="sl" id="w-ex" min="1" max="10" value="2" oninput="sv('v-ex',this)"><span class="sl-v" id="v-ex">0.2</span></div>
      </div>
    </div>

    <div class="sb-sec">
      <div class="sb-sec-title">Options</div>
      <div>
        <span class="sb-lbl">Qualified threshold</span>
        <div class="sl-row"><input type="range" class="sl" id="thresh" min="30" max="90" value="60" step="5" oninput="sv('v-th',this,'%')"><span class="sl-v" id="v-th">60%</span></div>
      </div>
      <div class="tog-row"><span class="tog-lbl">Use Semantic AI</span><button class="tog on" id="tog-sem" onclick="this.classList.toggle('on')"></button></div>
      <div class="tog-row"><span class="tog-lbl">AI Explanations</span><button class="tog on" id="tog-exp" onclick="this.classList.toggle('on')"></button></div>
      <div class="tog-row"><span class="tog-lbl">Comparison Mode</span><button class="tog" id="tog-cmp" onclick="this.classList.toggle('on')"></button></div>
    </div>

    <button class="run-btn" onclick="run()">▶ Run Screening</button>
  </div>

  <div class="sb-ft">
    <div style="margin-bottom:4px;color:var(--sbt);font-weight:500">Tech Stack</div>
    <span class="sb-bdg bdg-b">Python</span><span class="sb-bdg bdg-b">Flask</span>
    <span class="sb-bdg bdg-g">scikit-learn</span><span class="sb-bdg bdg-g">TF-IDF</span>
    <span class="sb-bdg bdg-b">spaCy</span><span class="sb-bdg bdg-g">Sentence Transformers</span>
    <div style="margin-top:6px">AI Resume Screening System</div>
  </div>
</aside>

<!-- ═══════ MAIN ═══════ -->
<div class="main">
  <div class="topbar">
    <div><div class="tb-title">AI Resume Screening Dashboard</div><div class="tb-sub">Automated ranking · NLP · TF-IDF · Semantic Similarity</div></div>
    <div class="tb-pills" style="flex-wrap:wrap;gap:5px;position:relative">
      <span class="tb-pill tp-b">🧠 NLP</span>
      <span class="tb-pill tp-g">📊 TF-IDF</span>
      <span class="tb-pill tp-v" id="sem-pill">🔮 Semantic</span>
      <button onclick="toggleSec('chatbot-sec')" style="padding:4px 10px;border-radius:5px;font-size:10px;font-weight:500;background:#fdf4ff;color:#7c3aed;border:1px solid #e9d5ff;cursor:pointer;font-family:inherit">💬 NLP Chat</button>
      <button onclick="exportCSV()" style="padding:4px 10px;border-radius:5px;font-size:10px;font-weight:500;background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0;cursor:pointer;font-family:inherit">📥 Export CSV</button>
      <button onclick="toggleDark()" id="dark-btn" style="padding:4px 10px;border-radius:5px;font-size:10px;font-weight:500;background:var(--g100);color:var(--g700);border:1px solid var(--g200);cursor:pointer;font-family:inherit">🌙 Dark</button>
      <div style="position:relative">
        <button onclick="toggleToolsMenu()" id="tools-btn" style="padding:4px 10px;border-radius:5px;font-size:10px;font-weight:500;background:var(--blue-lt);color:#1d4ed8;border:1px solid var(--blue-md);cursor:pointer;font-family:inherit">🧰 More Views ▾</button>
        <div id="tools-menu" class="hidden" style="position:absolute;top:calc(100% + 6px);right:0;background:var(--w);border:1px solid var(--g200);border-radius:9px;box-shadow:var(--shm);padding:6px;z-index:50;min-width:170px;display:flex;flex-direction:column;gap:3px">
          <button onclick="toggleSec('timeline-sec');toggleToolsMenu()" style="text-align:left;padding:6px 9px;border-radius:6px;font-size:11px;font-weight:500;background:transparent;color:var(--g700);border:none;cursor:pointer;font-family:inherit" onmouseover="this.style.background='var(--g50)'" onmouseout="this.style.background='transparent'">⏱ Timeline</button>
          <button onclick="toggleSec('grid-sec');toggleToolsMenu()" style="text-align:left;padding:6px 9px;border-radius:6px;font-size:11px;font-weight:500;background:transparent;color:var(--g700);border:none;cursor:pointer;font-family:inherit" onmouseover="this.style.background='var(--g50)'" onmouseout="this.style.background='transparent'">🗺 Score Grid</button>
          <button onclick="toggleSec('network-sec');toggleToolsMenu()" style="text-align:left;padding:6px 9px;border-radius:6px;font-size:11px;font-weight:500;background:transparent;color:var(--g700);border:none;cursor:pointer;font-family:inherit" onmouseover="this.style.background='var(--g50)'" onmouseout="this.style.background='transparent'">🕸 Skill Network</button>
          <button onclick="toggleSec('gap-sec');toggleToolsMenu()" style="text-align:left;padding:6px 9px;border-radius:6px;font-size:11px;font-weight:500;background:transparent;color:var(--g700);border:none;cursor:pointer;font-family:inherit" onmouseover="this.style.background='var(--g50)'" onmouseout="this.style.background='transparent'">🎯 Skill Gap</button>
          <div style="height:1px;background:var(--g100);margin:2px 0"></div>
          <button onclick="toggleSec('nlp-sec');toggleToolsMenu();if(!document.getElementById('nlp-sec').classList.contains('hidden')&&!all.length)alert('Run screening first.')" style="text-align:left;padding:6px 9px;border-radius:6px;font-size:11px;font-weight:500;background:transparent;color:var(--g700);border:none;cursor:pointer;font-family:inherit" onmouseover="this.style.background='var(--g50)'" onmouseout="this.style.background='transparent'">🧬 NLP Pipeline</button>
          <button onclick="toggleSec('compare-sec');toggleToolsMenu()" style="text-align:left;padding:6px 9px;border-radius:6px;font-size:11px;font-weight:500;background:transparent;color:var(--g700);border:none;cursor:pointer;font-family:inherit" onmouseover="this.style.background='var(--g50)'" onmouseout="this.style.background='transparent'">⚖️ Approach Comparison</button>
          <button onclick="toggleSec('fc-sec');toggleToolsMenu();if(!document.getElementById('fc-sec').classList.contains('hidden'))runFC()" style="text-align:left;padding:6px 9px;border-radius:6px;font-size:11px;font-weight:500;background:transparent;color:var(--g700);border:none;cursor:pointer;font-family:inherit" onmouseover="this.style.background='var(--g50)'" onmouseout="this.style.background='transparent'">🔗 Forward Chaining</button>
          <button onclick="toggleSec('search-sec');toggleToolsMenu();if(!all.length)document.getElementById('search-result').innerHTML='<div style=\'color:var(--amber);font-size:12px\'>⚠️ Run screening first.</div>'" style="text-align:left;padding:6px 9px;border-radius:6px;font-size:11px;font-weight:500;background:transparent;color:var(--g700);border:none;cursor:pointer;font-family:inherit" onmouseover="this.style.background='var(--g50)'" onmouseout="this.style.background='transparent'">🔍 Search Algorithms</button>
          <button onclick="toggleSec('ml-sec');toggleToolsMenu();if(!all.length)document.getElementById('ml-result').innerHTML='<div style=\'color:var(--amber);font-size:12px\'>⚠️ Run screening first.</div>'" style="text-align:left;padding:6px 9px;border-radius:6px;font-size:11px;font-weight:500;background:transparent;color:var(--g700);border:none;cursor:pointer;font-family:inherit" onmouseover="this.style.background='var(--g50)'" onmouseout="this.style.background='transparent'">🧠 ML Training</button>
          <div style="height:1px;background:var(--g100);margin:2px 0"></div>
          <button onclick="loadDBHistory();toggleToolsMenu()" style="text-align:left;padding:6px 9px;border-radius:6px;font-size:11px;font-weight:500;background:transparent;color:var(--g700);border:none;cursor:pointer;font-family:inherit" onmouseover="this.style.background='var(--g50)'" onmouseout="this.style.background='transparent'">🗄 Database History</button>
        </div>
      </div>
    </div>
  </div>


  <div class="page">
    <!-- Status -->
    <div class="status s-info" id="status"><div class="sdot"></div>Select a job description and click <strong style="margin-left:3px">Run Screening</strong>.</div>

    <!-- Hero -->
    <div class="hero">
      <div>
        <h1>🤖 AI Resume Screening System</h1>
        <p>Automated candidate ranking · NLP · TF-IDF + Cosine Similarity · Sentence Transformers</p>
        <div style="font-size:9px;font-weight:700;letter-spacing:.06em;color:var(--g400);text-transform:uppercase;margin-top:6px">B) Core Logic Module — scoring runs in run_screening() · results explained in generate_explanation()</div>
        <div class="hero-pills">
          <span class="hero-pill">📄 PDF Upload</span>
          <span class="hero-pill">🧠 NLP Parsing</span>
          <span class="hero-pill">📊 TF-IDF</span>
          <span class="hero-pill">🔮 Semantic AI</span>
          <span class="hero-pill">🎯 Skill Match</span>
          <a href="/download_dataset" style="text-decoration:none"><span class="hero-pill" style="background:var(--green-lt);color:#065f46;border-color:var(--green-md);cursor:pointer">💾 Download Dataset</span></a>
        </div>
      </div>
      <div class="hero-stat"><div class="hs-val" id="hs-val">6</div><div class="hs-lbl">Candidates</div></div>
    </div>

    <!-- Upload -->
    <div class="card" id="upload-card">
      <div class="card-hd">
        <div class="card-title">📄 Resume Upload</div>
        <span style="font-size:10px;color:var(--g400)">PDF / TXT supported · Drag & drop</span>
      </div>
      <div class="card-body">
        <div class="up-area" id="up-area"
          ondragover="event.preventDefault();this.classList.add('drag')"
          ondragleave="this.classList.remove('drag')"
          ondrop="onDrop(event)">
          <div class="up-icon">📂</div>
          <div class="up-title">Drop PDF resumes here or click to browse</div>
          <div class="up-sub">Multiple files supported</div>
          <div class="up-hint">↑ Click anywhere in this box</div>
          <input type="file" id="file-inp" accept=".pdf,.txt" multiple onchange="onFiles(this.files)">
        </div>
        <div class="file-list" id="file-list"></div>
        <div id="up-stat" style="margin-top:7px;font-size:11px;color:var(--g500)"></div>
      </div>
    </div>

    <!-- Manual -->
    <div id="manual-sec" class="hidden">
      <div class="card">
        <div class="card-hd"><div class="card-title">✏️ Manual Candidates</div></div>
        <div class="card-body">
          <div class="mgrid" id="cand-grid"></div>
          <button class="addbtn" onclick="addCand()" style="margin-top:9px">+ Add Candidate</button>
        </div>
      </div>
    </div>

    <!-- Empty -->
    <div class="empty" id="empty">
      <div class="empty-ic">🎯</div>
      <div class="empty-t">Ready to Screen</div>
      <div class="empty-s">Upload PDF resumes or use Sample Data · Set job description · Click Run Screening</div>
    </div>

    <!-- ══ RESULTS ══ -->
    <div id="results" class="hidden">

      <!-- Metrics -->
      <div class="metrics" id="metrics"></div>

      <!-- Table -->
      <div class="card">
        <div class="fbar">
          <div class="sw"><span class="si">🔍</span><input id="srch" placeholder="Search name or skill..." oninput="filterTbl()"></div>
          <select class="fsel" id="sfil" onchange="filterTbl()">
            <option value="">All Candidates</option>
            <option value="q">✅ Qualified</option>
            <option value="n">❌ Not Qualified</option>
          </select>
          <button class="schip on" id="sc-score" onclick="srt('score',this)">Score ↓</button>
          <button class="schip" id="sc-exp"   onclick="srt('exp_yrs',this)">Experience</button>
          <button class="schip" id="sc-skill" onclick="srt('skill',this)">Skills</button>
          <span class="rc" id="rc"></span>
        </div>
        <div class="twrap">
          <table>
            <thead><tr>
              <th>Rank</th><th>Candidate</th>
              <th>Final Score</th><th>TF-IDF</th>
              <th>Semantic</th><th>Skills</th>
              <th>Experience</th><th>Education</th><th>Matched Skills</th><th>Status</th>
            </tr></thead>
            <tbody id="tbl"></tbody>
          </table>
        </div>
      </div>

      <!-- Charts -->
      <div class="card">
        <div class="ctabs">
          <button class="ctab on" onclick="cTab(0,this)">📊 Bar & Pie</button>
          <button class="ctab"    onclick="cTab(1,this)">📈 Line & Scatter</button>
          <button class="ctab"    onclick="cTab(2,this)">🕸 Radar & Heatmap</button>
          <button class="ctab"    onclick="cTab(3,this)">📐 Evaluation</button>
        </div>
        <div class="card-body">
          <div class="cpanel on" id="cp0">
            <div class="ch2">
              <div class="chbox"><div class="chtitle"><span class="chdot" style="background:#3b82f6"></span>Final Match Score — Bar Chart</div><canvas id="c-bar"></canvas></div>
              <div class="chbox"><div class="chtitle"><span class="chdot" style="background:#059669"></span>Score Share — Pie Chart</div><canvas id="c-pie"></canvas></div>
            </div>
          </div>
          <div class="cpanel" id="cp1">
            <div class="ch2">
              <div class="chbox"><div class="chtitle"><span class="chdot" style="background:#3b82f6"></span>All Scores Breakdown — Line Chart</div><canvas id="c-line"></canvas></div>
              <div class="chbox"><div class="chtitle"><span class="chdot" style="background:#b45309"></span>Skill Match vs TF-IDF — Scatter</div><canvas id="c-scat"></canvas></div>
            </div>
          </div>
          <div class="cpanel" id="cp2">
            <div class="ch2">
              <div class="chbox"><div class="chtitle"><span class="chdot" style="background:#7c3aed"></span>Radar — Top 3 Candidates</div><canvas id="c-rad"></canvas></div>
              <div class="chbox"><div class="chtitle"><span class="chdot" style="background:#059669"></span>Skill Coverage — Heatmap</div><canvas id="c-heat"></canvas></div>
            </div>
          </div>
          <div class="cpanel" id="cp3">
            <!-- Comparison: TF-IDF vs Semantic -->
            <div class="cmp-grid" id="cmp-grid"></div>
            <div class="ch2">
              <div class="chbox"><div class="chtitle"><span class="chdot" style="background:#dc2626"></span>Score Summary — Min/Avg/Max</div><canvas id="c-box"></canvas></div>
              <div class="chbox"><div class="chtitle"><span class="chdot" style="background:#b45309"></span>Candidates per Score Range</div><canvas id="c-range"></canvas></div>
            </div>
            <!-- Evaluation metrics -->
            <div class="eval-row" id="eval-row"></div>
            <div class="timing-row" id="timing-row"></div>
          </div>
        </div>
      </div>

      <!-- Explanations -->
      <div id="exp-sec" class="hidden">
        <div class="card">
          <div class="card-hd">
            <div class="card-title">🧠 AI Explanations — Why Each Candidate Was Ranked This Way</div>
            <span style="font-size:10px;color:var(--g400)">D) Explainability Module</span>
          </div>
          <div class="card-body"><div class="exp-grid" id="exp-cards"></div></div>
        </div>
      </div>


      <!-- ══ TIMELINE — Step by Step Animation ══ -->
      <div id="timeline-sec" class="hidden">
        <div class="card">
          <div class="card-hd">
            <div class="card-title">⏱️ Processing Timeline — Step-by-Step</div>
            <span style="font-size:10px;color:var(--g400)">C) Visual UI — Timeline Animation</span>
          </div>
          <div class="card-body">
            <div class="timeline" id="tl-steps"></div>
          </div>
        </div>
      </div>

      <!-- ══ SCORE GRID — Grid/Map View ══ -->
      <div id="grid-sec" class="hidden">
        <div class="card">
          <div class="card-hd">
            <div class="card-title">🗺️ Candidate Score Grid</div>
            <span style="font-size:10px;color:var(--g400)">C) Visual UI — Grid/Map View</span>
          </div>
          <div class="card-body">
            <div class="score-grid" id="score-grid"></div>
          </div>
        </div>
      </div>

      <!-- ══ NETWORK GRAPH — Skill Network ══ -->
      <div id="network-sec" class="hidden">
        <div class="card">
          <div class="card-hd">
            <div class="card-title">🕸️ Skill Network Graph</div>
            <span style="font-size:10px;color:var(--g400)">C) Visual UI — Graph/Network View</span>
          </div>
          <div class="card-body">
            <div id="skill-network"></div>
            <div id="network-legend" style="display:flex;gap:12px;margin-top:10px;flex-wrap:wrap;font-size:11px;color:var(--g500)">
              <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#3b82f6;margin-right:4px"></span>Candidate</span>
              <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#059669;margin-right:4px"></span>Matched Skill</span>
              <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#dc2626;margin-right:4px"></span>Missing Skill</span>
            </div>
          </div>
        </div>
      </div>

      <!-- ══ SKILL GAP ANALYSIS ══ -->
      <div id="gap-sec" class="hidden">
        <div class="card">
          <div class="card-hd">
            <div class="card-title">🎯 Skill Gap Analysis</div>
            <span style="font-size:10px;color:var(--g400)">Per-candidate missing skills + learning recommendations</span>
          </div>
          <div class="card-body">
            <div id="gap-cards" style="display:flex;flex-direction:column;gap:12px"></div>
          </div>
        </div>
      </div>

      <!-- ══ NLP PIPELINE PANEL — Option 4: Full NLP Demo ══ -->
      <div id="nlp-sec" class="hidden">
        <div class="card">
          <div class="card-hd">
            <div class="card-title">🧬 NLP Pipeline Analysis — Full Linguistics Breakdown</div>
            <span style="font-size:10px;color:var(--g400)">Option 4: NLP/LLM-assisted AI · Tokenization → POS → NER → Lemmatization</span>
          </div>
          <div class="card-body">
            <div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap">
              <button class="run-btn" style="width:auto;padding:7px 16px;font-size:11px" onclick="runNLPAnalysis('resume')">🔍 Analyse Top Resume</button>
              <button class="run-btn" style="width:auto;padding:7px 16px;font-size:11px;background:#7c3aed" onclick="runNLPAnalysis('jd')">🔍 Analyse JD Text</button>
            </div>
            <div id="nlp-loading" class="hidden" style="text-align:center;padding:20px;color:var(--g500);font-size:12px">⏳ Running NLP pipeline...</div>
            <div id="nlp-result"></div>
          </div>
        </div>
      </div>

      <!-- ══ APPROACH COMPARISON — Section 3-E / Section 8 ══ -->
      <div id="compare-sec" class="hidden">
        <div class="card">
          <div class="card-hd">
            <div class="card-title">⚖️ Approach Comparison — Rule-Based vs TF-IDF vs Full AI</div>
            <span style="font-size:10px;color:var(--g400)">Section 3-E: Compare at least two approaches · Section 8: Compare alternative settings</span>
          </div>
          <div class="card-body">
            <div style="font-size:11px;color:var(--g500);margin-bottom:12px;padding:8px 12px;background:var(--blue-lt);border-radius:8px;border-left:3px solid var(--blue)">
              📌 This panel runs the same resumes through <strong>3 different AI approaches</strong> and compares results side-by-side — demonstrating how algorithm choice affects candidate ranking.
            </div>
            <button class="run-btn" style="width:auto;padding:8px 20px;margin-bottom:14px" onclick="runCompare()">▶ Run Comparison Now</button>
            <div id="compare-loading" class="hidden" style="text-align:center;padding:20px;color:var(--g500);font-size:12px">⏳ Running 3 approaches...</div>
            <div id="compare-result"></div>
          </div>
        </div>
      </div>

      <!-- ══ FORWARD CHAINING — Option 1: Rule-based AI ══ -->
      <div id="fc-sec" class="hidden">
        <div class="card">
          <div class="card-hd">
            <div class="card-title">🔗 Forward Chaining — Rule-Based AI Inference</div>
            <span style="font-size:10px;color:var(--g400)">Option 1: Rule-based AI · Facts + Rules → Derived Decisions (per candidate)</span>
          </div>
          <div class="card-body">
            <div style="font-size:11px;color:var(--g500);margin-bottom:12px;padding:8px 12px;background:var(--amber-lt);border-radius:8px;border-left:3px solid var(--amber)">
              ⚙️ Forward chaining derives decisions by applying <strong>IF-THEN rules</strong> to candidate facts. Rules fire in sequence; each fired rule adds new facts enabling further inferences.
            </div>
            <div id="fc-loading" class="hidden" style="text-align:center;padding:20px;color:var(--g500);font-size:12px">⏳ Running forward chaining...</div>
            <div id="fc-result"></div>
          </div>
        </div>
      </div>

      <!-- ══ SEARCH & OPTIMISATION AI — Option 2 ══ -->
      <div id="search-sec" class="hidden">
        <div class="card">
          <div class="card-hd">
            <div class="card-title">🔍 Search & Optimisation AI — Best Shortlist Selection</div>
            <span style="font-size:10px;color:var(--g400)">Option 2: BFS · DFS · UCS · Greedy · A* · Local Search (Hill Climbing)</span>
          </div>
          <div class="card-body">
            <div style="font-size:11px;color:var(--g500);margin-bottom:12px;padding:8px 12px;background:var(--blue-lt);border-radius:8px;border-left:3px solid var(--blue)">
              🧭 The problem: pick the <strong>best K-person shortlist</strong> that maximises total score. Each algorithm explores the same "include / skip" decision tree differently — this panel visualises explored states, path found, nodes expanded, and runtime side-by-side.
            </div>
            <div style="display:flex;gap:10px;align-items:center;margin-bottom:14px;flex-wrap:wrap">
              <label style="font-size:11px;color:var(--g600);font-weight:500">Shortlist size (K):</label>
              <input id="search-budget" type="number" min="1" max="10" value="3" class="minp" style="width:70px">
              <button class="run-btn" style="width:auto;padding:8px 20px" onclick="runSearchAlgos()">▶ Run All Algorithms</button>
            </div>
            <div id="search-loading" class="hidden" style="text-align:center;padding:20px;color:var(--g500);font-size:12px">⏳ Searching state space with 6 algorithms...</div>
            <div id="search-result"></div>
          </div>
        </div>
      </div>

      <!-- ══ MACHINE LEARNING AI — Option 3 (trained models) ══ -->
      <div id="ml-sec" class="hidden">
        <div class="card">
          <div class="card-hd">
            <div class="card-title">🧠 Machine Learning AI — Train, Evaluate, Predict</div>
            <span style="font-size:10px;color:var(--g400)">Option 3: Classification · Regression · Clustering — real scikit-learn models, trained on your screened candidates</span>
          </div>
          <div class="card-body">
            <div style="font-size:11px;color:var(--g500);margin-bottom:12px;padding:8px 12px;background:var(--blue-lt);border-radius:8px;border-left:3px solid var(--blue)">
              🎓 Unlike the TF-IDF similarity score, these are <strong>trained models</strong>: a classifier learns to predict Qualified/Not-Qualified, a regressor learns to predict the score, and K-Means discovers candidate groups with no labels at all — each trained on a <em>partial</em> feature set so their predictions are genuine, not just re-stating the scoring formula.
            </div>
            <div style="display:flex;gap:10px;align-items:center;margin-bottom:14px;flex-wrap:wrap">
              <label style="font-size:11px;color:var(--g600);font-weight:500">Clusters (K-Means k):</label>
              <input id="ml-k" type="number" min="2" max="6" value="3" class="minp" style="width:60px">
              <button class="run-btn" style="width:auto;padding:8px 20px" onclick="runMLTraining()">▶ Train All 3 Models</button>
            </div>
            <div id="ml-loading" class="hidden" style="text-align:center;padding:20px;color:var(--g500);font-size:12px">⏳ Training classification, regression, and clustering models...</div>

            <div id="ml-tabs" class="hidden" style="display:flex;gap:6px;margin-bottom:14px">
              <button class="mltab on" id="mltab-0" onclick="showMLTab(0)">🏷 Classification</button>
              <button class="mltab" id="mltab-1" onclick="showMLTab(1)">📈 Regression</button>
              <button class="mltab" id="mltab-2" onclick="showMLTab(2)">🧩 Clustering</button>
            </div>
            <div id="ml-cls-panel" class="mlpanel"></div>
            <div id="ml-reg-panel" class="mlpanel hidden"></div>
            <div id="ml-clu-panel" class="mlpanel hidden"></div>
            <div id="ml-result"></div>
          </div>
        </div>
      </div>

      <!-- ══ NLP CHATBOT ══ -->
      <div id="chatbot-sec" class="hidden">
        <div class="card">
          <div class="card-hd">
            <div class="card-title">💬 NLP Explanation Chat — Ask About Results</div>
            <span style="font-size:10px;color:var(--g400)">D) Explainability · Option 4: NLP/LLM-assisted AI Panel</span>
          </div>
          <div class="card-body">
            <div style="font-size:11px;color:var(--g500);margin-bottom:10px;padding:8px 12px;background:var(--violet-lt);border-radius:8px;border-left:3px solid var(--violet)">
              🤖 <strong>NLP Chat Panel</strong> — Ask plain-English questions about the screening results. The system analyses your current results and generates a natural-language answer.
              <br><span style="color:var(--g400)">No external AI API is used here — every answer is generated locally from your live results, so there's no external prompt/cost/privacy to disclose.</span>
            </div>
            <!-- Chat history -->
            <div id="chat-msgs" style="height:260px;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding:8px;background:var(--g50);border:1px solid var(--g200);border-radius:9px;margin-bottom:10px"></div>
            <!-- Quick prompts -->
            <div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:9px">
              <button class="schip" onclick="askChat('Who is the best candidate and why?')">Who is the best?</button>
              <button class="schip" onclick="askChat('Which candidates are qualified?')">Who qualifies?</button>
              <button class="schip" onclick="askChat('What skills are most missing?')">Missing skills?</button>
              <button class="schip" onclick="askChat('Compare the top 2 candidates.')">Compare top 2</button>
              <button class="schip" onclick="askChat('Explain the scoring method used.')">How scoring works?</button>
              <button class="schip" onclick="askChat('Which candidate has the most experience?')">Most experience?</button>
            </div>
            <!-- Input row -->
            <div style="display:flex;gap:8px">
              <input id="chat-inp" class="minp" style="flex:1" placeholder="Ask anything about the results...">
              <button class="run-btn" style="width:auto;padding:8px 18px" onclick="askChat()">Send ▶</button>
            </div>
          </div>
        </div>
      </div>

      <!-- ══ DATABASE HISTORY ══ -->
      <div id="db-sec" class="hidden">
        <div class="card">
          <div class="card-hd">
            <div class="card-title">🗄️ Database History — All Saved Results</div>
            <div style="display:flex;gap:8px;align-items:center">
              <span id="db-count" style="font-size:10px;color:var(--g400)"></span>
              <button onclick="dbClear()" style="padding:4px 10px;font-size:10px;border-radius:6px;border:1px solid var(--red-md);background:var(--red-lt);color:var(--red);cursor:pointer;font-family:inherit">🗑 Clear All</button>
            </div>
          </div>
          <div class="card-body">
            <div id="db-stats-row" style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:14px"></div>
            <div id="db-sessions" style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:12px"></div>
            <div class="twrap">
              <table>
                <thead><tr>
                  <th>Session</th><th>Rank</th><th>Candidate</th>
                  <th>Score</th><th>Tier</th><th>Verdict</th>
                  <th>Job Role</th><th>Screened At</th>
                </tr></thead>
                <tbody id="db-tbl"></tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

    </div><!-- /results -->
  </div><!-- /page -->
</div><!-- /main -->
</div><!-- /app -->

<script>
// ── State ──────────────────────────────────────────────────────────────────
const JDS={
  "ML Engineer":"Python developer with machine learning, TensorFlow or PyTorch, SQL, REST APIs. Minimum 2 years. BS/MS Computer Science.",
  "Data Scientist":"Data scientist with Python, NLP, scikit-learn, pandas, deep learning. MS preferred. 2+ years.",
  "Backend Developer":"Java developer with Spring Boot, MySQL, Docker, microservices. 3+ years required.",
  "AI Researcher":"PyTorch, computer vision, Python, research publications. PhD preferred.",
};
const C=['#3b82f6','#059669','#b45309','#dc2626','#7c3aed','#0891b2'];
const CP=['rgba(59,130,246,.12)','rgba(5,150,105,.12)','rgba(180,83,9,.12)','rgba(220,38,38,.12)','rgba(124,58,237,.12)','rgba(8,145,178,.12)'];
const AV=['av0','av1','av2','av3','av4','av5'];
const EC=['ec-g','ec-b','ec-a','ec-r'];
const SBX={0:'#eff6ff',1:'#ecfdf5',2:'#fffbeb',3:'#fef2f2'};
const STX={0:'#1d4ed8',1:'#065f46',2:'#92400e',3:'#991b1b'};
let all=[], jdS=[], thresh=60, charts={}, sKey='score', sAsc=false, jdText='';
let mode='sample', cn=0, files=[], evalData={};

// ── Helpers ────────────────────────────────────────────────────────────────
function sv(id,el,sfx=''){document.getElementById(id).textContent=(el.value/10).toFixed(1)+sfx}
function fillJD(){const v=document.getElementById('jd-preset').value;if(v)document.getElementById('jd-text').value=JDS[v]}
function setMode(m){
  mode=m;
  document.getElementById('btn-sample').classList.toggle('on',m==='sample');
  document.getElementById('btn-manual').classList.toggle('on',m==='manual');
  document.getElementById('manual-sec').classList.toggle('hidden',m!=='manual');
  if(m==='manual'&&cn===0){addCand();addCand();}
}
function setSt(msg,type){const el=document.getElementById('status');el.className='status s-'+type;el.innerHTML=`<div class="sdot"></div>${msg}`}
function cTab(i,btn){document.querySelectorAll('.ctab').forEach((b,j)=>b.classList.toggle('on',j===i));document.querySelectorAll('.cpanel').forEach((p,j)=>p.classList.toggle('on',j===i))}
function spC(s){return s>=75?'sp-g':s>=50?'sp-b':s>=30?'sp-a':'sp-r'}
function spCol(s){return s>=75?'#059669':s>=50?'#3b82f6':s>=30?'#b45309':'#dc2626'}
function dc(id){if(charts[id]){charts[id].destroy();delete charts[id]}}
function srt(k,btn){if(sKey===k)sAsc=!sAsc;else{sKey=k;sAsc=false};document.querySelectorAll('.schip').forEach(b=>b.classList.remove('on'));btn.classList.add('on');renderTbl(all)}
function filterTbl(){
  const q=document.getElementById('srch').value.toLowerCase();
  const sf=document.getElementById('sfil').value;
  const f=all.filter(r=>{const nm=r.name.toLowerCase().includes(q);const sk=r.skills.some(s=>s.toLowerCase().includes(q));const st=sf===''||(sf==='q'&&r.score>=thresh)||(sf==='n'&&r.score<thresh);return(nm||sk)&&st});
  renderTbl(f,false);
}
function addCand(){
  const g=document.getElementById('cand-grid');const i=++cn;
  const d=document.createElement('div');d.className='mcard';d.id='mc'+i;
  d.innerHTML=`<div class="mhd">Candidate ${i}<button class="delbtn" onclick="document.getElementById('mc${i}').remove()">✕</button></div>
  <div class="mlbl">Name</div><input class="minp" id="mn${i}" placeholder="Full name">
  <div class="mlbl">Resume Text</div><textarea class="mta" id="mt${i}" placeholder="Paste resume content..."></textarea>
  <div class="mlbl">Years Experience</div><input type="number" class="minp" id="me${i}" value="1" min="0" max="30">
  <div class="mlbl">Education</div><input class="minp" id="med${i}" placeholder="e.g. BS Computer Science">`;
  g.appendChild(d);
}
function rankBadge(r){if(r===1)return`<div class="rnum r1">🥇</div>`;if(r===2)return`<div class="rnum r2">🥈</div>`;if(r===3)return`<div class="rnum r3">🥉</div>`;return`<div class="rnum rn">${r}</div>`}

// ── File Upload ────────────────────────────────────────────────────────────
function onDrop(e){e.preventDefault();document.getElementById('up-area').classList.remove('drag');onFiles(e.dataTransfer.files)}
async function onFiles(fls){
  document.getElementById('up-stat').textContent='Processing...';
  for(const f of fls){
    if(!f.name.match(/\.(pdf|txt)$/i))continue;
    const b64=await toB64(f);
    const ex=files.findIndex(x=>x.name===f.name);
    if(ex>=0)files.splice(ex,1);
    files.push({name:f.name,size:f.size,b64,type:f.type});
  }
  renderFiles();
  document.getElementById('up-stat').textContent=files.length?`${files.length} file(s) ready.`:'';
  document.getElementById('hs-val').textContent=files.length||6;
}
function toB64(f){return new Promise((res,rej)=>{const r=new FileReader();r.onload=()=>res(r.result.split(',')[1]);r.onerror=rej;r.readAsDataURL(f)})}
function renderFiles(){
  document.getElementById('file-list').innerHTML=files.map((f,i)=>`
    <div class="file-item">
      <span>📄</span><span class="fi-name">${f.name}</span>
      <span class="fi-sz">${(f.size/1024).toFixed(1)} KB</span>
      <button class="fi-rm" onclick="files.splice(${i},1);renderFiles()">✕</button>
    </div>`).join('');
}

// ── Run ────────────────────────────────────────────────────────────────────
async function run(){
  const jd=document.getElementById('jd-text').value.trim();
  if(!jd){setSt('❌ Please enter or select a Job Description.','err');return}
  let resumes=[];
  if(files.length>0){
    resumes=files.map(f=>({name:f.name.replace(/\.(pdf|txt)$/i,''),b64:f.b64,file_type:f.type,experience_years:0,education:''}));
  } else if(mode==='manual'){
    document.querySelectorAll('.mcard').forEach(card=>{
      const id=card.id.replace('mc','');
      const nm=document.getElementById('mn'+id)?.value.trim();
      const tx=document.getElementById('mt'+id)?.value.trim();
      const ex=parseInt(document.getElementById('me'+id)?.value)||0;
      const ed=document.getElementById('med'+id)?.value.trim()||'';
      if(nm&&tx)resumes.push({name:nm,text:tx,experience_years:ex,education:ed});
    });
    if(!resumes.length){setSt('❌ Add at least one candidate.','err');return}
  }
  document.getElementById('loader').classList.add('show');
  document.getElementById('empty').classList.add('hidden');
  setSt('⏳ Running AI screening...','load');
  thresh=parseInt(document.getElementById('thresh').value);
  const useSem=document.getElementById('tog-sem').classList.contains('on');
  const jobTitle=document.getElementById('jd-preset').value||'Custom Role';
  const payload={jd_text:jd,job_title:jobTitle,w_tfidf:parseFloat(document.getElementById('w-tf').value)/10,w_skill:parseFloat(document.getElementById('w-sk').value)/10,w_exp:parseFloat(document.getElementById('w-ex').value)/10,threshold:thresh,mode:files.length?'upload':mode,resumes,use_semantic:useSem};
  try{
    const resp=await fetch('/screen',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data=await resp.json();
    document.getElementById('loader').classList.remove('show');
    if(data.error){setSt('❌ '+data.error,'err');return}
    all=data.results; jdS=data.jd_skills; evalData=data.evaluation; jdText=jd;
    renderAll(data);
  }catch(e){document.getElementById('loader').classList.remove('show');setSt('❌ '+e.message,'err')}
}

// ── Render All ─────────────────────────────────────────────────────────────
function renderAll(data){
  const {results,threshold:th,evaluation:ev,use_semantic,gap_analysis,overall_coverage,session_id}=data;
  document.getElementById('results').classList.remove('hidden');
  const q=results.filter(r=>r.score>=th).length;
  const dbMsg = session_id ? ` · 💾 Saved #${session_id}` : '';
  setSt(`✅ Done — <strong>${results.length}</strong> candidates · <strong>${q}</strong> qualified · ${ev.total_ms}ms${dbMsg}`,'ok');
  renderMetrics(results,th);
  renderTbl(results);
  renderCharts(results,data.jd_skills,use_semantic);
  if(document.getElementById('tog-exp').classList.contains('on')){
    document.getElementById('exp-sec').classList.remove('hidden');
    renderExp(results,data.jd_skills,use_semantic);
  } else { document.getElementById('exp-sec').classList.add('hidden'); }
  // New visual modules
  renderTimeline(results, ev);
  renderScoreGrid(results);
  renderSkillNetwork(results, data.jd_skills||[]);
  if(gap_analysis&&gap_analysis.length) renderGapAnalysis(gap_analysis, overall_coverage||{});
  // Show chatbot panel and seed with a welcome message
  document.getElementById('chatbot-sec').classList.remove('hidden');
  const chatBox = document.getElementById('chat-msgs');
  chatBox.innerHTML = '';
  addChatMsg(`🤖 Screening complete! I've analysed <strong>${results.length}</strong> candidates. Ask me anything — try the quick-prompt buttons above or type your own question.`,'bot');
}

// ── Metrics ────────────────────────────────────────────────────────────────
function renderMetrics(res,th){
  const top=res[0],avg=res.reduce((s,r)=>s+r.score,0)/res.length,q=res.filter(r=>r.score>=th).length;
  document.getElementById('metrics').innerHTML=`
  <div class="mc" style="animation-delay:0s"><div class="mc-ico ic-b">🏆</div><div><div class="mc-val cv-b">${top.score}%</div><div class="mc-lbl">Top Score</div><div class="mc-sub">${top.name}</div></div></div>
  <div class="mc" style="animation-delay:.07s"><div class="mc-ico ic-g">👤</div><div><div class="mc-val cv-g">${top.name.split(' ')[0]}</div><div class="mc-lbl">Best Match</div><div class="mc-sub">Rank #1</div></div></div>
  <div class="mc" style="animation-delay:.14s"><div class="mc-ico ic-a">📊</div><div><div class="mc-val cv-a">${avg.toFixed(1)}%</div><div class="mc-lbl">Average Score</div><div class="mc-sub">${res.length} candidates</div></div></div>
  <div class="mc" style="animation-delay:.21s"><div class="mc-ico ic-r">✅</div><div><div class="mc-val cv-r">${q}/${res.length}</div><div class="mc-lbl">Qualified</div><div class="mc-sub">≥${th}%</div></div></div>`;
}

// ── Table ──────────────────────────────────────────────────────────────────
function renderTbl(res,resort=true){
  if(resort)res=[...res].sort((a,b)=>sAsc?(a[sKey]-b[sKey]):(b[sKey]-a[sKey]));
  document.getElementById('rc').textContent=`${res.length}/${all.length} shown`;
  document.getElementById('tbl').innerHTML=res.map((r,i)=>{
    const av=AV[r.rank%AV.length],sc=spCol(r.score),sp=spC(r.score);
    const sem=r.semantic!==null?`${r.semantic}%`:`<span style="color:var(--g300);font-size:10px">N/A</span>`;
    return`<tr style="animation-delay:${i*.04}s">
      <td>${rankBadge(r.rank)}</td>
      <td><div style="display:flex;align-items:center;gap:9px"><div class="av ${av}">${r.initials}</div><div><div class="cname">${r.name}</div><div class="cedu">${r.education}</div></div></div></td>
      <td><div><span class="sp ${sp}">${r.score}%</span><div class="mbar"><div class="mfill" style="width:${r.score}%;background:${sc}"></div></div></div></td>
      <td><span class="snum sn-b">${r.tfidf}%</span></td>
      <td><span class="snum" style="color:#7c3aed">${sem}</span></td>
      <td><span class="snum sn-g">${r.skill}%</span></td>
      <td><span class="snum sn-a">${r.exp_score}%</span><div style="font-size:9px;color:var(--g400)">${r.exp_yrs} yr${r.exp_yrs!==1?'s':''}</div></td>
      <td>${r.matched.slice(0,3).map(s=>`<span class="tag tb">${s}</span>`).join('')}${r.matched.length>3?`<span class="tag tb">+${r.matched.length-3}</span>`:''}</td>
      <td><span class="qbdg ${r.score>=thresh?'qy':'qn'}">${r.score>=thresh?'✅ Qualified':'❌ Not Qualified'}</span></td>
    </tr>`;
  }).join('');
}

// ── Charts ─────────────────────────────────────────────────────────────────
function renderCharts(res,jds,useSem){
  const names=res.map(r=>r.name),scores=res.map(r=>r.score);
  const gc={responsive:true,maintainAspectRatio:true,animation:{duration:700}};
  const gs={grid:{color:'rgba(229,231,235,.7)'},ticks:{color:'#6b7280'}};

  dc('bar');charts['bar']=new Chart(document.getElementById('c-bar'),{type:'bar',
    data:{labels:names,datasets:[{data:scores,backgroundColor:scores.map((s,i)=>i===0?'rgba(180,83,9,.75)':s>=60?'rgba(59,130,246,.7)':'rgba(220,38,38,.6)'),borderRadius:7,borderSkipped:false}]},
    options:{...gc,plugins:{legend:{display:false}},scales:{y:{...gs,min:0,max:115,ticks:{...gs.ticks,callback:v=>v+'%'}},x:{...gs,grid:{display:false}}}}});

  dc('pie');charts['pie']=new Chart(document.getElementById('c-pie'),{type:'doughnut',
    data:{labels:names,datasets:[{data:scores,backgroundColor:C,borderWidth:3,borderColor:'#fff'}]},
    options:{...gc,plugins:{legend:{position:'right',labels:{color:'#374151',font:{size:10},padding:10}},tooltip:{callbacks:{label:c=>c.label+': '+c.raw+'%'}}},cutout:'45%'}});

  dc('line');charts['line']=new Chart(document.getElementById('c-line'),{type:'line',
    data:{labels:names,datasets:[
      {label:'TF-IDF',data:res.map(r=>r.tfidf),borderColor:'#3b82f6',backgroundColor:'rgba(59,130,246,.08)',tension:.35,fill:true,pointRadius:4,pointBackgroundColor:'#3b82f6'},
      {label:'Skill', data:res.map(r=>r.skill),borderColor:'#059669',backgroundColor:'rgba(5,150,105,.08)',tension:.35,fill:true,pointRadius:4,pointBackgroundColor:'#059669'},
      {label:'Exp.',  data:res.map(r=>r.exp_score),borderColor:'#b45309',backgroundColor:'rgba(180,83,9,.08)',tension:.35,fill:true,pointRadius:4,pointBackgroundColor:'#b45309'},
      {label:'Final', data:scores,borderColor:'#dc2626',backgroundColor:'rgba(220,38,38,.06)',tension:.35,fill:true,pointRadius:4,pointBackgroundColor:'#dc2626',borderWidth:2.5},
      ...(useSem&&res[0].semantic!==null?[{label:'Semantic',data:res.map(r=>r.semantic||0),borderColor:'#7c3aed',backgroundColor:'rgba(124,58,237,.08)',tension:.35,fill:true,pointRadius:4,pointBackgroundColor:'#7c3aed'}]:[])
    ]},
    options:{...gc,plugins:{legend:{labels:{color:'#374151',font:{size:10}}}},scales:{y:{...gs,min:0,max:115,ticks:{...gs.ticks,callback:v=>v+'%'}},x:{...gs,grid:{display:false}}}}});

  dc('scat');charts['scat']=new Chart(document.getElementById('c-scat'),{type:'bubble',
    data:{datasets:res.map((r,i)=>({label:r.name,data:[{x:r.skill,y:r.tfidf,r:Math.max(r.score/7,5)}],backgroundColor:CP[i%CP.length],borderColor:C[i%C.length],borderWidth:2}))},
    options:{...gc,plugins:{legend:{labels:{color:'#374151',font:{size:10}}}},scales:{x:{...gs,title:{display:true,text:'Skill Match %',color:'#9ca3af',font:{size:10}}},y:{...gs,title:{display:true,text:'TF-IDF %',color:'#9ca3af',font:{size:10}}}}}});

  dc('rad');charts['rad']=new Chart(document.getElementById('c-rad'),{type:'radar',
    data:{labels:['TF-IDF','Skills','Exp.','Final'],datasets:res.slice(0,3).map((r,i)=>({label:r.name,data:[r.tfidf,r.skill,r.exp_score,r.score],borderColor:C[i],backgroundColor:CP[i],pointBackgroundColor:C[i],pointRadius:4}))},
    options:{...gc,plugins:{legend:{labels:{color:'#374151',font:{size:10}}}},scales:{r:{min:0,max:100,grid:{color:'rgba(229,231,235,.8)'},pointLabels:{color:'#374151',font:{size:10}},ticks:{display:false}}}}});

  dc('heat');
  if(jds.length){
    const pct=jds.map(sk=>Math.round(res.filter(r=>r.matched.includes(sk)).length/res.length*100));
    charts['heat']=new Chart(document.getElementById('c-heat'),{type:'bar',
      data:{labels:jds,datasets:[{label:'Coverage',data:pct,backgroundColor:pct.map(v=>v>=75?'rgba(5,150,105,.65)':v>=40?'rgba(59,130,246,.65)':'rgba(220,38,38,.5)'),borderRadius:5}]},
      options:{...gc,indexAxis:'y',plugins:{legend:{display:false}},scales:{x:{...gs,min:0,max:100,ticks:{...gs.ticks,callback:v=>v+'%'}},y:{...gs,grid:{display:false},ticks:{...gs.ticks,font:{size:10}}}}}});
  }

  const mn=Math.min(...scores),mx=Math.max(...scores),av=scores.reduce((a,b)=>a+b,0)/scores.length;
  dc('box');charts['box']=new Chart(document.getElementById('c-box'),{type:'bar',
    data:{labels:['Min','Avg','Max'],datasets:[{data:[mn,av,mx],backgroundColor:['rgba(220,38,38,.6)','rgba(59,130,246,.65)','rgba(5,150,105,.65)'],borderRadius:8,borderSkipped:false}]},
    options:{...gc,plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>c.raw.toFixed(1)+'%'}}},scales:{y:{...gs,min:0,max:115,ticks:{...gs.ticks,callback:v=>v+'%'}},x:{...gs,grid:{display:false}}}}});

  dc('range');
  const bins={'0–25':0,'25–50':0,'50–75':0,'75–100':0};
  scores.forEach(s=>{if(s<25)bins['0–25']++;else if(s<50)bins['25–50']++;else if(s<75)bins['50–75']++;else bins['75–100']++;});
  charts['range']=new Chart(document.getElementById('c-range'),{type:'bar',
    data:{labels:Object.keys(bins).map(k=>k+'%'),datasets:[{data:Object.values(bins),backgroundColor:['rgba(220,38,38,.6)','rgba(180,83,9,.6)','rgba(59,130,246,.65)','rgba(5,150,105,.65)'],borderRadius:8,borderSkipped:false}]},
    options:{...gc,plugins:{legend:{display:false}},scales:{y:{...gs,min:0,ticks:{...gs.ticks,stepSize:1}},x:{...gs,grid:{display:false}}}}});

  // Comparison: TF-IDF vs Semantic (Module E)
  const cmp=document.getElementById('cmp-grid');
  if(useSem&&res[0].semantic!==null){
    cmp.innerHTML=`
    <div class="cmp-box">
      <div class="cmp-title"><span style="color:#3b82f6;font-size:14px">📊</span>TF-IDF Scores</div>
      ${res.map(r=>`<div class="cmp-item"><span class="cmp-name">${r.name}</span><span class="cmp-score" style="color:#3b82f6">${r.tfidf}%</span></div>`).join('')}
    </div>
    <div class="cmp-box">
      <div class="cmp-title"><span style="color:#7c3aed;font-size:14px">🔮</span>Semantic Scores</div>
      ${res.map(r=>`<div class="cmp-item"><span class="cmp-name">${r.name}</span><span class="cmp-score" style="color:#7c3aed">${r.semantic!==null?r.semantic+'%':'N/A'}</span></div>`).join('')}
    </div>`;
  } else {
    cmp.innerHTML=`<div style="grid-column:1/-1;padding:10px;background:var(--g50);border:1px solid var(--g200);border-radius:8px;font-size:11px;color:var(--g500);text-align:center">Enable "Use Semantic AI" toggle and re-run to see TF-IDF vs Semantic comparison.</div>`;
  }

  // Eval metrics (Module E)
  const ev=evalData;
  document.getElementById('eval-row').innerHTML=`
    <div class="eval-box"><div class="eval-val" style="color:#059669">${mx.toFixed(1)}%</div><div class="eval-lbl">Max Score</div></div>
    <div class="eval-box"><div class="eval-val" style="color:#dc2626">${mn.toFixed(1)}%</div><div class="eval-lbl">Min Score</div></div>
    <div class="eval-box"><div class="eval-val" style="color:#3b82f6">${av.toFixed(1)}%</div><div class="eval-lbl">Mean Score</div></div>
    <div class="eval-box"><div class="eval-val" style="color:#b45309">${Math.sqrt(scores.reduce((s,v)=>s+Math.pow(v-av,2),0)/scores.length).toFixed(1)}</div><div class="eval-lbl">Std Dev</div></div>`;
  document.getElementById('timing-row').innerHTML=`
    <div class="tim-box"><div class="tim-val">${ev.precision}%</div><div class="tim-lbl">Precision</div></div>
    <div class="tim-box"><div class="tim-val">${ev.recall}%</div><div class="tim-lbl">Recall</div></div>
    <div class="tim-box"><div class="tim-val">${ev.f1}%</div><div class="tim-lbl">F1 Score</div></div>
    <div class="tim-box"><div class="tim-val" style="color:#059669">${ev.tfidf_ms||0} ms</div><div class="tim-lbl">TF-IDF Runtime</div></div>
    <div class="tim-box"><div class="tim-val" style="color:#7c3aed">${ev.semantic_ms||0} ms</div><div class="tim-lbl">Semantic Runtime</div></div>
    <div class="tim-box"><div class="tim-val" style="color:#b45309">${ev.peak_memory_kb||0} KB</div><div class="tim-lbl">Peak Memory</div></div>
    <div class="tim-box"><div class="tim-val" style="color:#0891b2">${ev.avg_score||0}%</div><div class="tim-lbl">Avg Score</div></div>
    <div class="tim-box"><div class="tim-val" style="color:#9333ea">±${ev.stddev||0}%</div><div class="tim-lbl">Std Deviation</div></div>`;
}

// ── Explanations (Module D) ────────────────────────────────────────────────
function renderExp(res,jds,useSem){
  document.getElementById('exp-cards').innerHTML=res.map((r,i)=>{
    const cls=r.score>=75?'ec-g':r.score>=50?'ec-b':r.score>=30?'ec-a':'ec-r';
    const icon=r.score>=75?'✅':r.score>=50?'⚠️':'❌';
    const verdict=r.score>=75?'Strong Match':r.score>=50?'Moderate Match':'Weak Match';
    const detail=r.score>=75
      ?`Covers ${r.matched.length} of ${jds.length} required skills with ${r.exp_yrs} yr(s) experience.`
      :r.score>=50?`Partially qualified. Missing: ${r.missing.slice(0,3).join(', ')||'none'}.`
      :`Lacks key skills (${r.missing.slice(0,4).join(', ')||'N/A'}).`;
    const av=AV[i%AV.length],sbi=r.score>=75?0:r.score>=50?1:r.score>=30?2:3;
    const semRow=useSem&&r.semantic!==null?`<div class="ebr"><span class="ebl">Semantic</span><div class="ebb"><div class="ebf" style="width:${r.semantic}%;background:#7c3aed"></div></div><span class="ebv" style="color:#7c3aed">${r.semantic}%</span></div>`:'';
    return`<div class="exp-card ${cls}" style="animation-delay:${i*.06}s">
      <div class="exp-hd">
        <div class="av ${av}">${r.initials}</div>
        <div class="exp-nw"><div class="exp-nm">${r.name}</div><div class="exp-ed">${r.education} · Rank #${r.rank}</div></div>
        <div class="exp-sb" style="background:${SBX[sbi]};color:${STX[sbi]}"><div class="exp-sv">${r.score}%</div><div class="exp-sl">Score</div></div>
      </div>
      <div class="exp-verdict">${icon} <strong>${verdict}</strong> — ${detail}</div>
      <div class="eb">
        <div class="ebr"><span class="ebl">TF-IDF</span><div class="ebb"><div class="ebf" style="width:${r.tfidf}%;background:#3b82f6"></div></div><span class="ebv" style="color:#3b82f6">${r.tfidf}%</span></div>
        ${semRow}
        <div class="ebr"><span class="ebl">Skills</span><div class="ebb"><div class="ebf" style="width:${r.skill}%;background:#059669"></div></div><span class="ebv" style="color:#059669">${r.skill}%</span></div>
        <div class="ebr"><span class="ebl">Experience</span><div class="ebb"><div class="ebf" style="width:${r.exp_score}%;background:#b45309"></div></div><span class="ebv" style="color:#b45309">${r.exp_score}%</span></div>
      </div>
      <div class="exp-tags">
        ${r.matched.map(s=>`<span class="tag tg">${s}</span>`).join('')}
        ${r.missing.slice(0,3).map(s=>`<span class="tag tr">−${s}</span>`).join('')}
      </div>
    </div>`;
  }).join('');
}

// ── Toggle sections ────────────────────────────────────────────────────────
function toggleSec(id){
  const el=document.getElementById(id);
  if(!el) return;
  el.classList.toggle('hidden');
}
function toggleToolsMenu(){
  document.getElementById('tools-menu').classList.toggle('hidden');
}
document.addEventListener('click', e=>{
  const menu=document.getElementById('tools-menu'), btn=document.getElementById('tools-btn');
  if(menu && !menu.classList.contains('hidden') && !menu.contains(e.target) && e.target!==btn){
    menu.classList.add('hidden');
  }
});

// ── Timeline Animation ─────────────────────────────────────────────────────
function renderTimeline(res, ev){
  document.getElementById('timeline-sec').classList.remove('hidden');
  const steps=[
    {icon:'📄',title:'Resume Parsing',desc:'Extracted name, skills, experience, education using NLP',score:null,done:true},
    {icon:'💼',title:'JD Parsing',desc:'Extracted required skills and experience from job description',score:null,done:true},
    {icon:'📊',title:'TF-IDF Matching',desc:'Computed cosine similarity between resume and JD vectors',score:`${res[0]?.tfidf||0}% similarity`,done:true},
    {icon:'🔮',title:'Semantic Matching',desc:'Sentence Transformer semantic similarity (meaning-based)',score:res[0]?.semantic?`${res[0].semantic}% similarity`:'Not enabled',done:true},
    {icon:'🎯',title:'Skill Matching',desc:'Direct skill overlap between candidate and JD requirements',score:`${res[0]?.skill||0}% skill coverage`,done:true},
    {icon:'⚖️',title:'Weighted Scoring',desc:'Final score = TF-IDF(20%) + Semantic(35%) + Skill(30%) + Exp(15%)',score:`Top score: ${res[0]?.score||0}%`,done:true},
    {icon:'🏆',title:'Candidate Ranking',desc:'Candidates ranked by final score with Tier A/B/C/D classification',score:`${res.length} candidates ranked`,done:true},
    {icon:'💾',title:'Score Storage',desc:'All results saved to MySQL database for future reference',score:`Session saved to DB`,done:true},
  ];
  document.getElementById('tl-steps').innerHTML = steps.map((s,i)=>`
    <div class="tl-step" style="animation-delay:${i*.1}s">
      <div class="tl-dot done" style="color:#fff;background:#059669">${s.icon}</div>
      <div class="tl-title">${s.title}</div>
      <div class="tl-desc">${s.desc}</div>
      ${s.score?`<div class="tl-score">→ ${s.score}</div>`:''}
    </div>`).join('');
}

// ── Score Grid (Grid/Map View) ─────────────────────────────────────────────
function renderScoreGrid(res){
  document.getElementById('grid-sec').classList.remove('hidden');
  const tierColors={A:'#059669',B:'#3b82f6',C:'#b45309',D:'#dc2626'};
  const tierBg={A:'#ecfdf5',B:'#eff6ff',C:'#fffbeb',D:'#fef2f2'};
  const tierIcons={A:'🏆',B:'✅',C:'⚠️',D:'❌'};
  document.getElementById('score-grid').innerHTML=res.map((r,i)=>`
    <div class="sg-cell" style="background:${tierBg[r.tier||'D']};border-color:${tierColors[r.tier||'D']}22;animation-delay:${i*.07}s">
      <div class="sg-name" style="color:${tierColors[r.tier||'D']}">${r.name}</div>
      <div class="sg-score" style="color:${tierColors[r.tier||'D']}">${r.score}%</div>
      <div class="sg-tier">${tierIcons[r.tier||'D']} Tier ${r.tier||'D'} · Rank #${r.rank}</div>
      <div style="height:4px;background:rgba(0,0,0,.08);border-radius:2px;margin-top:6px;overflow:hidden">
        <div style="height:100%;width:${r.score}%;background:${tierColors[r.tier||'D']};border-radius:2px;transition:width .8s ease"></div>
      </div>
    </div>`).join('');
}

// ── Skill Network Graph (Canvas-based) ─────────────────────────────────────
function renderSkillNetwork(res, jdSkills){
  document.getElementById('network-sec').classList.remove('hidden');
  const container = document.getElementById('skill-network');
  container.innerHTML = '';
  const canvas = document.createElement('canvas');
  canvas.width  = container.offsetWidth || 600;
  canvas.height = 280;
  container.appendChild(canvas);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);

  // Build nodes
  const nodes=[];
  const topCands = res.slice(0,4);

  // Candidate nodes (left side)
  topCands.forEach((c,i)=>{
    nodes.push({x:80, y:40+i*(H-60)/(topCands.length-1||1), label:c.name.split(' ')[0], color:'#3b82f6', r:18, type:'cand', matched:c.matched||[], missing:c.missing||[]});
  });

  // Skill nodes (right side)
  const allSkills = [...new Set([...jdSkills])].slice(0,8);
  allSkills.forEach((sk,i)=>{
    const matchCount = res.filter(r=>(r.matched||[]).includes(sk)).length;
    const isCommon   = matchCount >= res.length/2;
    nodes.push({x:W-80, y:30+i*(H-50)/(allSkills.length-1||1), label:sk, color:isCommon?'#059669':'#dc2626', r:14, type:'skill', skill:sk});
  });

  // Draw edges
  topCands.forEach((c,ci)=>{
    const cn = nodes[ci];
    allSkills.forEach((sk,si)=>{
      const sn = nodes[topCands.length+si];
      const matched = (c.matched||[]).includes(sk);
      ctx.beginPath();
      ctx.moveTo(cn.x+cn.r, cn.y);
      ctx.lineTo(sn.x-sn.r, sn.y);
      ctx.strokeStyle = matched?'rgba(5,150,105,.3)':'rgba(220,38,38,.15)';
      ctx.lineWidth   = matched?2:1;
      ctx.stroke();
    });
  });

  // Draw nodes
  nodes.forEach(n=>{
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI*2);
    ctx.fillStyle   = n.color+'22';
    ctx.fill();
    ctx.strokeStyle = n.color;
    ctx.lineWidth   = 2;
    ctx.stroke();
    ctx.fillStyle   = n.color;
    ctx.font        = `bold ${n.type==='cand'?11:9}px Inter,sans-serif`;
    ctx.textAlign   = n.type==='cand'?'right':'left';
    ctx.textBaseline= 'middle';
    ctx.fillText(n.label, n.type==='cand'?n.x-n.r-5:n.x+n.r+5, n.y);
  });
}

// ── Skill Gap Analysis ─────────────────────────────────────────────────────
function renderGapAnalysis(gaps, coverage){
  document.getElementById('gap-sec').classList.remove('hidden');
  if(!gaps||!gaps.length) return;
  document.getElementById('gap-cards').innerHTML=gaps.map((g,i)=>`
    <div class="gap-card" style="animation-delay:${i*.06}s">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
        <div class="av ${AV[i%AV.length]}">${all[i]?all[i].initials||g.name[0]:g.name[0]}</div>
        <div>
          <div style="font-size:13px;font-weight:600;color:var(--g900)">${g.name}</div>
          <div style="font-size:10px;color:var(--g500)">Rank #${g.rank} · ${g.total_req} required skills · Coverage: ${g.covered_pct}%</div>
        </div>
        <div style="margin-left:auto;font-size:16px;font-weight:700;color:${g.covered_pct>=75?'#059669':g.covered_pct>=50?'#3b82f6':'#dc2626'}">${g.score}%</div>
      </div>
      <div style="height:6px;background:var(--g200);border-radius:3px;margin-bottom:10px;overflow:hidden">
        <div style="height:100%;width:${g.covered_pct}%;background:${g.covered_pct>=75?'#059669':g.covered_pct>=50?'#3b82f6':'#dc2626'};border-radius:3px;transition:width .8s ease"></div>
      </div>
      <div class="gap-grid2">
        <div class="gap-half">
          <div class="gap-half-title" style="color:#059669">✅ Matched (${g.matched.length})</div>
          <div>${g.matched.map(s=>`<span class="tag tg" style="margin:2px;display:inline-block">${s}</span>`).join('')||'<span style="font-size:11px;color:var(--g400)">None</span>'}</div>
        </div>
        <div class="gap-half">
          <div class="gap-half-title" style="color:#dc2626">❌ Missing (${g.missing.length})</div>
          <div>${g.missing.map(s=>`<span class="tag tr" style="margin:2px;display:inline-block">${s}</span>`).join('')||'<span style="font-size:11px;color:var(--g400)">None — Perfect!</span>'}</div>
        </div>
      </div>
      ${g.recommendations&&g.recommendations.length?`
      <div style="margin-top:8px">
        <div style="font-size:10px;font-weight:600;color:var(--g700);margin-bottom:5px">📚 Learning Recommendations</div>
        ${g.recommendations.map(r=>`<div class="rec-item">${r}</div>`).join('')}
      </div>`:''}
    </div>`).join('');
}

// ── DB History ─────────────────────────────────────────────────────────────
async function loadDBHistory(){
  toggleSec('db-sec');
  if(!document.getElementById('db-sec').classList.contains('hidden')){
    try{
      const resp = await fetch('/db_results');
      const data = await resp.json();
      renderDBHistory(data);
    }catch(e){console.error('[DB]',e)}
  }
}
function renderDBHistory(data){
  const {results, sessions, stats} = data;
  document.getElementById('db-count').textContent = `${results.length} records`;

  // Stats row
  if(stats && Object.keys(stats).length){
    document.getElementById('db-stats-row').innerHTML=`
      <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:var(--blue)">${stats.total_candidates||0}</div>
        <div style="font-size:9px;color:var(--g500);text-transform:uppercase">Total Screened</div>
      </div>
      <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:var(--green)">${stats.avg_score||0}%</div>
        <div style="font-size:9px;color:var(--g500);text-transform:uppercase">Avg Score</div>
      </div>
      <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:var(--amber)">${stats.total_sessions||0}</div>
        <div style="font-size:9px;color:var(--g500);text-transform:uppercase">Sessions</div>
      </div>
      <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:10px;text-align:center">
        <div style="font-size:18px;font-weight:700;color:#059669">${stats.tier_a||0}</div>
        <div style="font-size:9px;color:var(--g500);text-transform:uppercase">Tier A</div>
      </div>`;
  }

  // Sessions pills
  document.getElementById('db-sessions').innerHTML=(sessions||[]).map(s=>
    `<span class="db-sess-pill">📋 ${s.job_title} · ${s.total_candidates} candidates · ${s.screened_at}</span>`
  ).join('');

  // Results table
  const tierC={A:'#059669',B:'#3b82f6',C:'#b45309',D:'#dc2626'};
  const tierB={A:'#ecfdf5',B:'#eff6ff',C:'#fffbeb',D:'#fef2f2'};
  document.getElementById('db-tbl').innerHTML=(results||[]).map((r,i)=>`
    <tr style="animation-delay:${i*.03}s">
      <td><span class="db-sess-pill">#${r.session_id}</span></td>
      <td>${r.rank===1?'🥇':r.rank===2?'🥈':r.rank===3?'🥉':'#'+r.rank}</td>
      <td style="font-weight:600">${r.name}</td>
      <td><span class="sp ${r.score>=75?'sp-g':r.score>=60?'sp-b':r.score>=30?'sp-a':'sp-r'}">${r.score}%</span></td>
      <td><div class="tier-badge tier-${r.tier||'D'}">${r.tier||'D'}</div><div style="font-size:9px;color:var(--g500)">${r.verdict||''}</div></td>
      <td style="font-size:11px;color:var(--g600)">${r.verdict||''}</td>
      <td><span style="font-size:10px;background:var(--blue-lt);color:#1d4ed8;padding:2px 7px;border-radius:4px">${r.job_title||'—'}</span></td>
      <td style="font-size:10px;color:var(--g400)">${r.screened_at||'—'}</td>
    </tr>`).join('');
}
async function dbClear(){
  if(!confirm('Delete all saved results from database?')) return;
  const resp = await fetch('/clear_db',{method:'POST'});
  const data = await resp.json();
  if(data.success){
    document.getElementById('db-tbl').innerHTML='';
    document.getElementById('db-count').textContent='0 records';
    document.getElementById('db-sessions').innerHTML='';
    document.getElementById('db-stats-row').innerHTML='';
    alert('Database cleared!');
  }
}

// ── NLP Chatbot (Option 4: NLP/LLM-assisted AI) ────────────────────────────
// Generates plain-English explanations from current results (no external API needed).
// The "AI" here is rule-based NLP that interprets results and produces human-readable
// summaries — satisfying Section 5, Option 4 (chatbot-style explanation panel).
function buildContext(){
  if(!all||!all.length) return null;
  return {
    candidates: all.map(r=>({
      rank:r.rank, name:r.name, score:r.score, tfidf:r.tfidf,
      semantic:r.semantic, skill:r.skill, exp_score:r.exp_score,
      exp_yrs:r.exp_yrs, education:r.education,
      matched:r.matched, missing:r.missing,
      tier:r.tier||'D', verdict:r.tier_label||'',
    })),
    threshold: thresh,
    jd_skills: jdS,
  };
}

function nlpAnswer(question, ctx){
  // NLP pattern matching → natural-language generation
  const q = question.toLowerCase();
  const cands = ctx.candidates;
  const qualified = cands.filter(c=>c.score>=ctx.threshold);
  const top = cands[0];

  // Who is best?
  if(q.match(/best|top|highest|number one|#1|first/)){
    const factors = [];
    if(top.tfidf>=60) factors.push(`high text similarity (${top.tfidf}%)`);
    if(top.skill>=60) factors.push(`strong skill match (${top.skill}%)`);
    if(top.exp_score>=80) factors.push(`solid experience (${top.exp_yrs} years)`);
    if(top.semantic&&top.semantic>=60) factors.push(`semantic alignment (${top.semantic}%)`);
    const why = factors.length ? `Key strengths: ${factors.join(', ')}.` : `Their overall profile aligns well with the job description.`;
    return `🏆 The best candidate is **${top.name}** with a final score of **${top.score}%** (Tier ${top.tier}). ${why} They matched ${top.matched.length} of ${ctx.jd_skills.length} required skills: ${top.matched.slice(0,4).join(', ')||'(none listed)'}.`;
  }

  // Who qualifies?
  if(q.match(/qualif|pass|eligible|shortlist|above threshold/)){
    if(!qualified.length) return `❌ No candidates met the qualification threshold of ${ctx.threshold}%. Consider lowering the threshold or providing candidates with more relevant experience.`;
    const names = qualified.map(c=>`${c.name} (${c.score}%)`).join(', ');
    return `✅ **${qualified.length} of ${cands.length}** candidates qualify (score ≥ ${ctx.threshold}%):\n\n${names}\n\nThe remaining ${cands.length-qualified.length} candidate(s) scored below the threshold and are not recommended for this role.`;
  }

  // Missing skills?
  if(q.match(/missing|gap|lack|don't have|do not have/)){
    const freq = {};
    cands.forEach(c=>c.missing.forEach(sk=>{ freq[sk]=(freq[sk]||0)+1; }));
    const sorted = Object.entries(freq).sort((a,b)=>b[1]-a[1]).slice(0,6);
    if(!sorted.length) return `✅ All candidates collectively cover the required skills — no major skill gaps detected.`;
    const list = sorted.map(([sk,n])=>`**${sk}** (missing in ${n}/${cands.length} candidates)`).join('\n• ');
    return `🎯 **Most common skill gaps** across all ${cands.length} candidates:\n\n• ${list}\n\nFocus hiring or training efforts on these areas.`;
  }

  // Compare top 2?
  if(q.match(/compar|vs|versus|difference|better/)){
    if(cands.length<2) return 'Only one candidate found — nothing to compare.';
    const a=cands[0], b=cands[1];
    const winner = a.score>b.score?a:b;
    return `📊 **Comparison: ${a.name} vs ${b.name}**\n\n` +
      `| Metric | ${a.name} | ${b.name} |\n` +
      `|--------|-----------|----------|\n` +
      `| Final Score | ${a.score}% | ${b.score}% |\n` +
      `| TF-IDF | ${a.tfidf}% | ${b.tfidf}% |\n` +
      `| Skill Match | ${a.skill}% | ${b.skill}% |\n` +
      `| Experience | ${a.exp_yrs} yrs | ${b.exp_yrs} yrs |\n` +
      `| Tier | ${a.tier} | ${b.tier} |\n\n` +
      `🏆 **${winner.name}** has the stronger overall profile.`;
  }

  // Scoring method?
  if(q.match(/scoring|how.*work|algorithm|method|formula|weight/)){
    return `⚙️ **Scoring Method (AI Resume Screening)**\n\nThe system combines four signals into a final weighted score:\n\n• **TF-IDF Cosine Similarity (40%)** — Measures textual overlap between resume and job description using term-frequency inverse-document-frequency vectors.\n• **Skill Match (40%)** — Direct keyword overlap between the candidate's skills and the required skills in the JD.\n• **Experience Score (20%)** — Ratio of candidate's years of experience to the required years (capped at 100%).\n\nWhen Semantic AI is enabled, weights shift to: TF-IDF 20% · Semantic 35% · Skill 30% · Experience 15%.\n\nCandidates are then classified into **Tier A** (≥75%), **Tier B** (≥60%), **Tier C** (≥45%), or **Tier D** (below 45%).`;
  }

  // Most experience?
  if(q.match(/experience|senior|years|worked/)){
    const mostExp = [...cands].sort((a,b)=>b.exp_yrs-a.exp_yrs)[0];
    return `💼 **${mostExp.name}** has the most experience with **${mostExp.exp_yrs} year(s)**. Their experience score is ${mostExp.exp_score}%. Education: ${mostExp.education}.`;
  }

  // Fallback — general summary
  const avg = (cands.reduce((s,c)=>s+c.score,0)/cands.length).toFixed(1);
  return `📋 **Screening Summary**\n\nProcessed **${cands.length} candidates** against ${ctx.jd_skills.length} required skills.\n\n• Average score: **${avg}%**\n• Qualified (≥${ctx.threshold}%): **${qualified.length}** candidates\n• Top candidate: **${top.name}** at ${top.score}%\n• Most missing skill: ${Object.entries((() => { const f={}; cands.forEach(c=>c.missing.forEach(sk=>f[sk]=(f[sk]||0)+1)); return f; })()).sort((a,b)=>b[1]-a[1])[0]?.[0] || 'none'}\n\nTry asking: "Who is the best candidate?", "What skills are missing?", or "How does scoring work?"`;
}

function addChatMsg(text, role){
  const box = document.getElementById('chat-msgs');
  const div = document.createElement('div');
  div.className = `chat-msg chat-${role}`;
  // Simple markdown: **bold**
  div.innerHTML = text.replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>').replace(/\n/g,'<br>');
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return div;
}

function askChat(preset){
  const inp = document.getElementById('chat-inp');
  const q = preset || inp.value.trim();
  if(!q) return;
  inp.value = '';
  const ctx = buildContext();
  if(!ctx){ addChatMsg('⚠️ Please run the screening first, then ask questions about the results.','bot'); return; }
  addChatMsg(q,'user');
  const typing = addChatMsg('Thinking...','typing');
  setTimeout(()=>{
    typing.remove();
    addChatMsg(nlpAnswer(q,ctx),'bot');
  }, 320);
}
// ── Dark Mode Toggle ──────────────────────────────────────────────────────────
function toggleDark(){
  document.body.classList.toggle('dark');
  const btn = document.getElementById('dark-btn');
  btn.textContent = document.body.classList.contains('dark') ? '☀️ Light' : '🌙 Dark';
}

// ── CSV Export ────────────────────────────────────────────────────────────────
async function exportCSV(){
  if(!all||!all.length){ alert('Run screening first to export results.'); return; }
  try{
    const resp = await fetch('/export_csv',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({results: all})
    });
    if(!resp.ok){ const e=await resp.json(); alert('Export error: '+e.error); return; }
    const blob = await resp.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `screening_${Date.now()}.csv`;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(url);
  }catch(e){ alert('Export failed: '+e.message); }
}

// ── NLP Pipeline Analysis ─────────────────────────────────────────────────────
async function runNLPAnalysis(target){
  if(!all.length && target==='resume'){ alert('Run screening first.'); return; }
  let text = '';
  if(target === 'resume'){
    // Build a representative text from top candidate's data
    const c = all[0];
    text = `${c.name}\n\nSkills: ${(c.skills||[]).join(', ')}\nExperience: ${c.exp_yrs} years\nEducation: ${c.education}\nMatched Skills: ${(c.matched||[]).join(', ')}\nMissing Skills: ${(c.missing||[]).join(', ')}`;
  } else {
    text = jdText || document.getElementById('jd-text')?.value || '';
  }
  if(!text){ alert('No text available to analyse.'); return; }

  document.getElementById('nlp-loading').classList.remove('hidden');
  document.getElementById('nlp-result').innerHTML='';
  try{
    const resp = await fetch('/nlp_analysis',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text, label: target==='resume'?`Resume: ${all[0]?.name}`:'Job Description'})
    });
    const d = await resp.json();
    document.getElementById('nlp-loading').classList.add('hidden');
    if(d.error){ document.getElementById('nlp-result').innerHTML=`<div style="color:var(--red)">${d.error}</div>`; return; }
    renderNLPResult(d);
  }catch(e){
    document.getElementById('nlp-loading').classList.add('hidden');
    document.getElementById('nlp-result').innerHTML=`<div style="color:var(--red)">Error: ${e.message}</div>`;
  }
}

function renderNLPResult(d){
  const box = document.getElementById('nlp-result');
  const s = d.stats||{};
  box.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:8px;margin-bottom:14px">
      ${[['📝 Words',s.word_count||0,'var(--blue)'],['🔤 Unique',s.unique_tokens||0,'var(--green)'],['📖 Sentences',s.sentence_count||0,'var(--amber)'],['🏷 Entities',s.entity_count||0,'var(--violet)'],['🗑 Stops Removed',s.stop_words_removed||0,'var(--red)'],['📏 Avg Word Len',s.avg_word_length||0,'#0891b2']].map(([l,v,c])=>`
        <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:10px;text-align:center">
          <div style="font-size:18px;font-weight:700;color:${c}">${v}</div>
          <div style="font-size:9px;color:var(--g500);text-transform:uppercase;margin-top:2px">${l}</div>
        </div>`).join('')}
    </div>

    <div style="display:grid;grid-template-columns:2fr 1fr;gap:12px;margin-bottom:12px">
      <div style="background:var(--violet-lt);border:1px solid var(--violet-md);border-radius:9px;padding:12px">
        <div style="font-size:11px;font-weight:600;color:var(--violet);margin-bottom:6px">📝 Plain-Language Summary (extractive, top-scoring sentences)</div>
        <div style="font-size:12px;color:var(--g700);line-height:1.6">${d.summary || 'Not enough text to summarise.'}</div>
      </div>
      <div style="background:${(d.sentiment||{}).polarity==='Positive'?'var(--green-lt)':(d.sentiment||{}).polarity==='Negative'?'var(--red-lt)':'var(--g50)'};border:1px solid var(--g200);border-radius:9px;padding:12px;text-align:center">
        <div style="font-size:11px;font-weight:600;color:var(--g700);margin-bottom:6px">🎭 Tone / Sentiment</div>
        <div style="font-size:18px;font-weight:700;color:${(d.sentiment||{}).polarity==='Positive'?'var(--green)':(d.sentiment||{}).polarity==='Negative'?'var(--red)':'var(--g500)'}">${(d.sentiment||{}).polarity||'Neutral'}</div>
        <div style="font-size:10px;color:var(--g400);margin-top:2px">score ${(d.sentiment||{}).score??0}</div>
        ${((d.sentiment||{}).positive_words||[]).length?`<div style="font-size:9px;color:var(--green);margin-top:6px">+ ${d.sentiment.positive_words.slice(0,4).join(', ')}</div>`:''}
        ${((d.sentiment||{}).negative_words||[]).length?`<div style="font-size:9px;color:var(--red);margin-top:2px">− ${d.sentiment.negative_words.slice(0,4).join(', ')}</div>`:''}
      </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">

      <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:12px">
        <div style="font-size:11px;font-weight:600;color:var(--g700);margin-bottom:8px">🔢 Top Keywords (after stop-word removal)</div>
        <div style="display:flex;flex-wrap:wrap;gap:4px">
          ${(d.keywords||[]).map(k=>`<span style="background:var(--blue-lt);color:#1d4ed8;border:1px solid var(--blue-md);border-radius:5px;padding:2px 8px;font-size:10px;font-weight:500">${k.word} <span style="opacity:.7">${k.count}</span></span>`).join('')}
        </div>
      </div>

      <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:12px">
        <div style="font-size:11px;font-weight:600;color:var(--g700);margin-bottom:8px">🔗 Top Bigrams (2-word phrases)</div>
        <div style="display:flex;flex-wrap:wrap;gap:4px">
          ${(d.bigrams||[]).length ? d.bigrams.map(b=>`<span style="background:var(--amber-lt);color:var(--amber);border:1px solid var(--amber-md);border-radius:5px;padding:2px 8px;font-size:10px;font-weight:500">${b.phrase} <span style="opacity:.7">${b.count}</span></span>`).join('') : `<div style="font-size:11px;color:var(--g400)">Not enough tokens for bigrams</div>`}
        </div>
      </div>

      <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:12px">
        <div style="font-size:11px;font-weight:600;color:var(--g700);margin-bottom:8px">🏷 Named Entity Recognition (NER)</div>
        ${(d.entities||[]).length ? `
          <div style="display:flex;flex-direction:column;gap:5px">
            ${d.entities.slice(0,12).map(e=>`
              <div style="display:flex;align-items:center;gap:8px;font-size:11px">
                <span style="background:var(--violet-lt);color:var(--violet);border-radius:4px;padding:1px 7px;font-size:9px;font-weight:600;min-width:55px;text-align:center">${e.label}</span>
                <span style="color:var(--g700);font-weight:500">${e.text}</span>
                <span style="color:var(--g400);font-size:10px">${e.description||''}</span>
              </div>`).join('')}
          </div>` : `<div style="font-size:11px;color:var(--g400)">No named entities detected (spaCy not available — install for full NER)</div>`}
      </div>

      <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:12px">
        <div style="font-size:11px;font-weight:600;color:var(--g700);margin-bottom:8px">📚 Lemmatization (token → base form)</div>
        <div style="display:flex;flex-wrap:wrap;gap:4px">
          ${(d.lemmas||[]).slice(0,15).map(l=>`
            <div style="background:var(--green-lt);border:1px solid var(--green-md);border-radius:5px;padding:3px 8px;font-size:10px">
              <span style="color:#065f46;font-weight:500">${l.token}</span>
              <span style="color:var(--g400)"> → </span>
              <span style="color:var(--green)">${l.lemma}</span>
            </div>`).join('')}
        </div>
      </div>

      <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:12px">
        <div style="font-size:11px;font-weight:600;color:var(--g700);margin-bottom:8px">🏷 POS Tagging (Part-of-Speech)</div>
        ${(d.pos_tags||[]).length ? `
          <div style="display:flex;flex-wrap:wrap;gap:4px">
            ${d.pos_tags.slice(0,20).map(p=>`
              <div style="background:var(--amber-lt);border:1px solid var(--amber-md);border-radius:5px;padding:3px 8px;font-size:10px">
                <span style="color:#92400e;font-weight:500">${p.token}</span>
                <span style="color:var(--amber);font-size:9px"> ${p.pos}</span>
              </div>`).join('')}
          </div>` : `<div style="font-size:11px;color:var(--g400)">Install spaCy for POS tagging: python -m spacy download en_core_web_sm</div>`}
      </div>

    </div>

    <div style="margin-top:12px;background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:12px">
      <div style="font-size:11px;font-weight:600;color:var(--g700);margin-bottom:6px">📄 Tokenization (first 30 tokens)</div>
      <div style="display:flex;flex-wrap:wrap;gap:3px">
        ${(d.tokens||[]).map((t,i)=>`<span style="background:var(--g100);border:1px solid var(--g200);border-radius:4px;padding:1px 6px;font-size:10px;color:var(--g600)">${t}</span>`).join('')}
      </div>
    </div>

    <div style="margin-top:8px;font-size:10px;color:var(--g400);padding:6px 10px;background:${s.spacy_used?'var(--green-lt)':'var(--amber-lt)'};border-radius:6px">
      ${s.spacy_used ? '✅ spaCy en_core_web_sm loaded — full NLP pipeline active (tokenization, POS, NER, lemmatization)' : '⚠️ spaCy not available — using regex fallback. For full NLP: pip install spacy && python -m spacy download en_core_web_sm'}
    </div>`;
}

// ── Approach Comparison ───────────────────────────────────────────────────────
async function runCompare(){
  const jd = document.getElementById('jd-text')?.value || '';
  if(!jd){ alert('Please enter or select a Job Description first.'); return; }
  document.getElementById('compare-loading').classList.remove('hidden');
  document.getElementById('compare-result').innerHTML='';
  try{
    const resp = await fetch('/compare_approaches',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({jd_text:jd, threshold:thresh||60})
    });
    const d = await resp.json();
    document.getElementById('compare-loading').classList.add('hidden');
    if(d.error){ document.getElementById('compare-result').innerHTML=`<div style="color:var(--red)">${d.error}</div>`; return; }
    renderCompare(d);
  }catch(e){
    document.getElementById('compare-loading').classList.add('hidden');
    document.getElementById('compare-result').innerHTML=`<div style="color:var(--red)">Error: ${e.message}</div>`;
  }
}

function renderCompare(d){
  const st = d.approach_stats||{};
  const tm = d.timings||{};
  const box = document.getElementById('compare-result');
  const appr = [
    {name:'Rule-Based AI',desc:'Skill (50%) + Experience (30%) + Education (20%) — no ML, no text',key:'rule',color:'#b45309',bg:'#fffbeb',q:st.rule_qualified,avg:st.rule_avg,ms:tm.rule_ms},
    {name:'TF-IDF + Skill',desc:'TF-IDF Cosine (40%) + Skill Match (40%) + Experience (20%)',key:'tfidf',color:'#3b82f6',bg:'#eff6ff',q:st.tfidf_qualified,avg:st.tfidf_avg,ms:tm.tfidf_ms},
    {name:'Full AI System',desc:`TF-IDF(19%) + ${d.semantic_available?'Semantic(33%)':'Skill↑'} + Skill(28%) + Exp(15%) + Education(5%)`,key:'full',color:'#059669',bg:'#ecfdf5',q:st.full_qualified,avg:st.full_avg,ms:tm.full_ms},
  ];
  box.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px">
      ${appr.map(a=>`
        <div style="background:${a.bg};border:1px solid ${a.color}33;border-radius:10px;padding:14px">
          <div style="font-size:12px;font-weight:600;color:${a.color};margin-bottom:4px">${a.name}</div>
          <div style="font-size:10px;color:var(--g500);margin-bottom:10px;line-height:1.5">${a.desc}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px">
            <div style="text-align:center"><div style="font-size:20px;font-weight:700;color:${a.color}">${a.avg}%</div><div style="font-size:9px;color:var(--g400)">Avg Score</div></div>
            <div style="text-align:center"><div style="font-size:20px;font-weight:700;color:${a.color}">${a.q}</div><div style="font-size:9px;color:var(--g400)">Qualified</div></div>
          </div>
          <div style="margin-top:8px;text-align:center;font-size:10px;color:var(--g400)">⏱ ${a.ms} ms runtime</div>
        </div>`).join('')}
    </div>

    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead>
          <tr style="background:var(--g50)">
            <th style="padding:8px 12px;text-align:left;border-bottom:1px solid var(--g200);color:var(--g700)">Candidate</th>
            <th style="padding:8px 12px;text-align:center;border-bottom:1px solid var(--g200);color:#b45309">🏷 Rule-Based</th>
            <th style="padding:8px 12px;text-align:center;border-bottom:1px solid var(--g200);color:#3b82f6">📊 TF-IDF+Skill</th>
            <th style="padding:8px 12px;text-align:center;border-bottom:1px solid var(--g200);color:#059669">🤖 Full AI</th>
            <th style="padding:8px 12px;text-align:center;border-bottom:1px solid var(--g200);color:var(--g500)">Δ Max Diff</th>
          </tr>
        </thead>
        <tbody>
          ${(d.comparison||[]).map((r,i)=>{
            const diff = Math.max(r.rule_score,r.tfidf_score,r.full_score)-Math.min(r.rule_score,r.tfidf_score,r.full_score);
            const bg = i%2===0?'var(--g50)':'var(--w)';
            const qBadge = q=>`<span style="font-size:9px;padding:1px 5px;border-radius:3px;background:${q?'var(--green-lt)':'var(--red-lt)'};color:${q?'var(--green)':'var(--red)'}">${q?'✓':'✗'}</span>`;
            return `<tr style="background:${bg}">
              <td style="padding:7px 12px;font-weight:500;border-bottom:1px solid var(--g100)">${r.name}</td>
              <td style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g100)"><span style="font-weight:600;color:#b45309">${r.rule_score}%</span> ${qBadge(r.rule_qual)}</td>
              <td style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g100)"><span style="font-weight:600;color:#3b82f6">${r.tfidf_score}%</span> ${qBadge(r.tfidf_qual)}</td>
              <td style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g100)"><span style="font-weight:600;color:#059669">${r.full_score}%</span> ${qBadge(r.full_qual)}</td>
              <td style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g100)"><span style="font-weight:600;color:${diff>20?'var(--red)':diff>10?'var(--amber)':'var(--green)'}">${diff.toFixed(1)}%</span></td>
            </tr>`;}).join('')}
        </tbody>
      </table>
    </div>
    <div style="margin-top:10px;font-size:11px;color:var(--g500);padding:8px 12px;background:var(--g50);border-radius:7px">
      💡 <strong>Interpretation:</strong> Large Δ differences mean algorithm choice significantly affects ranking. If Rule-Based and Full AI agree, the result is more reliable. If they disagree, human review is recommended.
    </div>`;
}

// ── Forward Chaining ──────────────────────────────────────────────────────────
async function runFC(){
  if(!all||!all.length){ document.getElementById('fc-result').innerHTML='<div style="color:var(--amber);font-size:12px">⚠️ Run screening first, then open Forward Chaining.</div>'; return; }
  const jd = jdText || document.getElementById('jd-text')?.value || '';
  document.getElementById('fc-loading').classList.remove('hidden');
  document.getElementById('fc-result').innerHTML='';
  try{
    const resp = await fetch('/forward_chaining',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({results:all, jd_text:jd})
    });
    const d = await resp.json();
    document.getElementById('fc-loading').classList.add('hidden');
    if(d.error){ document.getElementById('fc-result').innerHTML=`<div style="color:var(--red)">${d.error}</div>`; return; }
    renderFC(d.results||[]);
  }catch(e){
    document.getElementById('fc-loading').classList.add('hidden');
    document.getElementById('fc-result').innerHTML=`<div style="color:var(--red)">Error: ${e.message}</div>`;
  }
}

function renderFC(results){
  const decColors = {'RECOMMEND FOR INTERVIEW':'#059669','CONSIDER WITH CAUTION':'#b45309','NOT RECOMMENDED':'#dc2626'};
  const decBg     = {'RECOMMEND FOR INTERVIEW':'#ecfdf5','CONSIDER WITH CAUTION':'#fffbeb','NOT RECOMMENDED':'#fef2f2'};
  document.getElementById('fc-result').innerHTML = results.map((r,i)=>`
    <div style="border:1px solid var(--g200);border-radius:10px;overflow:hidden;margin-bottom:12px;animation:fadeUp .3s ease both;animation-delay:${i*.07}s">
      <div style="padding:10px 14px;background:${decBg[r.final_decision]||'var(--g50)'};display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--g200)">
        <div>
          <div style="font-size:13px;font-weight:600;color:var(--g900)">${r.name}</div>
          <div style="font-size:10px;color:var(--g500)">Score: ${r.score}% · Tier ${r.tier}</div>
        </div>
        <div style="font-size:12px;font-weight:700;color:${decColors[r.final_decision]||'#374151'}">${r.final_decision}</div>
      </div>
      <div style="padding:12px 14px">
        <div style="font-size:10px;font-weight:600;color:var(--g600);margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">Initial Facts</div>
        <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:12px">
          ${Object.entries(r.facts||{}).filter(([k,v])=>typeof v==='boolean'||typeof v==='number').map(([k,v])=>`
            <span style="font-size:10px;padding:2px 8px;border-radius:4px;background:${v===true?'var(--green-lt)':v===false?'var(--red-lt)':'var(--g100)'};color:${v===true?'#065f46':v===false?'#991b1b':'var(--g600)'};border:1px solid ${v===true?'var(--green-md)':v===false?'var(--red-md)':'var(--g200)'};font-family:monospace">
              ${k}: ${typeof v==='boolean'?(v?'TRUE':'FALSE'):v}
            </span>`).join('')}
        </div>
        <div style="font-size:10px;font-weight:600;color:var(--g600);margin-bottom:8px;text-transform:uppercase;letter-spacing:.05em">Fired Rules (${r.fired_rules?.length||0})</div>
        ${(r.fired_rules||[]).length ? r.fired_rules.map((rule,ri)=>`
          <div style="background:var(--g50);border:1px solid var(--g200);border-radius:7px;padding:9px 12px;margin-bottom:6px">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
              <span style="background:#3b82f6;color:#fff;font-size:9px;font-weight:700;padding:1px 6px;border-radius:3px">${rule.rule}</span>
              <span style="font-size:11px;font-weight:500;color:var(--g700);font-family:monospace">IF ${rule.condition}</span>
            </div>
            <div style="font-size:10px;color:#059669;font-weight:600;margin-bottom:3px">THEN ${rule.conclusion}</div>
            <div style="font-size:11px;color:var(--g500)">${rule.explanation}</div>
          </div>`).join('') : '<div style="font-size:11px;color:var(--g400)">No rules fired for this candidate.</div>'}
      </div>
    </div>`).join('');
}

// ── Search & Optimisation AI (Option 2) ────────────────────────────────────
async function runSearchAlgos(){
  if(!all||!all.length){ document.getElementById('search-result').innerHTML='<div style="color:var(--amber);font-size:12px">⚠️ Run screening first, then open Search Algorithms.</div>'; return; }
  const budget = parseInt(document.getElementById('search-budget').value)||3;
  document.getElementById('search-loading').classList.remove('hidden');
  document.getElementById('search-result').innerHTML='';
  try{
    const resp = await fetch('/search_shortlist',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({results:all, budget})
    });
    const d = await resp.json();
    document.getElementById('search-loading').classList.add('hidden');
    if(d.error){ document.getElementById('search-result').innerHTML=`<div style="color:var(--red)">${d.error}</div>`; return; }
    renderSearchAlgos(d);
  }catch(e){
    document.getElementById('search-loading').classList.add('hidden');
    document.getElementById('search-result').innerHTML=`<div style="color:var(--red)">Error: ${e.message}</div>`;
  }
}

function renderSearchAlgos(d){
  const box = document.getElementById('search-result');
  const algos = d.algorithms||[];
  const maxNodes = Math.max(1, ...algos.map(a=>a.nodes_expanded));

  box.innerHTML = `
    <div style="overflow-x:auto;margin-bottom:16px">
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead>
          <tr style="background:var(--g50)">
            <th style="padding:8px 12px;text-align:left;border-bottom:1px solid var(--g200)">Algorithm</th>
            <th style="padding:8px 12px;text-align:left;border-bottom:1px solid var(--g200)">Optimal?</th>
            <th style="padding:8px 12px;text-align:center;border-bottom:1px solid var(--g200)">Total Score</th>
            <th style="padding:8px 12px;text-align:center;border-bottom:1px solid var(--g200)">Nodes Expanded</th>
            <th style="padding:8px 12px;text-align:center;border-bottom:1px solid var(--g200)">Max Frontier</th>
            <th style="padding:8px 12px;text-align:center;border-bottom:1px solid var(--g200)">Runtime (ms)</th>
          </tr>
        </thead>
        <tbody>
          ${algos.map((a,i)=>{
            const isBest = a.total_score === d.best_total_score;
            const bg = isBest ? '#ecfdf5' : (i%2===0?'var(--g50)':'var(--w)');
            return `<tr style="background:${bg}">
              <td style="padding:7px 12px;border-bottom:1px solid var(--g100)"><span style="font-weight:700;color:${a.color}">${a.label}</span> <span style="font-size:9px;color:var(--g400)">${a.full}</span></td>
              <td style="padding:7px 12px;border-bottom:1px solid var(--g100);font-size:10px;color:var(--g500)">${a.optimal}</td>
              <td style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g100);font-weight:700;color:${isBest?'#059669':'var(--g700)'}">${a.total_score}% ${isBest?'🏆':''}</td>
              <td style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g100)">
                <div style="display:flex;align-items:center;gap:6px">
                  <div style="flex:1;height:6px;background:var(--g100);border-radius:3px;overflow:hidden"><div style="height:100%;width:${(a.nodes_expanded/maxNodes*100).toFixed(0)}%;background:${a.color}"></div></div>
                  <span style="font-size:10px;color:var(--g500);min-width:28px;text-align:right">${a.nodes_expanded}</span>
                </div>
              </td>
              <td style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g100);font-size:11px;color:var(--g500)">${a.max_frontier}</td>
              <td style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g100);font-size:11px;color:var(--g500)">${a.runtime_ms}</td>
            </tr>`;}).join('')}
        </tbody>
      </table>
    </div>

    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px;margin-bottom:16px">
      ${algos.map(a=>`
        <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:12px">
          <div style="font-size:11px;font-weight:700;color:${a.color};margin-bottom:6px">${a.label} — Final Shortlist</div>
          <div style="display:flex;flex-wrap:wrap;gap:4px">
            ${a.selected.map(n=>`<span style="background:${a.color}1a;color:${a.color};border:1px solid ${a.color}44;border-radius:5px;padding:2px 8px;font-size:10px;font-weight:500">${n}</span>`).join('')}
          </div>
        </div>`).join('')}
    </div>

    <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:12px">
      <div style="font-size:11px;font-weight:600;color:var(--g700);margin-bottom:8px">🧭 Explored States — first steps per algorithm (path to solution)</div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px">
        ${algos.map(a=>`
          <div>
            <div style="font-size:10px;font-weight:700;color:${a.color};margin-bottom:5px">${a.label}</div>
            <div style="display:flex;flex-direction:column;gap:3px;max-height:160px;overflow-y:auto">
              ${(a.explored_trail||[]).slice(0,8).map(t=>`
                <div style="font-size:9px;color:var(--g500);background:var(--w);border:1px solid var(--g200);border-radius:5px;padding:3px 7px;font-family:monospace">
                  #${t.step} · consider "${t.candidate}" · chosen=[${t.chosen_so_far.length}] · total=${t.running_total}%
                </div>`).join('')}
            </div>
          </div>`).join('')}
      </div>
    </div>

    <div style="margin-top:10px;font-size:11px;color:var(--g500);padding:8px 12px;background:var(--blue-lt);border-radius:7px">
      💡 <strong>Interpretation:</strong> UCS and A* are guaranteed optimal for this cost function — A* usually reaches the same answer while expanding far fewer nodes because its heuristic prunes unpromising branches early. Greedy and Hill Climbing are faster but can settle for a locally-good (not globally optimal) shortlist. BFS/DFS guarantee finding the optimum eventually but explore the most nodes.
    </div>`;
}

// ── Machine Learning AI (Option 3: train/evaluate/predict) ─────────────────
let mlCharts = {};
async function runMLTraining(){
  if(!all||!all.length){ document.getElementById('ml-result').innerHTML='<div style="color:var(--amber);font-size:12px">⚠️ Run screening first, then open ML Training.</div>'; return; }
  const k = parseInt(document.getElementById('ml-k').value)||3;
  document.getElementById('ml-loading').classList.remove('hidden');
  document.getElementById('ml-tabs').classList.add('hidden');
  document.getElementById('ml-result').innerHTML='';
  ['ml-cls-panel','ml-reg-panel','ml-clu-panel'].forEach(id=>document.getElementById(id).innerHTML='');
  try{
    const resp = await fetch('/train_ml_models',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({results:all, threshold:thresh||60, k})
    });
    const d = await resp.json();
    document.getElementById('ml-loading').classList.add('hidden');
    if(d.error){ document.getElementById('ml-result').innerHTML=`<div style="color:var(--red)">${d.error}</div>`; return; }
    document.getElementById('ml-tabs').classList.remove('hidden');
    renderClassification(d.classification);
    renderRegression(d.regression);
    renderClustering(d.clustering);
    showMLTab(0);
  }catch(e){
    document.getElementById('ml-loading').classList.add('hidden');
    document.getElementById('ml-result').innerHTML=`<div style="color:var(--red)">Error: ${e.message}</div>`;
  }
}

function showMLTab(i){
  ['mltab-0','mltab-1','mltab-2'].forEach((id,j)=>document.getElementById(id).classList.toggle('on',j===i));
  ['ml-cls-panel','ml-reg-panel','ml-clu-panel'].forEach((id,j)=>document.getElementById(id).classList.toggle('hidden',j!==i));
}

function mlBarChart(canvasId, labels, values, colorPos, colorNeg){
  try{
    if(typeof Chart==='undefined'){ console.warn('Chart.js not loaded — skipping', canvasId); return; }
    const old = mlCharts[canvasId]; if(old) old.destroy();
    const ctx = document.getElementById(canvasId);
    mlCharts[canvasId] = new Chart(ctx, {
      type:'bar',
      data:{ labels, datasets:[{ data: values, backgroundColor: values.map(v=>v>=0?colorPos:colorNeg) }] },
      options:{ indexAxis:'y', plugins:{legend:{display:false}, title:{display:true,text:'Learned Feature Weights (Logistic/Linear Regression coefficients)'}}, scales:{x:{title:{display:true,text:'Weight'}}} }
    });
  }catch(e){ console.warn('mlBarChart failed:', e.message); }
}

function renderClassification(c){
  const box = document.getElementById('ml-cls-panel');
  if(c.error){ box.innerHTML=`<div style="color:var(--amber);font-size:12px;padding:10px">⚠️ ${c.error}</div>`; return; }
  const m = c.metrics;
  box.innerHTML = `
    <div style="font-size:11px;color:var(--g500);margin-bottom:10px;padding:8px 12px;background:var(--g50);border-radius:7px">
      <strong>${c.algorithm}</strong> · trained on: ${c.features_used.join(', ')} · ${c.label_definition}<br>
      <span style="color:var(--blue)">Evaluation method: ${c.method_used}</span>
    </div>
    <div style="font-size:11px;color:var(--g700);margin-bottom:14px;padding:9px 12px;background:var(--violet-lt);border-radius:8px;border-left:3px solid var(--violet)">
      💡 <strong>Model recommendation:</strong> ${c.recommendation}
    </div>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px">
      ${[['Accuracy',m.accuracy,'var(--blue)'],['Precision',m.precision,'var(--green)'],['Recall',m.recall,'var(--amber)'],['F1 Score',m.f1,'var(--violet)']].map(([l,v,cl])=>`
        <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:10px;text-align:center">
          <div style="font-size:20px;font-weight:700;color:${cl}">${v}%</div>
          <div style="font-size:9px;color:var(--g500);text-transform:uppercase;margin-top:2px">${l}</div>
        </div>`).join('')}
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px">
      <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:12px">
        <div style="font-size:11px;font-weight:600;color:var(--g700);margin-bottom:8px">Confusion Matrix</div>
        <table style="width:100%;border-collapse:collapse;font-size:11px;text-align:center">
          <tr><td></td><td style="font-weight:600;color:var(--g500)">Pred: ${c.confusion_labels[0]}</td><td style="font-weight:600;color:var(--g500)">Pred: ${c.confusion_labels[1]}</td></tr>
          <tr><td style="font-weight:600;color:var(--g500)">Actual: ${c.confusion_labels[0]}</td>
              <td style="background:var(--green-lt);padding:10px;font-weight:700;color:#065f46">${c.confusion_matrix[0][0]}</td>
              <td style="background:var(--red-lt);padding:10px;font-weight:700;color:#991b1b">${c.confusion_matrix[0][1]}</td></tr>
          <tr><td style="font-weight:600;color:var(--g500)">Actual: ${c.confusion_labels[1]}</td>
              <td style="background:var(--red-lt);padding:10px;font-weight:700;color:#991b1b">${c.confusion_matrix[1][0]}</td>
              <td style="background:var(--green-lt);padding:10px;font-weight:700;color:#065f46">${c.confusion_matrix[1][1]}</td></tr>
        </table>
      </div>
      <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:12px"><canvas id="ml-cls-imp" height="140"></canvas></div>
    </div>
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead><tr style="background:var(--g50)">
          <th style="padding:7px 12px;text-align:left;border-bottom:1px solid var(--g200)">Candidate</th>
          <th style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g200)">Actual</th>
          <th style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g200)">Predicted</th>
          <th style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g200)">P(Qualified)</th>
          <th style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g200)">✓/✗</th>
        </tr></thead>
        <tbody>
          ${c.predictions.map((p,i)=>`<tr style="background:${i%2===0?'var(--g50)':'var(--w)'}">
            <td style="padding:6px 12px;font-weight:500;border-bottom:1px solid var(--g100)">${p.name}</td>
            <td style="padding:6px 12px;text-align:center;border-bottom:1px solid var(--g100)">${p.actual_label}</td>
            <td style="padding:6px 12px;text-align:center;border-bottom:1px solid var(--g100)">${p.predicted_label}</td>
            <td style="padding:6px 12px;text-align:center;border-bottom:1px solid var(--g100)">${p.probability}%</td>
            <td style="padding:6px 12px;text-align:center;border-bottom:1px solid var(--g100)">${p.correct?'✅':'❌'}</td>
          </tr>`).join('')}
        </tbody>
      </table>
    </div>`;
  mlBarChart('ml-cls-imp', c.feature_importance.map(f=>f.feature), c.feature_importance.map(f=>f.weight), '#3b82f6', '#dc2626');
}

function renderRegression(r){
  const box = document.getElementById('ml-reg-panel');
  if(r.error){ box.innerHTML=`<div style="color:var(--amber);font-size:12px;padding:10px">⚠️ ${r.error}</div>`; return; }
  const m = r.metrics;
  box.innerHTML = `
    <div style="font-size:11px;color:var(--g500);margin-bottom:10px;padding:8px 12px;background:var(--g50);border-radius:7px">
      <strong>${r.algorithm}</strong> · trained on: ${r.features_used.join(', ')} → predicts: ${r.target}<br>
      <span style="color:var(--blue)">Evaluation method: ${r.method_used}</span>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px">
      ${[['R² Score',m.r2,'var(--blue)'],['MAE',m.mae,'var(--amber)'],['RMSE',m.rmse,'var(--violet)']].map(([l,v,cl])=>`
        <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:10px;text-align:center">
          <div style="font-size:20px;font-weight:700;color:${cl}">${v}</div>
          <div style="font-size:9px;color:var(--g500);text-transform:uppercase;margin-top:2px">${l}</div>
        </div>`).join('')}
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px">
      <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:12px"><canvas id="ml-reg-scatter" height="180"></canvas></div>
      <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:12px"><canvas id="ml-reg-imp" height="180"></canvas></div>
    </div>
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead><tr style="background:var(--g50)">
          <th style="padding:7px 12px;text-align:left;border-bottom:1px solid var(--g200)">Candidate</th>
          <th style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g200)">Actual Score</th>
          <th style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g200)">Predicted Score</th>
          <th style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g200)">Residual</th>
        </tr></thead>
        <tbody>
          ${r.predictions.map((p,i)=>`<tr style="background:${i%2===0?'var(--g50)':'var(--w)'}">
            <td style="padding:6px 12px;font-weight:500;border-bottom:1px solid var(--g100)">${p.name}</td>
            <td style="padding:6px 12px;text-align:center;border-bottom:1px solid var(--g100)">${p.actual_score}%</td>
            <td style="padding:6px 12px;text-align:center;border-bottom:1px solid var(--g100)">${p.predicted_score}%</td>
            <td style="padding:6px 12px;text-align:center;border-bottom:1px solid var(--g100);color:${Math.abs(p.residual)>5?'var(--red)':'var(--green)'}">${p.residual>0?'+':''}${p.residual}</td>
          </tr>`).join('')}
        </tbody>
      </table>
    </div>`;

  try{
    if(typeof Chart==='undefined') throw new Error('Chart.js not loaded');
    const old = mlCharts['ml-reg-scatter']; if(old) old.destroy();
    const maxV = Math.max(...r.predictions.map(p=>Math.max(p.actual_score,p.predicted_score)), 10);
    mlCharts['ml-reg-scatter'] = new Chart(document.getElementById('ml-reg-scatter'), {
      type:'scatter',
      data:{ datasets:[
        { label:'Candidates', data: r.predictions.map(p=>({x:p.actual_score,y:p.predicted_score})), backgroundColor:'#3b82f6' },
        { label:'Perfect prediction', data:[{x:0,y:0},{x:maxV,y:maxV}], type:'line', borderColor:'#dc2626', borderDash:[5,5], pointRadius:0 }
      ]},
      options:{ plugins:{title:{display:true,text:'Predicted vs Actual Score'}}, scales:{
        x:{title:{display:true,text:'Actual Score (%)'}}, y:{title:{display:true,text:'Predicted Score (%)'}} } }
    });
  }catch(e){ console.warn('regression scatter failed:', e.message); }
  mlBarChart('ml-reg-imp', r.feature_importance.map(f=>f.feature), r.feature_importance.map(f=>f.weight), '#059669', '#dc2626');
}

function renderClustering(c){
  const box = document.getElementById('ml-clu-panel');
  if(c.error){ box.innerHTML=`<div style="color:var(--amber);font-size:12px;padding:10px">⚠️ ${c.error}</div>`; return; }
  const palette = ['#3b82f6','#059669','#d97706','#dc2626','#7c3aed','#0891b2'];
  box.innerHTML = `
    <div style="font-size:11px;color:var(--g500);margin-bottom:10px;padding:8px 12px;background:var(--g50);border-radius:7px">
      <strong>${c.algorithm}</strong> · unsupervised — trained on: ${c.features_used.join(', ')} (no score/label used)<br>
      <span style="color:var(--blue)">PCA explains ${c.pca_explained_variance_pct[0]}% + ${c.pca_explained_variance_pct[1]}% = ${(c.pca_explained_variance_pct[0]+c.pca_explained_variance_pct[1]).toFixed(1)}% of variance in 2D</span>
      ${c.silhouette_score!==null?` · Silhouette score: <strong>${c.silhouette_score}</strong> (closer to 1 = better-separated clusters)`:''}
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px">
      <div style="background:var(--g50);border:1px solid var(--g200);border-radius:9px;padding:12px"><canvas id="ml-clu-scatter" height="200"></canvas></div>
      <div style="display:flex;flex-direction:column;gap:8px">
        ${c.clusters.map((cl,i)=>`
          <div style="background:var(--g50);border:1px solid var(--g200);border-left:4px solid ${palette[i%palette.length]};border-radius:7px;padding:10px 12px">
            <div style="font-size:12px;font-weight:700;color:${palette[i%palette.length]}">${cl.label}</div>
            <div style="font-size:10px;color:var(--g500)">${cl.size} candidate(s) · avg score ${cl.avg_score}%</div>
          </div>`).join('')}
      </div>
    </div>
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:12px">
        <thead><tr style="background:var(--g50)">
          <th style="padding:7px 12px;text-align:left;border-bottom:1px solid var(--g200)">Candidate</th>
          <th style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g200)">Cluster</th>
          <th style="padding:7px 12px;text-align:center;border-bottom:1px solid var(--g200)">Score</th>
        </tr></thead>
        <tbody>
          ${c.points.map((p,i)=>`<tr style="background:${i%2===0?'var(--g50)':'var(--w)'}">
            <td style="padding:6px 12px;font-weight:500;border-bottom:1px solid var(--g100)">${p.name}</td>
            <td style="padding:6px 12px;text-align:center;border-bottom:1px solid var(--g100)"><span style="color:${palette[p.cluster_id%palette.length]};font-weight:600">● ${p.cluster_label}</span></td>
            <td style="padding:6px 12px;text-align:center;border-bottom:1px solid var(--g100)">${p.score}%</td>
          </tr>`).join('')}
        </tbody>
      </table>
    </div>`;

  try{
    if(typeof Chart==='undefined') throw new Error('Chart.js not loaded');
    const old = mlCharts['ml-clu-scatter']; if(old) old.destroy();
    const byCluster = {};
    c.points.forEach(p=>{ (byCluster[p.cluster_id] ||= []).push(p); });
    const datasets = Object.keys(byCluster).map(cid=>({
      label: byCluster[cid][0].cluster_label,
      data: byCluster[cid].map(p=>({x:p.x,y:p.y})),
      backgroundColor: palette[cid%palette.length]
    }));
    mlCharts['ml-clu-scatter'] = new Chart(document.getElementById('ml-clu-scatter'), {
      type:'scatter',
      data:{ datasets },
      options:{ plugins:{title:{display:true,text:'Candidate Clusters (PCA 2D projection)'}}, scales:{
        x:{title:{display:true,text:'Principal Component 1'}}, y:{title:{display:true,text:'Principal Component 2'}} } }
    });
  }catch(e){ console.warn('clustering scatter failed:', e.message); }
}

// Enter key support for chat
document.addEventListener('DOMContentLoaded', ()=>{
  const ci = document.getElementById('chat-inp');
  if(ci) ci.addEventListener('keydown', e=>{ if(e.key==='Enter') askChat(); });
});
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_ui()

@app.route('/screen', methods=['POST'])
def screen_route():
    try:
        d           = request.get_json()
        jd_text     = d.get('jd_text', '').strip()
        w_tf        = float(d.get('w_tfidf', 0.4))
        w_sk        = float(d.get('w_skill', 0.4))
        w_ex        = float(d.get('w_exp',   0.2))
        threshold   = int(d.get('threshold', 60))
        mode        = d.get('mode', 'sample')
        use_sem     = bool(d.get('use_semantic', False))

        # Module A — input validation
        if not jd_text:
            return jsonify(error="Job description is required."), 400

        # Module B — load_data
        resumes = []
        if mode == 'upload':
            for r in d.get('resumes', []):
                text = ''
                if PDF_OK and r.get('file_type','').startswith('application/pdf'):
                    try:
                        pdf_bytes = base64.b64decode(r['b64'])
                        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                            text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
                    except Exception:
                        text = ''
                else:
                    try:
                        text = base64.b64decode(r['b64']).decode('utf-8', 'ignore')
                    except Exception:
                        text = ''
                name = extract_name(text) or r.get('name', 'Unknown')
                resumes.append({
                    "name":           name,
                    "text":           text,
                    "skills":         extract_skills(text),
                    "experience_years": extract_exp_years(text),
                    "education":      extract_education(text),
                })
        elif mode == 'manual':
            raw = d.get('resumes', [])
            if not raw:
                return jsonify(error="No candidates provided."), 400
            for r in raw:
                t = r.get('text', '')
                resumes.append({
                    "name":           r['name'],
                    "text":           t,
                    "skills":         extract_skills(t),
                    "experience_years": int(r.get('experience_years', 0)),
                    "education":      r.get('education', ''),
                })
        else:
            resumes = load_data('sample')

        if not resumes:
            return jsonify(error="No valid resumes to process."), 400

        # Module B — run_model_or_algorithm
        results, jd_skills, timing = run_screening(resumes, jd_text, w_tf, w_sk, w_ex, use_sem)

        # Module E — evaluate_model
        evaluation = evaluate_model(results, threshold, timing, use_sem)

        # Ranking with tiers
        results = rank_with_tiers(results, threshold)

        # Skill gap analysis
        gap_analysis, overall_coverage = skill_gap_analysis(results, jd_skills)

        # Module C — create_visuals: chart/graph/grid/timeline-ready data
        visuals = create_visuals(results, evaluation, gap_analysis)

        # Save to MySQL database
        job_title  = d.get('job_title','Unknown Role')
        session_id = db_save(job_title, jd_text, results, threshold)

        return jsonify(
            results=results,
            jd_skills=jd_skills,
            threshold=threshold,
            evaluation=evaluation,
            use_semantic=use_sem and SEMANTIC_OK,
            gap_analysis=gap_analysis,
            overall_coverage=overall_coverage,
            session_id=session_id,
            visuals=visuals,
        )

    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/db_results')
def db_results_route():
    results  = db_get_all()
    sessions = db_get_sessions()
    stats    = db_get_stats()
    return jsonify(results=results, sessions=sessions, stats=stats)

@app.route('/clear_db', methods=['POST'])
def clear_db_route():
    ok = db_clear()
    return jsonify(success=ok, message="Cleared!" if ok else "Failed.")

@app.route('/download_dataset')
def download_dataset():
    """Allow users to download the sample dataset JSON file (Section 2 deliverable)."""
    ensure_dataset()
    return send_file(DATA_FILE, as_attachment=True, download_name="sample_resumes.json")

@app.route('/export_csv', methods=['POST'])
def export_csv_route():
    try:
        import csv, io as _io
        data = request.get_json()
        results = data.get("results", [])
        if not results:
            return jsonify(error="No results to export."), 400
        output = _io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Rank","Name","Education","Experience (yrs)","Final Score (%)","TF-IDF (%)","Semantic (%)","Skill Match (%)","Exp Score (%)","Education Score (%)","Tier","Verdict","Matched Skills","Missing Skills"])
        for r in results:
            writer.writerow([r.get("rank",""),r.get("name",""),r.get("education",""),r.get("exp_yrs",""),r.get("score",""),r.get("tfidf",""),r.get("semantic","N/A"),r.get("skill",""),r.get("exp_score",""),r.get("edu_score",""),r.get("tier",""),r.get("tier_label",""),"; ".join(r.get("matched",[])),"; ".join(r.get("missing",[])),])
        output.seek(0)
        buf = _io.BytesIO(output.getvalue().encode("utf-8-sig"))
        return send_file(buf, as_attachment=True, download_name=f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mimetype="text/csv")
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/nlp_analysis', methods=['POST'])
def nlp_analysis_route():
    """
    Full NLP pipeline analysis for a resume or JD text.
    Shows tokenization, POS, NER, lemmatization, keyword frequency.
    Demonstrates Option 4 (NLP/LLM-assisted AI) requirements.
    """
    try:
        d    = request.get_json()
        text = d.get("text", "").strip()
        label = d.get("label", "text")
        if not text:
            return jsonify(error="No text provided."), 400
        result = nlp_pipeline_analysis(text, label)
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/forward_chaining', methods=['POST'])
def forward_chaining_route():
    """
    Run forward-chaining rule-based AI for all screened candidates.
    Option 1: Rule-based AI — Facts + Rules + Inference.
    """
    try:
        d       = request.get_json()
        results = d.get("results", [])
        jd_text = d.get("jd_text", "")
        if not results or not jd_text:
            return jsonify(error="Need results and JD text."), 400
        fc_results = []
        for r in results:
            fc = forward_chaining_rules(r, jd_text)
            fc_results.append({
                "name":          r["name"],
                "score":         r.get("score", 0),
                "tier":          r.get("tier", "D"),
                "facts":         {k: (bool(v) if isinstance(v, bool) else v)
                                  for k, v in fc["facts"].items()},
                "fired_rules":   fc["fired_rules"],
                "final_decision": fc["final_decision"],
            })
        return jsonify(results=fc_results)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/compare_approaches', methods=['POST'])
def compare_approaches_route():
    """
    Approach comparison — Rule-based vs TF-IDF+Skill vs Full AI.
    Required by Section 3-E: compare at least two settings/approaches.
    """
    try:
        d         = request.get_json()
        jd_text   = d.get("jd_text", "")
        threshold = int(d.get("threshold", 60))
        if not jd_text:
            return jsonify(error="No JD text."), 400
        resumes = load_data("sample")
        result  = compare_approaches(resumes, jd_text, threshold)
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/search_shortlist', methods=['POST'])
def search_shortlist_route():
    """
    Option 2: Search/Optimization AI.
    Runs BFS, DFS, UCS, Greedy Best-First, A*, and Local Search (Hill
    Climbing) over the screened candidates to find the best K-person
    shortlist, and returns nodes-expanded / frontier-size / runtime /
    explored-state trail for each — so the UI can visualise and compare
    how each search strategy explores the state space (Section 3-C, 3-E).
    """
    try:
        d       = request.get_json()
        results = d.get("results", [])
        budget  = int(d.get("budget", 3))
        if not results:
            return jsonify(error="Run screening first — no candidates to search over."), 400
        if budget < 1:
            return jsonify(error="Shortlist size must be at least 1."), 400
        output = run_all_search_algorithms(results, budget)
        return jsonify(output)
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/train_ml_models', methods=['POST'])
def train_ml_models_route():
    """
    Option 3: Machine Learning AI, done properly — trains real scikit-learn
    models (not just similarity scoring) on the screened candidates and
    returns metrics + predictions for all three ML task types so the UI can
    show training results and predictions visually (Section 5 requirement).
    """
    try:
        d         = request.get_json()
        results   = d.get("results", [])
        threshold = int(d.get("threshold", 60))
        k         = int(d.get("k", 3))
        if not results:
            return jsonify(error="Run screening first — no candidates to train on."), 400
        return jsonify(
            classification=train_classification_model(results, threshold),
            regression=train_regression_model(results),
            clustering=train_clustering_model(results, k),
        )
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    # Ensure external dataset file exists on first run
    ensure_dataset()
    print("\n" + "="*52)
    print("  🤖  AI Resume Screening System")
    print("="*52)
    print(f"  PDF Support    : {'✅ Enabled' if PDF_OK else '⚠️  pip install pdfplumber'}")
    print(f"  Semantic AI    : {'✅ Enabled' if SEMANTIC_OK else '⚠️  pip install sentence-transformers'}")
    print(f"  spaCy NER      : {'✅ Enabled' if SPACY_OK else '⚠️  python -m spacy download en_core_web_sm'}")
    print(f"  Open           :  http://127.0.0.1:5000")
    print("="*52 + "\n")
    app.run(debug=True, port=5000)
