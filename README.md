# AI Resume Screening System v3.0

**Team ID: TEAM_F2EBFE**

AI Lab Final Project — Rule-Based AI + Search/Optimisation AI + Machine
Learning AI + NLP/LLM-assisted AI, built with Flask in a single-file app.

An interactive web app that screens and ranks candidate resumes against a job
description using all four AI techniques from the project guide, and lets
you visually compare how each one performs.

---

## 0. Requirement Coverage Checklist

Every item below is implemented and verified against the running app.

| Guide requirement | Status | Where |
|---|---|---|
| A) Problem Setup — define problem, select input, validate + error messages | ✅ | Sidebar, `run()` input validation |
| B) Core Logic — modular, intermediate steps shown | ✅ | `load_data`, `preprocess_data`, `run_screening` |
| C) Visual UI — charts, tables, network/grid/timeline views, controls, status | ✅ | `renderCharts()`, 🗺 Score Grid, 🕸 Network, 🕐 Timeline |
| D) Explainability — factors/rules shown, natural-language explanation | ✅ | `generate_explanation()`, 🔗 Forward Chaining |
| E) Evaluation — accuracy/precision/recall, cost/nodes, runtime, ≥2 approaches compared | ✅ | `evaluate_model()`, `compare_approaches()`, `run_all_search_algorithms()` |
| Option 1 — Rule-based AI (forward chaining) | ✅ | `forward_chaining_rules()` |
| Option 2 — Search/Optimisation AI (BFS/DFS/UCS/Greedy/A\*/Hill Climbing) | ✅ | `search_optimal_shortlist()` |
| Option 3 — ML AI (classification/regression/clustering, trained + visualised) | ✅ | `train_classification_model()`, `train_regression_model()`, `train_clustering_model()` |
| Option 4 — NLP/LLM-assisted AI (chatbot, summary, recommendation) | ✅ | `nlp_pipeline_analysis()`, `generate_summary()`, chatbot panel |
| External AI API disclosure | ✅ N/A | No external API used — stated explicitly in-app and in `report.md` |
| UI Quality Checklist (readable, consistent, responsive, labelled graphs, no code edits needed) | ✅ | Responsive breakpoints included; sample-data mode needs zero code edits |
| Suggested function names (Section 4) | ✅ | `load_data`, `preprocess_data`, `run_screening`, `generate_explanation`, `create_visuals`, `render_ui` |
| Deliverables (source, README, requirements.txt, dataset, screenshots, report) | ✅ | This folder |


## 1. What This Project Does

**Problem:** Manually screening dozens/hundreds of resumes against a job
description is slow and inconsistent. This app automates it and — just as
importantly — **shows the user how and why** each candidate was scored the
way they were.

**Input:** A job description (typed or picked from a preset) + resumes
(sample dataset, manual entry, or uploaded PDFs).
**Output:** A ranked shortlist with scores, tiers, explanations, charts, and
an evaluation of the AI methods used.

## 2. AI Techniques Used (Section 5 of the project guide)

| Option | Technique | Where in the app |
|---|---|---|
| **1 — Rule-based AI** | Forward chaining (IF–THEN rules over facts like `high_score`, `meets_base_requirement`) | 🔗 Forward Chaining panel |
| **2 — Search / Optimisation AI** | BFS, DFS, UCS, Greedy Best-First, A\*, and Local Search (Hill Climbing) — all solving "pick the best K-candidate shortlist" | 🔍 Search Algorithms panel |
| **3 — Machine Learning AI** | **Real trained models**, not just similarity scoring: Logistic Regression (classification), Linear Regression (regression), and K-Means + PCA (clustering) — each with metrics, confusion matrix / R² / silhouette score, and visual predictions. Plus: TF-IDF cosine similarity + optional Sentence-Transformer semantic embeddings for the core screening score | 🧠 ML Training panel (trained models) + core screening run (similarity score) |
| **4 — NLP / LLM-assisted AI** | Tokenisation, stop-word removal, lemmatisation, POS tagging, NER (spaCy), keyword frequency, bigram (2-word phrase) extraction, lexicon-based sentiment/tone analysis, extractive summarisation, and a chatbot-style Q&A panel covering 15+ question types (best/worst candidate, named-candidate lookup, skill/tier/rank lookup, qualified vs not-qualified, score-threshold filters, education, statistics, and more) | 🧬 NLP Pipeline + 💬 NLP Chat panels |

### 2.1 Machine Learning AI, in detail

The core screening score (TF-IDF/semantic similarity) is a *similarity metric*,
not a trained model — so a dedicated **🧠 ML Training** panel trains three
genuine scikit-learn models on the screened candidates:

- **Classification (Logistic Regression):** predicts Qualified / Not-Qualified
  from TF-IDF + Skill + Experience scores only (education is withheld so the
  model must generalise, not just replay the scoring formula). Shows
  accuracy/precision/recall/F1, a confusion matrix, learned feature weights,
  and a per-candidate predicted-probability table.
- **Regression (Linear Regression):** predicts the final score from Skill +
  Experience only (TF-IDF withheld). Shows R²/MAE/RMSE, a predicted-vs-actual
  scatter plot, and per-candidate residuals.
- **Clustering (K-Means + PCA):** unsupervised grouping of candidates by their
  full score profile, with **no label used at all**. A PCA 2D projection
  visualises the clusters, auto-labelled "High/Moderate/Low Fit" by their
  mean score, with a silhouette score reported.

Because resume-screening demos typically have very few candidates, the app
automatically falls back from a 70/30 train-test split (≥8 samples) to
**Leave-One-Out cross-validation** for smaller samples — this is disclosed to
the user in the UI ("Evaluation method: ...") rather than hidden, and the app
also shows a clear message if a threshold makes classification impossible
(e.g. every candidate on one side of it) instead of crashing.

## 3. Project Structure

```
ProjectName_GroupName/
├── main.py                 # entire app: backend + HTML/CSS/JS UI (single file)
├── requirements.txt
├── README.md                (this file)
├── report.md                 # short report: problem, method, AI used, results
├── data/
│   └── sample_resumes.json  # auto-generated sample dataset (12 candidates)
└── screenshots/
    ├── 01_landing_problem_setup.png
    ├── 02_results_visual_ui.png
    ├── 03_nlp_pipeline_explainability.png
    ├── 04_forward_chaining_rulebased.png
    ├── 05_search_optimisation_ai.png
    ├── 06_evaluation_approach_comparison.png
    ├── 07_nlp_chatbot_explainability.png
    ├── 08_ml_classification.png
    ├── 09_ml_regression.png
    └── 10_ml_clustering.png
```

## 4. Setup

Requires **Python 3.10+**.

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional, for full NLP features) download the spaCy English model
python -m spacy download en_core_web_sm
```

### Feature flags — the app degrades gracefully

You do **not** need every optional package to run the app. Each is detected
at startup and the related feature simply turns off if missing:

| Package missing | Feature that's disabled | What still works |
|---|---|---|
| `pdfplumber` | PDF resume upload | Sample data & manual entry still work |
| `sentence-transformers` | Semantic AI matching | Falls back to TF-IDF + skill/experience scoring |
| `spacy` | POS tagging & NER | Tokenisation, lemmatisation (regex fallback), keyword frequency, and summarisation still work |
| `mysql-connector-python` / no MySQL server | Score persistence / Database History panel | Everything else works; results just aren't saved between sessions |

## 5. Running the App

```bash
python main.py
```

Then open **http://127.0.0.1:5000** in your browser.

## 6. How to Use It (Example Flow)

1. **Problem Setup** — pick a job description preset (or paste your own),
   choose *Sample Data* or *Manual* candidates (or upload PDFs), and set the
   scoring weights / qualification threshold with the sliders.
2. Click **▶ Run Screening**.
3. **Visual results** appear: ranked table, bar/pie/radar/scatter charts,
   skill-gap heatmap, network view, timeline, and score grid (via **🧰 More
   Views**).
4. Open **🧬 NLP Pipeline** to see the full linguistic breakdown (tokens,
   POS, NER, lemmas, keywords, and a plain-language summary) for the top
   resume or the JD.
5. Open **🔗 Forward Chaining** to see the rule-based reasoning behind each
   candidate's recommendation.
6. Open **🔍 Search Algorithms** to pick a shortlist size K and compare how
   BFS/DFS/UCS/Greedy/A\*/Hill Climbing each search for the optimal K-person
   shortlist — including nodes expanded, runtime, and the explored-state
   trail.
7. Open **🧠 ML Training** to train Logistic Regression (classification),
   Linear Regression (regression), and K-Means (clustering) on the current
   candidates and see accuracy/R²/silhouette scores plus visual predictions
   for each.
8. Open **⚖️ Approach Comparison** to see Rule-Based vs TF-IDF vs Full AI
   scored side-by-side on the same candidates (Evaluation Module).
9. Open **💬 NLP Chat** to ask plain-English questions about the results.
10. Use **📥 Export CSV** to download the ranked results.

## 7. Troubleshooting

- **Port already in use:** change `app.run(debug=True, port=5000)` at the
  bottom of `main.py` to a free port.
- **`ModuleNotFoundError`:** re-run `pip install -r requirements.txt` inside
  your active virtual environment.
- **MySQL errors printed at startup:** harmless — the app logs a warning and
  continues without database persistence.

## 8. Viva / Presentation Prep

Straight answers to the guide's 5 viva questions:

**Q1. Why is this problem relevant?**
Recruiters routinely get far more resumes than they can carefully read, and
manual screening is slow and inconsistent between reviewers. Candidates also
get no visibility into why they were or weren't shortlisted. Automating the
screening — while keeping every decision explainable — solves both problems
at once.

**Q2. Why was each algorithm/model selected?**
- *Forward chaining* fits qualification decisions naturally: they already
  are a set of IF–THEN business rules ("if score ≥ X and experience ≥ Y →
  recommend").
- *BFS/DFS/UCS/Greedy/A\*/Hill Climbing* were all implemented on the same
  "pick the best K-candidate shortlist" problem specifically so their
  trade-offs (completeness vs. speed, nodes expanded) could be compared
  head-to-head, not just described.
- *Logistic/Linear Regression + K-Means* were chosen because they're the
  standard, interpretable baseline for classification/regression/clustering
  respectively — appropriate given the small, tabular candidate-feature data.
- *TF-IDF + optional Sentence-Transformers* were chosen for the core score
  because resume-JD matching is fundamentally a text-similarity problem;
  TF-IDF handles exact keyword overlap, embeddings add meaning-based matches.

**Q3. What data is used and how is it processed?**
A 12-candidate sample resume dataset (`data/sample_resumes.json`), or the
user's own manual entries / uploaded PDFs. Each resume is cleaned and
tokenised (`preprocess_data`), skills/experience/education are extracted via
keyword and regex matching, then vectorised (TF-IDF, optionally semantic
embeddings) and scored against the job description.

**Q4. How does the UI help understand the result?**
Every score is broken into its components (TF-IDF/semantic/skill/experience)
in the results table; charts show the score distribution and skill gaps;
the Forward Chaining panel shows every rule that fired; the Search
Algorithms panel visualises explored states, not just a final answer; and
the ML Training panel shows confusion matrices, R², and predicted-vs-actual
plots rather than a single accuracy number.

**Q5. What AI component was added and why, and what are the limitations?**
All four options from the guide were implemented (rule-based, search/
optimisation, ML, NLP) so every kind of AI reasoning taught in the lab is
represented and comparable in one app. Main limitations: skill/education
extraction is keyword/regex-based, not full resume-parsing NLP; the small
sample size limits how much the trained ML models can generalise; and the
chatbot is template-based over real results rather than a free-form LLM
(no external API is used anywhere in this app, so there's nothing to
disclose under the guide's external-API note). See `report.md` Section 5
for the full discussion, including future improvements.
