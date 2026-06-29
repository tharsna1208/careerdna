from flask import Flask, request, jsonify, render_template
import numpy as np
import json, re, os, pickle

app = Flask(__name__)

# ── Load TensorFlow ANN ───────────────────────────────────────
MODEL  = None
SCALER = None
META   = None

def load_model():
    global MODEL, SCALER, META
    try:
        import tensorflow as tf

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(BASE_DIR, "model")

        print("BASE_DIR:", BASE_DIR)
        print("MODEL_DIR:", MODEL_DIR)
        print("FILES:", os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else "MODEL DIR NOT FOUND")

        MODEL = tf.keras.models.load_model(os.path.join(MODEL_DIR, "career_model"))

        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            SCALER = pickle.load(f)

        with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
            META = json.load(f)

        print(f"[OK] ANN loaded — {META['total_params']:,} params, {META['test_accuracy']}% accuracy")

    except Exception as e:
        print("MODEL LOAD ERROR:", e)

load_model()

# ══════════════════════════════════════════════════════════════
# CORE MATH FUNCTIONS 
# ══════════════════════════════════════════════════════════════

def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Cosine similarity between two skill sets.
    Same math as Word2Vec / BERT semantic similarity.
    cos(θ) = |A∩B| / (√|A| × √|B|)
    """
    a = set(s.lower() for s in vec_a)
    b = set(s.lower() for s in vec_b)
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    magnitude = (len(a) ** 0.5) * (len(b) ** 0.5)
    return round(intersection / magnitude, 4) if magnitude > 0 else 0.0

def relu(x: float) -> float:
    return max(0.0, float(x))

def softmax(scores: list) -> list:
    """Softmax: converts raw ANN scores to probability distribution."""
    import math
    max_s = max(scores) if scores else 0
    exps = [math.exp((s - max_s) * 3) for s in scores]
    total = sum(exps)
    return [e / total for e in exps] if total > 0 else [1/len(scores)] * len(scores)

def tfidf_weight(term: str, doc_terms: list, corpus_freq: dict) -> float:
    """
    TF-IDF weight for a term in a document.
    TF = occurrences / total_terms
    IDF = log(total_docs / docs_containing_term)
    """
    import math
    tf = doc_terms.count(term) / max(len(doc_terms), 1)
    idf = math.log(100 / max(corpus_freq.get(term, 1), 1)) + 1
    return tf * idf

def tokenize(text: str) -> set:
    """Extract words + bigrams from text for matching."""
    t = text.lower()
    words = re.findall(r'\b[\w+#\-\.]+\b', t)
    bigrams = {words[i] + ' ' + words[i+1] for i in range(len(words)-1)}
    return set(words) | bigrams


# ══════════════════════════════════════════════════════════════
# CAREER DATABASE (role definitions — not results, just labels)
# ══════════════════════════════════════════════════════════════

CAREER_DB = {
    "AI/ML Engineer":       {"must":["python","machine learning","deep learning","tensorflow","pytorch","numpy","pandas"],"good":["nlp","computer vision","mlops","docker","fastapi","huggingface","cuda"],"salary":"₹8–28 LPA","growth":"Very High","icon":"🤖","domains":["ai","ml","research"]},
    "Data Scientist":       {"must":["python","statistics","sql","pandas","scikit-learn","data analysis"],"good":["r","tableau","power bi","machine learning","spark","a/b testing","feature engineering"],"salary":"₹7–24 LPA","growth":"High","icon":"📊","domains":["data","analysis"]},
    "Full Stack Developer": {"must":["javascript","html","css","react","node","sql","git"],"good":["typescript","mongodb","docker","aws","next.js","graphql","tailwind"],"salary":"₹6–22 LPA","growth":"High","icon":"🌐","domains":["web","frontend","backend"]},
    "Backend Engineer":     {"must":["python","java","sql","api","git","linux"],"good":["docker","kubernetes","redis","microservices","aws","postgresql","kafka"],"salary":"₹6–20 LPA","growth":"Steady","icon":"⚙️","domains":["backend","systems"]},
    "DevOps Engineer":      {"must":["linux","docker","kubernetes","ci/cd","git","aws","bash"],"good":["terraform","ansible","prometheus","jenkins","helm","azure","python"],"salary":"₹8–26 LPA","growth":"Very High","icon":"🔧","domains":["cloud","devops"]},
    "Cloud Architect":      {"must":["aws","cloud","linux","networking","terraform"],"good":["azure","gcp","kubernetes","docker","python","cost optimisation"],"salary":"₹12–35 LPA","growth":"Very High","icon":"☁️","domains":["cloud","architecture"]},
    "Cybersecurity Analyst":{"must":["networking","linux","security","python"],"good":["penetration testing","kali","wireshark","cryptography","siem","owasp"],"salary":"₹6–22 LPA","growth":"High","icon":"🔒","domains":["security"]},
    "Frontend Developer":   {"must":["javascript","html","css","react","git"],"good":["typescript","vue","tailwind","figma","next.js","accessibility"],"salary":"₹5–18 LPA","growth":"Steady","icon":"🎨","domains":["web","frontend","design"]},
    "MLOps Engineer":       {"must":["python","docker","kubernetes","mlops","machine learning","linux"],"good":["mlflow","airflow","spark","terraform","aws","fastapi","dvc"],"salary":"₹10–30 LPA","growth":"Hottest","icon":"🛠️","domains":["ml","devops","cloud"]},
    "Data Engineer":        {"must":["python","sql","spark","etl","airflow"],"good":["kafka","aws","dbt","snowflake","bigquery","scala","docker"],"salary":"₹7–24 LPA","growth":"Very High","icon":"🗄️","domains":["data","engineering"]},
    "NLP Engineer":         {"must":["python","nlp","deep learning","pytorch","transformers"],"good":["bert","gpt","huggingface","spacy","nltk","text classification"],"salary":"₹10–32 LPA","growth":"Hottest","icon":"💬","domains":["nlp","ai","research"]},
    "Mobile Developer":     {"must":["kotlin","swift","react native","java","git"],"good":["flutter","dart","firebase","ui/ux","rest api","android","ios"],"salary":"₹5–18 LPA","growth":"Steady","icon":"📱","domains":["mobile"]},
    "Research Scientist":   {"must":["python","deep learning","mathematics","statistics","pytorch"],"good":["nlp","computer vision","paper writing","cuda","huggingface","c++"],"salary":"₹10–40 LPA","growth":"High","icon":"🔬","domains":["research","ai","ml"]},
    "Product Manager":      {"must":["product","agile","scrum","stakeholder","user research"],"good":["sql","analytics","figma","jira","a/b testing","roadmap","gtm"],"salary":"₹8–30 LPA","growth":"High","icon":"📋","domains":["product","business"]},
}

# ── Skill market data (2025 job market signals) ───────────────
SKILL_MARKET = {
    "python":         {"trend":"rising",  "velocity":9.2, "note":"#1 demanded language 2025"},
    "llm":            {"trend":"rising",  "velocity":9.8, "note":"Hottest skill of 2025"},
    "mlops":          {"trend":"rising",  "velocity":9.5, "note":"Huge demand gap — few engineers"},
    "kubernetes":     {"trend":"rising",  "velocity":8.9, "note":"Cloud-native essential"},
    "terraform":      {"trend":"rising",  "velocity":8.7, "note":"IaC becoming mandatory"},
    "rust":           {"trend":"rising",  "velocity":8.5, "note":"Replacing C++ in systems"},
    "typescript":     {"trend":"rising",  "velocity":8.4, "note":"React+TS is now standard"},
    "next.js":        {"trend":"rising",  "velocity":8.2, "note":"Full-stack React standard"},
    "fastapi":        {"trend":"rising",  "velocity":8.0, "note":"Replacing Flask for new APIs"},
    "docker":         {"trend":"rising",  "velocity":8.0, "note":"Non-negotiable in 2025"},
    "pytorch":        {"trend":"rising",  "velocity":8.8, "note":"Preferred over TF for research"},
    "huggingface":    {"trend":"rising",  "velocity":9.0, "note":"Gateway to LLM ecosystem"},
    "golang":         {"trend":"rising",  "velocity":7.8, "note":"High pay, low supply"},
    "aws":            {"trend":"rising",  "velocity":8.1, "note":"Most certified = most hired"},
    "cybersecurity":  {"trend":"rising",  "velocity":8.6, "note":"3.5M unfilled jobs globally"},
    "machine learning":{"trend":"rising", "velocity":8.5, "note":"Core skill across all industries"},
    "deep learning":  {"trend":"rising",  "velocity":8.3, "note":"Research + product roles"},
    "react":          {"trend":"rising",  "velocity":7.2, "note":"Dominant frontend framework"},
    "sql":            {"trend":"rising",  "velocity":7.0, "note":"Evergreen — always in demand"},
    "spark":          {"trend":"rising",  "velocity":7.5, "note":"Big data still growing"},
    "java":           {"trend":"stable",  "velocity":5.0, "note":"Enterprise still relies on it"},
    "javascript":     {"trend":"stable",  "velocity":6.5, "note":"Web's lingua franca"},
    "linux":          {"trend":"stable",  "velocity":6.0, "note":"Fundamental, always needed"},
    "git":            {"trend":"stable",  "velocity":6.0, "note":"Table stakes — not differentiating"},
    "html":           {"trend":"stable",  "velocity":5.0, "note":"Necessary but not hireable alone"},
    "css":            {"trend":"stable",  "velocity":5.0, "note":"Need frameworks on top"},
    "node":           {"trend":"stable",  "velocity":5.5, "note":"Holding steady"},
    "c++":            {"trend":"stable",  "velocity":5.5, "note":"Games + embedded + HPC"},
    "flutter":        {"trend":"stable",  "velocity":5.5, "note":"Strong mobile niche"},
    "kotlin":         {"trend":"stable",  "velocity":5.5, "note":"Android standard"},
    "php":            {"trend":"declining","velocity":2.5, "note":"Legacy web — avoid for new projects"},
    "jquery":         {"trend":"declining","velocity":1.5, "note":"Effectively dead for new work"},
    "perl":           {"trend":"declining","velocity":1.5, "note":"Almost no new demand"},
    "cobol":          {"trend":"declining","velocity":1.0, "note":"Legacy banking maintenance only"},
    "svn":            {"trend":"declining","velocity":1.5, "note":"Git replaced it entirely"},
    "soap":           {"trend":"declining","velocity":1.8, "note":"REST/GraphQL dominates"},
    "hadoop":         {"trend":"declining","velocity":2.5, "note":"Spark + cloud storage replacing it"},
    "bootstrap":      {"trend":"declining","velocity":2.8, "note":"Tailwind CSS winning the market"},
    "redux":          {"trend":"declining","velocity":3.0, "note":"React Query + Zustand winning"},
    "webpack":        {"trend":"declining","velocity":3.0, "note":"Vite replacing it fast"},
    "visual basic":   {"trend":"declining","velocity":1.2, "note":"Microsoft legacy only"},
    "objective-c":    {"trend":"declining","velocity":1.2, "note":"Swift replaced it"},
}

# ── YouTube resources (real URLs) ────────────────────────────
YOUTUBE = {
    "python":           [{"title":"Python Full Course","url":"https://www.youtube.com/watch?v=rfscVS0vtbw","channel":"freeCodeCamp"},{"title":"Python in 100 Seconds","url":"https://www.youtube.com/watch?v=x7X9w_GIm1s","channel":"Fireship"}],
    "machine learning": [{"title":"ML Course — Andrew Ng","url":"https://www.youtube.com/watch?v=jGwO_UgTS7I","channel":"Stanford Online"},{"title":"ML Zero to Hero","url":"https://www.youtube.com/watch?v=VwVg9jCtqaU","channel":"Google Developers"}],
    "deep learning":    [{"title":"Deep Learning Specialization","url":"https://www.youtube.com/watch?v=CS4cs9xVecg","channel":"DeepLearning.AI"},{"title":"Neural Networks from Scratch","url":"https://www.youtube.com/watch?v=Wo5dMEP_BbI","channel":"Sentdex"}],
    "tensorflow":       [{"title":"TensorFlow 2.0 Complete Course","url":"https://www.youtube.com/watch?v=tPYj3fFJGjk","channel":"freeCodeCamp"}],
    "pytorch":          [{"title":"PyTorch for Deep Learning","url":"https://www.youtube.com/watch?v=V_xro1bcAuA","channel":"freeCodeCamp"},{"title":"PyTorch in 100 Seconds","url":"https://www.youtube.com/watch?v=ORMx45xqWkA","channel":"Fireship"}],
    "docker":           [{"title":"Docker Tutorial for Beginners","url":"https://www.youtube.com/watch?v=3c-iBn73dDE","channel":"TechWorld with Nana"},{"title":"Docker in 100 Seconds","url":"https://www.youtube.com/watch?v=Gjnup-PuquQ","channel":"Fireship"}],
    "kubernetes":       [{"title":"Kubernetes Full Course","url":"https://www.youtube.com/watch?v=X48VuDVv0do","channel":"TechWorld with Nana"},{"title":"Kubernetes in 100 Seconds","url":"https://www.youtube.com/watch?v=PziYflu8cB8","channel":"Fireship"}],
    "react":            [{"title":"React Full Course 2024","url":"https://www.youtube.com/watch?v=bMknfKXIFA8","channel":"freeCodeCamp"},{"title":"React in 100 Seconds","url":"https://www.youtube.com/watch?v=Tn6-PIqc4UM","channel":"Fireship"}],
    "sql":              [{"title":"SQL Full Course","url":"https://www.youtube.com/watch?v=HXV3zeQKqGY","channel":"freeCodeCamp"},{"title":"SQL in 100 Seconds","url":"https://www.youtube.com/watch?v=zsjvFFKOm3c","channel":"Fireship"}],
    "aws":              [{"title":"AWS Certified Cloud Practitioner","url":"https://www.youtube.com/watch?v=SOTamWNgDKc","channel":"freeCodeCamp"},{"title":"AWS in 10 Minutes","url":"https://www.youtube.com/watch?v=a9__D53WsMs","channel":"Fireship"}],
    "git":              [{"title":"Git and GitHub Full Course","url":"https://www.youtube.com/watch?v=RGOj5yH7evk","channel":"freeCodeCamp"},{"title":"Git in 100 Seconds","url":"https://www.youtube.com/watch?v=hwP7WQkmECE","channel":"Fireship"}],
    "linux":            [{"title":"Linux Command Line Full Course","url":"https://www.youtube.com/watch?v=iwolPf6kN-k","channel":"freeCodeCamp"},{"title":"Linux for Hackers","url":"https://www.youtube.com/watch?v=VbEx7B_PTOE","channel":"NetworkChuck"}],
    "javascript":       [{"title":"JavaScript Full Course","url":"https://www.youtube.com/watch?v=PkZNo7MFNFg","channel":"freeCodeCamp"},{"title":"JS in 100 Seconds","url":"https://www.youtube.com/watch?v=DHjqpvDnNGE","channel":"Fireship"}],
    "typescript":       [{"title":"TypeScript Full Course","url":"https://www.youtube.com/watch?v=30LWjhZzg50","channel":"freeCodeCamp"},{"title":"TypeScript in 100 Seconds","url":"https://www.youtube.com/watch?v=zQnBQ4tB3ZA","channel":"Fireship"}],
    "nlp":              [{"title":"NLP with Python","url":"https://www.youtube.com/watch?v=X2vAabgKiuM","channel":"freeCodeCamp"},{"title":"HuggingFace NLP Course","url":"https://www.youtube.com/watch?v=00GKzGyWFEs","channel":"HuggingFace"}],
    "statistics":       [{"title":"Statistics — StatQuest","url":"https://www.youtube.com/watch?v=qBigTkBLU6g","channel":"StatQuest"},{"title":"Statistics Full Course","url":"https://www.youtube.com/watch?v=OyddY7DlV58","channel":"freeCodeCamp"}],
    "cybersecurity":    [{"title":"Ethical Hacking Full Course","url":"https://www.youtube.com/watch?v=3Kq1MIfTWCE","channel":"freeCodeCamp"},{"title":"How to get into Cybersecurity","url":"https://www.youtube.com/watch?v=a83ASGn_V_s","channel":"NetworkChuck"}],
    "terraform":        [{"title":"Terraform Full Course","url":"https://www.youtube.com/watch?v=SLB_c_ayRMo","channel":"freeCodeCamp"}],
    "mlops":            [{"title":"MLOps Zoomcamp","url":"https://www.youtube.com/watch?v=s0uaFZSzwfI","channel":"DataTalks.Club"},{"title":"MLOps Full Course","url":"https://www.youtube.com/watch?v=oixRe8JnLtQ","channel":"freeCodeCamp"}],
    "rust":             [{"title":"Rust Full Course","url":"https://www.youtube.com/watch?v=MsocPEZBd-M","channel":"freeCodeCamp"},{"title":"Rust in 100 Seconds","url":"https://www.youtube.com/watch?v=5C_HPTJg1lc","channel":"Fireship"}],
    "golang":           [{"title":"Go Full Course","url":"https://www.youtube.com/watch?v=un6ZyFkqFKo","channel":"freeCodeCamp"},{"title":"Go in 100 Seconds","url":"https://www.youtube.com/watch?v=446E-r0rXHI","channel":"Fireship"}],
    "html":             [{"title":"HTML Full Course","url":"https://www.youtube.com/watch?v=kUMe1FH4CHE","channel":"freeCodeCamp"}],
    "css":              [{"title":"CSS Full Course","url":"https://www.youtube.com/watch?v=OXGznpKZ_sA","channel":"freeCodeCamp"}],
    "next.js":          [{"title":"Next.js Full Course","url":"https://www.youtube.com/watch?v=KjY94sAKLlw","channel":"freeCodeCamp"}],
    "java":             [{"title":"Java Full Course","url":"https://www.youtube.com/watch?v=GoXwIVyNvX0","channel":"freeCodeCamp"}],
}

FREE_COURSES = {
    "AI/ML Engineer":       [{"title":"Fast.ai — Practical Deep Learning (Free)","url":"https://course.fast.ai"},{"title":"Google ML Crash Course","url":"https://developers.google.com/machine-learning/crash-course"},{"title":"Kaggle Learn — Free ML","url":"https://www.kaggle.com/learn"},{"title":"CS229 Stanford — Free","url":"https://cs229.stanford.edu"}],
    "Data Scientist":       [{"title":"Kaggle Learn — Python, Pandas, ML","url":"https://www.kaggle.com/learn"},{"title":"Google Data Analytics Certificate","url":"https://grow.google/certificates/data-analytics"},{"title":"Mode SQL Tutorial","url":"https://mode.com/sql-tutorial"}],
    "Full Stack Developer": [{"title":"The Odin Project — Full Stack Free","url":"https://www.theodinproject.com"},{"title":"freeCodeCamp Full Stack","url":"https://www.freecodecamp.org"},{"title":"Full Stack Open — Helsinki Uni","url":"https://fullstackopen.com"}],
    "DevOps Engineer":      [{"title":"KodeKloud — Free K8s Playground","url":"https://kodekloud.com"},{"title":"AWS Free Tier + Labs","url":"https://aws.amazon.com/free"},{"title":"Play with Docker Lab","url":"https://labs.play-with-docker.com"}],
    "Cybersecurity Analyst":[{"title":"TryHackMe — Free Learning Paths","url":"https://tryhackme.com"},{"title":"HackTheBox Academy","url":"https://academy.hackthebox.com"},{"title":"OWASP WebGoat (Free Labs)","url":"https://owasp.org/www-project-webgoat"}],
    "MLOps Engineer":       [{"title":"MLOps Zoomcamp — DataTalks.Club","url":"https://github.com/DataTalksClub/mlops-zoomcamp"},{"title":"Made With ML — MLOps Guide","url":"https://madewithml.com"},{"title":"Full Stack Deep Learning","url":"https://fullstackdeeplearning.com"}],
}
DEFAULT_COURSES = [
    {"title":"freeCodeCamp — 1000+ hours free","url":"https://www.freecodecamp.org"},
    {"title":"Kaggle Learn — Free hands-on ML","url":"https://www.kaggle.com/learn"},
    {"title":"The Odin Project — Full Stack","url":"https://www.theodinproject.com"},
    {"title":"CS50 Harvard — Free","url":"https://cs50.harvard.edu"},
    {"title":"MIT OpenCourseWare","url":"https://ocw.mit.edu"},
]

def get_resources(skills: list, role: str = None) -> dict:
    videos, seen = [], set()
    for skill in skills:
        for v in YOUTUBE.get(skill.lower(), []):
            if v["url"] not in seen:
                videos.append({**v, "for_skill": skill})
                seen.add(v["url"])
        if len(videos) >= 8:
            break
    courses = FREE_COURSES.get(role, DEFAULT_COURSES)[:5]
    return {"youtube": videos[:8], "courses": courses}


# ══════════════════════════════════════════════════════════════
# ROUTE 1: CAREER PATH PREDICTION
# ══════════════════════════════════════════════════════════════
@app.route('/api/career', methods=['POST'])
def career():
    if MODEL is None or SCALER is None:
        return jsonify({"error": "ANN model not loaded. Run: python train_model.py"}), 503

    data       = request.get_json()
    skills     = [s.lower().strip() for s in data.get('skills', []) if s.strip()]
    exp_level  = int(data.get('exp_level', 0))
    work_style = data.get('work_style', 'builder')
    interests  = [i.lower() for i in data.get('interests', [])]

    if not skills:
        return jsonify({"error": "Please add at least one skill."}), 400

    # Build skill feature vector (30 skills from training)
    skill_index = META.get('skill_index', {})
    n_features  = META.get('n_features', 30)
    x = np.zeros(n_features, dtype=np.float32)
    for skill in skills:
        if skill in skill_index:
            x[skill_index[skill]] = 1.0
        # partial match
        for trained_skill, idx in skill_index.items():
            if skill in trained_skill or trained_skill in skill:
                x[idx] = max(x[idx], 0.7)

    # Scale and predict via TensorFlow ANN
    x_scaled   = SCALER.transform(x.reshape(1, -1))
    raw_probs   = MODEL.predict(x_scaled, verbose=0)[0]  # softmax probabilities

    career_names = META.get('career_names', list(CAREER_DB.keys()))

    # Build result — combine ANN probability with cosine similarity signal
    results = []
    for i, name in enumerate(career_names):
        if name not in CAREER_DB:
            continue
        career_info  = CAREER_DB[name]
        ann_prob     = float(raw_probs[i]) if i < len(raw_probs) else 0.0
        cos_sim      = cosine_similarity(skills, career_info['must'] + career_info['good'])

        # Hybrid score: ANN (primary) + cosine (secondary signal)
        hybrid = relu(ann_prob * 0.65 + cos_sim * 0.35)

        # Domain interest boost
        domain_boost = 0.0
        for interest in interests:
            if any(interest in d for d in career_info.get('domains', [])):
                domain_boost = 0.12
                break

        # Work style boost
        style_map = {
            "builder":    ["Full Stack","Backend","Mobile","DevOps","MLOps"],
            "analyst":    ["Data Scientist","NLP","Research"],
            "researcher": ["Research","NLP","AI/ML"],
            "leader":     ["Product","Cloud","DevOps"]
        }
        style_boost = 0.08 if any(s in name for s in style_map.get(work_style, [])) else 0.0

        final_score = relu(hybrid + domain_boost + style_boost)

        have    = [k for k in career_info['must'] if k in skills]
        missing = [k for k in career_info['must'] if k not in skills]

        results.append({
            "name":       name,
            "score":      final_score,
            "ann_prob":   round(ann_prob * 100, 1),
            "cos_sim":    round(cos_sim * 100, 1),
            "icon":       career_info['icon'],
            "salary":     career_info['salary'],
            "growth":     career_info['growth'],
            "have":       have,
            "missing":    missing[:5],
        })

    results.sort(key=lambda r: r['score'], reverse=True)
    top5 = results[:5]

    # Apply softmax to scores → final match percentages
    raw_scores = [r['score'] for r in top5]
    probs      = softmax(raw_scores)
    for i, r in enumerate(top5):
        r['match_pct'] = min(97, max(38, round(probs[i] * 260 + 38 - i * 7)))
        del r['score']

    top_role  = top5[0]['name'] if top5 else None
    missing_0 = top5[0].get('missing', []) if top5 else []

    return jsonify({
        "careers":   top5,
        "resources": get_resources(missing_0[:4] + skills[:2], top_role),
        "ann_used":  True,
        "model_accuracy": META.get('test_accuracy', '?')
    })


# ══════════════════════════════════════════════════════════════
# ROUTE 2: SKILL DECAY ANALYSIS
# ══════════════════════════════════════════════════════════════
@app.route('/api/decay', methods=['POST'])
def decay():
    data     = request.get_json()
    skills   = [s.lower().strip() for s in data.get('skills', []) if s.strip()]
    industry = data.get('industry', 'Software / Tech')

    if not skills:
        return jsonify({"error": "Please add at least one skill."}), 400

    results = []
    for skill in skills:
        key  = skill.lower()
        info = SKILL_MARKET.get(key)

        if info:
            # ReLU normalise velocity → demand score
            demand_score = round(relu((info['velocity'] / 10.0)) * 100)
            trend, note  = info['trend'], info['note']
        else:
            # Fuzzy match
            matched = None
            for k, v in SKILL_MARKET.items():
                if key in k or k in key:
                    matched = (k, v)
                    break
            if matched:
                demand_score = round(relu(matched[1]['velocity'] / 10.0) * 100)
                trend = matched[1]['trend']
                note  = matched[1]['note'] + f" (matched as '{matched[0]}')"
            else:
                demand_score = 50
                trend = "stable"
                note  = "No market data — research on LinkedIn Jobs"

        results.append({
            "skill":        skill,
            "trend":        trend,
            "demand_score": demand_score,
            "note":         note,
        })

    results.sort(key=lambda x: x['demand_score'], reverse=True)

    # Recommend skills based on industry gaps
    industry_recs = {
        "Software / Tech":    ["python","kubernetes","rust","golang","mlops"],
        "Data Science / AI":  ["pytorch","huggingface","mlops","spark","fastapi"],
        "Web Development":    ["typescript","next.js","react","fastapi","docker"],
        "Cloud / DevOps":     ["terraform","kubernetes","aws","docker","golang"],
        "Cybersecurity":      ["cybersecurity","python","kubernetes","aws","linux"],
    }
    recs = [r for r in industry_recs.get(industry, ["python","docker","kubernetes"])
            if r not in [s['skill'] for s in results]][:5]

    return jsonify({
        "results":         results,
        "recommendations": recs,
        "resources":       get_resources(recs[:3]),
        "summary": {
            "rising":    sum(1 for r in results if r['trend'] == 'rising'),
            "stable":    sum(1 for r in results if r['trend'] == 'stable'),
            "declining": sum(1 for r in results if r['trend'] == 'declining'),
        }
    })


# ══════════════════════════════════════════════════════════════
# ROUTE 3: INTERVIEW IQ SCORING
# ══════════════════════════════════════════════════════════════
@app.route('/api/interview', methods=['POST'])
def interview():
    data   = request.get_json()
    jd     = data.get('jd', '').strip()
    resume = data.get('resume', '').strip()

    if len(jd) < 40:
        return jsonify({"error": "Please paste a full job description (at least 40 characters)."}), 400
    if len(resume) < 30:
        return jsonify({"error": "Please paste your resume or skills."}), 400

    jd_lower  = jd.lower()
    res_lower = resume.lower()

    # Topic taxonomy with keywords
    TOPICS = {
        "Programming Languages": {"kws": ["python","java","javascript","c++","golang","rust","typescript","kotlin","swift","php","ruby","scala"], "weight": 0.9},
        "Web Frameworks":        {"kws": ["react","angular","vue","django","flask","fastapi","spring","express","node","next.js","laravel","rails"], "weight": 0.8},
        "Databases":             {"kws": ["sql","mysql","postgresql","mongodb","redis","elasticsearch","cassandra","dynamodb","oracle","sqlite"], "weight": 0.7},
        "Cloud & DevOps":        {"kws": ["aws","azure","gcp","docker","kubernetes","terraform","ci/cd","linux","jenkins","ansible","helm"], "weight": 0.8},
        "ML / AI":               {"kws": ["machine learning","deep learning","tensorflow","pytorch","nlp","computer vision","scikit-learn","mlops","neural network","llm"], "weight": 0.9},
        "Data & Analytics":      {"kws": ["pandas","numpy","spark","hadoop","tableau","power bi","statistics","data analysis","bigquery","airflow"], "weight": 0.7},
        "System Design":         {"kws": ["microservices","rest api","graphql","system design","scalability","load balancing","caching","kafka","grpc","rabbitmq"], "weight": 0.8},
        "Soft Skills & Process": {"kws": ["agile","scrum","communication","teamwork","jira","confluence","leadership","problem solving"], "weight": 0.5},
        "Security":              {"kws": ["security","owasp","penetration","cryptography","oauth","jwt","ssl","vulnerability","firewall"], "weight": 0.7},
        "Testing & Quality":     {"kws": ["testing","unit test","integration test","jest","pytest","selenium","test driven","qa","ci"], "weight": 0.6},
    }

    # Collect all terms for IDF estimation
    all_terms = [kw for td in TOPICS.values() for kw in td['kws']]
    corpus_freq = {t: sum(1 for td in TOPICS.values() if t in td['kws']) for t in all_terms}

    topics_scored = []
    for topic_name, td in TOPICS.items():
        jd_hits  = [k for k in td['kws'] if k in jd_lower]
        if not jd_hits:
            continue

        res_hits = [k for k in td['kws'] if k in res_lower]
        missing  = [k for k in jd_hits if k not in res_lower]

        # TF-IDF weighted cosine similarity
        jd_words  = jd_lower.split()
        jd_vec    = [tfidf_weight(k, jd_words, corpus_freq) if k in jd_lower else 0.0 for k in td['kws']]
        res_vec   = [tfidf_weight(k, res_lower.split(), corpus_freq) if k in res_lower else 0.0 for k in td['kws']]

        dot       = sum(a * b for a, b in zip(jd_vec, res_vec))
        mag_jd    = sum(v**2 for v in jd_vec) ** 0.5
        mag_res   = sum(v**2 for v in res_vec) ** 0.5
        sim       = dot / (mag_jd * mag_res) if mag_jd > 0 and mag_res > 0 else 0.0
        readiness = round(relu(sim) * 100)

        topics_scored.append({
            "topic":     topic_name,
            "readiness": readiness,
            "jd_kws":    jd_hits,
            "matched":   res_hits,
            "missing":   missing,
            "weight":    td['weight'],
        })

    topics_scored.sort(key=lambda t: t['weight'] * len(t['jd_kws']), reverse=True)

    # Weighted overall score
    total_w = sum(t['weight'] * len(t['jd_kws']) for t in topics_scored)
    overall = round(sum(t['readiness'] * t['weight'] * len(t['jd_kws']) for t in topics_scored) / max(total_w, 1))

    # Small bonus for experience/projects
    if re.search(r'internship|experience|project|worked|built|developed', res_lower):
        overall = min(97, overall + 5)

    all_missing = list({m for t in topics_scored for m in t['missing']})

    return jsonify({
        "overall":    overall,
        "topics":     topics_scored[:9],
        "revise":     all_missing[:12],
        "resources":  get_resources(all_missing[:5]),
        "verdict":    "Interview Ready ✅" if overall >= 80 else "Mostly Ready ⚠️" if overall >= 60 else "Needs Revision 🟠" if overall >= 40 else "Major Gaps ❌"
    })


# ══════════════════════════════════════════════════════════════
# ROUTE 4: ATS RESUME CHECKER
# ══════════════════════════════════════════════════════════════

ATS_ROLE_DB = {
    "Software Engineer":{"must":["python","java","javascript","c++","sql","git","algorithms","data structures","oop"],"good":["docker","kubernetes","aws","react","node","rest api","microservices","ci/cd","linux","agile","scrum"],"tools":["git","github","jira","postman","docker","vs code"],"soft":["teamwork","communication","problem solving","leadership"],"edu":["b.tech","be","bsc","computer science","cse","it","mca","bca"],"flags":["responsible for","helped with","familiar with","worked on"]},
    "Data Scientist":   {"must":["python","machine learning","statistics","sql","pandas","scikit-learn"],"good":["tensorflow","pytorch","nlp","tableau","power bi","spark","r","a/b testing","feature engineering"],"tools":["jupyter","git","matplotlib","seaborn","excel","colab"],"soft":["analytical","research","communication","curious"],"edu":["b.tech","be","bsc","msc","data science","statistics","mathematics","cse"],"flags":["responsible for","familiar with","basic ml"]},
    "AI/ML Engineer":   {"must":["python","machine learning","deep learning","neural network","tensorflow","pytorch"],"good":["nlp","computer vision","cnn","rnn","transformer","bert","docker","mlops","fastapi"],"tools":["jupyter","git","colab","wandb","anaconda"],"soft":["research","analytical","innovative","self-motivated"],"edu":["b.tech","be","mtech","artificial intelligence","machine learning","computer science","cse"],"flags":["basic ml","beginner","familiar with"]},
    "Web Developer":    {"must":["html","css","javascript","git"],"good":["react","angular","vue","node","mongodb","sql","typescript","tailwind","docker","aws","next.js"],"tools":["git","vs code","figma","npm","postman","chrome devtools"],"soft":["creative","detail-oriented","teamwork","communication"],"edu":["b.tech","be","bsc","computer science","cse","it","mca"],"flags":["knows html","basic knowledge","introductory"]},
    "DevOps Engineer":  {"must":["linux","docker","kubernetes","ci/cd","git","aws","bash"],"good":["terraform","ansible","prometheus","grafana","azure","python","nginx","redis","elk"],"tools":["docker","kubernetes","jenkins","terraform","helm","github actions"],"soft":["problem solving","collaboration","proactive","attention to detail"],"edu":["b.tech","be","computer science","cse","it"],"flags":["basic linux","familiar with aws","knows docker"]},
}

STRONG_VERBS = {"developed","built","implemented","designed","created","deployed","architected","engineered","automated","optimised","improved","reduced","increased","delivered","launched","led","managed","analysed","researched","published","trained","evaluated","integrated","migrated","scaled","secured","debugged","refactored","tested","maintained","achieved","generated","saved","modelled","predicted","fine-tuned","benchmarked","coordinated","streamlined"}
FILLER_WORDS = ["hardworking","passionate","go-getter","think outside the box","team player","results-driven","self-starter","synergy","dynamic","motivated individual","proven track record","responsible for","go getter","detail oriented"]
ATS_SECTIONS = {
    "contact":    ["email","phone","linkedin","github","@","mobile","contact"],
    "education":  ["education","b.tech","be","bsc","mtech","msc","degree","university","college","cgpa","gpa"],
    "experience": ["experience","internship","work experience","company","worked at","employment"],
    "skills":     ["skills","technical skills","technologies","tools","stack","expertise","proficiencies"],
    "projects":   ["project","projects","built","developed a","implemented a","created a"],
}
QUANT_RE = [r'\d+\s*%',r'\d+\s*x\b',r'\$[\d,]+',r'₹[\d,]+',r'\d+\s*(users|clients|requests|records|models|projects)',r'cgpa\s*[:\-]?\s*[\d.]+',r'\d+/\d+',r'(improved|reduced|increased|saved)\s+by\s+\d+',r'\d+\+?\s*(years|yrs|months)',r'(top|rank|first)\s+\d+']

@app.route('/api/ats', methods=['POST'])
def ats():
    data   = request.get_json()
    resume = data.get('resume', '').strip()
    role   = data.get('role', 'Software Engineer')

    if len(resume) < 80:
        return jsonify({"error": "Resume too short. Please paste your full resume text."}), 400
    if role not in ATS_ROLE_DB:
        role = "Software Engineer"

    db  = ATS_ROLE_DB[role]
    rw  = tokenize(resume)
    tl  = resume.lower()

    # 1. Keyword scoring
    must_m  = [k for k in db["must"]  if k in rw or k in tl]
    must_mi = [k for k in db["must"]  if k not in rw and k not in tl]
    good_m  = [k for k in db["good"]  if k in rw or k in tl]
    good_mi = [k for k in db["good"]  if k not in rw and k not in tl][:10]
    tools_m = [k for k in db["tools"] if k in rw or k in tl]
    soft_m  = [k for k in db["soft"]  if k in rw or k in tl]
    ms = len(must_m)/len(db["must"])*100 if db["must"] else 0
    gs = len(good_m)/max(len(db["good"]),1)*100
    ts = len(tools_m)/max(len(db["tools"]),1)*100
    ss = len(soft_m)/max(len(db["soft"]),1)*100
    kw_score = min(100, ms*0.50 + gs*0.25 + ts*0.15 + ss*0.10)

    # 2. Section detection
    sec_found = [s for s, triggers in ATS_SECTIONS.items() if any(t in tl for t in triggers)]
    sec_miss  = [s for s in ATS_SECTIONS if s not in sec_found]
    sec_score = len(sec_found)/len(ATS_SECTIONS)*100

    # 3. Quantification
    q_all = []
    for pat in QUANT_RE:
        found = re.findall(pat, tl)
        q_all.extend([str(f) if not isinstance(f, str) else f for f in found])
    q_count = len(set(q_all))
    q_score = min(100, q_count * 14)

    # 4. Action verbs
    wset    = set(re.findall(r'\b\w+\b', tl))
    verbs_m = [v for v in STRONG_VERBS if v in wset]
    verbs_mi= [v for v in list(STRONG_VERBS)[:20] if v not in wset][:8]
    v_score = min(100, len(verbs_m) * 9)

    # 5. Contact info
    con = {
        "email":    bool(re.search(r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}', tl)),
        "phone":    bool(re.search(r'(\+91)?[\s\-]?[6-9]\d{9}|\d{10}', resume)),
        "linkedin": "linkedin" in tl,
        "github":   "github" in tl or "gitlab" in tl,
        "location": bool(re.search(r'\b(chennai|mumbai|delhi|bangalore|bengaluru|hyderabad|pune|india|tamil nadu|city|state)\b', tl)),
    }
    con_found = [k for k, v in con.items() if v]
    con_miss  = [k for k, v in con.items() if not v]
    con_score = len(con_found)/len(con)*100

    # 6. Education
    edu_m    = [k for k in db["edu"] if k in tl]
    has_cgpa = bool(re.search(r'(cgpa|gpa|percentage)[:\s]*[\d.]+', tl))
    has_year = bool(re.search(r'20\d\d', resume))
    edu_score= min(100, len(edu_m)*20 + (15 if has_cgpa else 0) + (10 if has_year else 0))

    # 7. Filler words
    filler_found = [f for f in FILLER_WORDS if f in tl]
    filler_score = max(0, 100 - len(filler_found)*12)

    # 8. Length
    wc = len(resume.split())
    len_score = 100 if 350<=wc<=900 else (40 if wc<200 else (65 if wc<350 else 70))

    # 9. Red flags
    rf = []
    for flag in db["flags"]:
        if flag in tl: rf.append(f'Weak phrase: "{flag}" — replace with action verb')
    if len(resume.split('\n')) < 8:
        rf.append("Very few lines — resume appears too thin")
    if (resume.count('•') + resume.count('-') + resume.count('*')) < 3:
        rf.append("No bullet points found — use bullets for experience/projects")

    # 10. Final weighted score (ATS formula)
    raw = (kw_score*0.35 + sec_score*0.15 + q_score*0.12 +
           v_score*0.10 + con_score*0.10 + edu_score*0.10 +
           filler_score*0.05 + len_score*0.03)
    overall = round(max(0, min(100, raw - len(rf)*5)), 1)

    if overall >= 80:   verdict, vc = "Excellent — Shortlist Likely ✅", "#16a34a"
    elif overall >= 65: verdict, vc = "Good — May Pass ATS ⚠️",          "#d97706"
    elif overall >= 45: verdict, vc = "Needs Work — May Be Rejected ❌",  "#dc2626"
    else:               verdict, vc = "Very Weak — Will Be Rejected ❌",   "#7f1d1d"

    # Build improvement plan
    improvements = []
    if must_mi:
        improvements.append({"priority":"🔴 Critical","category":"Missing Core Keywords","issue":f"Missing {len(must_mi)} must-have keywords","fix":f"Add to Skills section: {', '.join(must_mi[:4])}","impact":"+15–20 pts"})
    for s in sec_miss:
        improvements.append({"priority":"🔴 Critical","category":f"Missing '{s}' Section","issue":f"ATS cannot detect a {s} section","fix":f"Add a clearly labelled '{s.upper()}' heading","impact":"+8–12 pts"})
    if q_count < 3:
        improvements.append({"priority":"🟠 High","category":"No Quantified Achievements","issue":f"Only {q_count} metric(s) found — recruiters need to see impact","fix":"Add numbers: '40% faster', '500+ users', '94% accuracy', 'CGPA 8.7/10'","impact":"+10–15 pts"})
    if v_score < 50:
        improvements.append({"priority":"🟠 High","category":"Weak Action Verbs","issue":"Experience bullets use passive/weak language","fix":f"Start every bullet with: {', '.join(verbs_mi[:5])}","impact":"+5–8 pts"})
    for m in con_miss:
        desc = {"linkedin":"Add LinkedIn URL — recruiters always check","github":"Add GitHub link — shows real code to recruiters","email":"Add professional email address","phone":"Add phone number","location":"Add city and state"}
        improvements.append({"priority":"🟡 Medium","category":f"Missing: {m.title()}","issue":f"No {m} found in header","fix":desc.get(m, f"Add your {m}"),"impact":"+3–5 pts"})
    if filler_found:
        improvements.append({"priority":"🟡 Medium","category":"Filler/Buzzwords Found","issue":f"Found: {', '.join(filler_found[:3])}","fix":"Delete buzzwords — replace with specific achievements","impact":"+3–5 pts"})
    if good_mi:
        improvements.append({"priority":"🟢 Optional","category":"Additional Keywords","issue":f"{len(good_mi)} relevant keywords missing","fix":f"Add if you know them: {', '.join(good_mi[:5])}","impact":"+5–10 pts"})

    return jsonify({
        "overall":   overall,
        "verdict":   verdict,
        "verdict_color": vc,
        "scores": {
            "Keyword Match":   round(kw_score, 1),
            "Resume Structure":round(sec_score, 1),
            "Quantification":  round(q_score, 1),
            "Action Verbs":    round(v_score, 1),
            "Contact Info":    round(con_score, 1),
            "Education Match": round(edu_score, 1),
            "No Filler Words": round(filler_score, 1),
            "Resume Length":   round(len_score, 1),
        },
        "keywords":{"must_matched":must_m,"must_missing":must_mi,"good_matched":good_m,"good_missing":good_mi,"tools_matched":tools_m},
        "sections": {"found":sec_found,"missing":sec_miss},
        "quantification":{"count":q_count,"examples":list(set(q_all))[:6]},
        "action_verbs":{"matched":verbs_m[:12],"suggestions":verbs_mi},
        "contact":{"checks":con,"missing":con_miss},
        "red_flags": rf,
        "improvements": improvements,
        "stats":{"word_count":wc,"keywords_found":len(must_m)+len(good_m)+len(tools_m),"quant_count":q_count,"verb_count":len(verbs_m),"sections_found":len(sec_found)},
        "resources": get_resources(must_mi[:4]),
    })


# ══════════════════════════════════════════════════════════════
# ROUTE 5: LEARNING ROADMAP
# ══════════════════════════════════════════════════════════════

ROADMAP_DB = {
    "AI/ML Engineer":{"required":["python","mathematics","machine learning","deep learning","tensorflow","pytorch","mlops","docker","sql","statistics"],"weeks":{4:[{"t":"Python + Math Foundations","items":["Python OOP, NumPy, Pandas, Matplotlib","Linear Algebra + Statistics (Khan Academy)","Jupyter + Google Colab setup","Build 2 data analysis projects"]},{"t":"ML Fundamentals","items":["Supervised/Unsupervised learning concepts","Linear + Logistic Regression from scratch","scikit-learn: Iris, Titanic, Boston datasets","Cross-validation + overfitting detection"]}],8:[{"t":"Deep Learning","items":["Neural networks from scratch — no library","TensorFlow/Keras: build ANN, CNN, RNN","Backpropagation derivation + Adam optimizer","Image classification project (CIFAR-10)"]},{"t":"NLP + Advanced DL","items":["Text preprocessing: tokenize, TF-IDF, embeddings","Transformers + BERT basics","HuggingFace Pipelines","Build a sentiment analysis project"]}],12:[{"t":"MLOps + Deployment","items":["FastAPI for ML model serving","Docker: containerise your model","MLflow: experiment tracking","Deploy to AWS EC2 / GCP free tier"]},{"t":"Capstone Project","items":["End-to-end Kaggle project","Technical blog post on Medium","Polish GitHub with README + demo","Apply for ML internships"]}]}},
    "Full Stack Developer":{"required":["html","css","javascript","react","node","sql","git","docker","aws","typescript"],"weeks":{4:[{"t":"Frontend Foundations","items":["HTML5 semantic elements + Accessibility","CSS Flexbox + Grid + Animations","JavaScript ES6+: async/await, fetch, modules","Build 3 responsive static websites"]},{"t":"React","items":["Components, Props, State management","Hooks: useState, useEffect, useContext","React Router v6 + forms handling","Build: Todo App + Weather App with API"]}],8:[{"t":"Backend: Node + Express","items":["Node.js + Express REST API","PostgreSQL database + Prisma ORM","JWT authentication + bcrypt","Deploy API on Railway or Render"]},{"t":"Full Stack Integration","items":["Connect React frontend to Express API","MongoDB for NoSQL use case","Build full CRUD application","CI/CD with GitHub Actions"]}],12:[{"t":"Production-Grade Skills","items":["TypeScript: strict mode, interfaces, generics","Docker + docker-compose","AWS S3 for file storage, EC2 for hosting","System design: caching, load balancing"]},{"t":"Portfolio","items":["2 full-stack deployed projects with README","Record 2-minute demo video","Polish LinkedIn + GitHub","Apply to internships and junior roles"]}]}},
    "Data Scientist":{"required":["python","sql","statistics","pandas","numpy","scikit-learn","matplotlib","machine learning","tableau"],"weeks":{4:[{"t":"Python + SQL","items":["Python: functions, OOP, list comprehensions","Pandas: merge, groupby, pivot, clean data","SQL: SELECT, JOIN, GROUP BY, CTEs, window functions","Jupyter notebook best practices"]},{"t":"Statistics Foundation","items":["Descriptive stats + distributions","Hypothesis testing: p-values, t-test, chi-square","Probability: Bayes theorem","A/B testing fundamentals"]}],8:[{"t":"Machine Learning","items":["Regression: Linear, Ridge, Lasso","Classification: Logistic, SVM, Random Forest","Cross-validation + hyperparameter tuning","Kaggle beginner competition (Titanic)"]},{"t":"Visualisation + Storytelling","items":["Matplotlib + Seaborn — 20 chart types","Tableau or Power BI dashboard","Data storytelling principles","Build a business insight dashboard"]}],12:[{"t":"Advanced ML + Deployment","items":["Ensemble methods: XGBoost, LightGBM","Time series: ARIMA, Prophet","NLP basics: TF-IDF, sentiment analysis","Flask API for model serving"]},{"t":"Portfolio","items":["3 Kaggle notebooks (bronze medals)","1 end-to-end data project","Medium blog post with charts","LinkedIn: post weekly data insights"]}]}},
    "DevOps / Cloud Engineer":{"required":["linux","docker","kubernetes","aws","terraform","ci/cd","python","bash","git","monitoring"],"weeks":{4:[{"t":"Linux + Networking","items":["Linux CLI mastery: 50 essential commands","Bash scripting: loops, functions, cron jobs","Networking: OSI, TCP/IP, DNS, HTTP/HTTPS","SSH, firewalls, file permissions"]},{"t":"Docker","items":["Docker concepts: images, containers, volumes","Write production-grade Dockerfiles","Docker Compose: multi-service apps","Containerise a Python Flask app"]}],8:[{"t":"Kubernetes","items":["K8s architecture: pods, services, deployments","kubectl: 30 essential commands","Helm charts: install, customise, create","Minikube local cluster setup"]},{"t":"AWS + Terraform","items":["AWS core: EC2, S3, VPC, IAM, RDS, Lambda","Terraform: variables, modules, state","Deploy 3-tier app on AWS with Terraform","Cost estimation + optimisation"]}],12:[{"t":"CI/CD + Monitoring","items":["GitHub Actions: build, test, deploy pipeline","Prometheus + Grafana: full monitoring stack","ELK stack: centralised logging","Security scanning: SAST, DAST in pipeline"]},{"t":"Certifications","items":["AWS Solutions Architect (SAA-C03) — practice exam","CKA Certified Kubernetes Admin — study","Terraform Associate — study guide","Add certs to LinkedIn, update resume"]}]}},
    "Cybersecurity Analyst":{"required":["networking","linux","python","security","penetration testing","cryptography"],"weeks":{4:[{"t":"Foundations","items":["Networking deep dive: OSI, TCP/IP, DNS, DHCP","Linux + command line for security","Python scripting for automation","CIA Triad, attack vectors, threat modelling"]},{"t":"Web Security","items":["OWASP Top 10 — all vulnerabilities with demos","Burp Suite: intercept, scan, exploit","SQL injection hands-on (DVWA)","XSS + CSRF exploits and mitigations"]}],8:[{"t":"Ethical Hacking","items":["Kali Linux setup + essential tools","Nmap: host discovery, port scan, OS detection","Metasploit Framework basics","Penetration testing methodology (PTES)"]}],12:[{"t":"Certifications + CTF","items":["CompTIA Security+ full study + exam","TryHackMe: complete 5 learning paths","HackTheBox: 10 easy machines","Build a home lab with VirtualBox"]}]}},
}

@app.route('/api/roadmap', methods=['POST'])
def roadmap():
    data     = request.get_json()
    role     = data.get('role', 'AI/ML Engineer')
    skills   = [s.lower().strip() for s in data.get('skills', []) if s.strip()]
    timeline = int(data.get('timeline', 8))
    hours    = int(data.get('hours', 10))

    rm = ROADMAP_DB.get(role, ROADMAP_DB["AI/ML Engineer"])
    required = rm["required"]

    # gap analysis via cosine similarity per skill
    have    = [r for r in required if any(cosine_similarity([s], [r]) > 0.5 for s in skills)]
    missing = [r for r in required if r not in have]
    readiness = round(len(have) / max(len(required), 1) * 100)

    # Pick appropriate week plan
    wk_key = 4 if timeline <= 4 else (8 if timeline <= 8 else 12)
    weeks  = rm["weeks"].get(wk_key, rm["weeks"][8])

    # Resources for missing skills
    resources = get_resources(missing[:4], role)

    return jsonify({
        "readiness": readiness,
        "have":      have,
        "missing":   missing,
        "weeks":     weeks,
        "timeline":  timeline,
        "hours":     hours,
        "total_hours": hours * timeline,
        "resources": resources,
    })


# ══════════════════════════════════════════════════════════════
# UTILITY + MAIN
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    """Frontend calls this to verify server is running."""
    return jsonify({
        "status":     "online",
        "model":      "loaded" if MODEL is not None else "not_loaded",
        "accuracy":   META.get('test_accuracy', 'N/A') if META else 'N/A',
        "message":    "CareerDNA AI backend running" if MODEL else "Run python train_model.py first"
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)