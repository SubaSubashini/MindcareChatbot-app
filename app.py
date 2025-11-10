import os
from pathlib import Path
import sqlite3
import hashlib
import io 
from io import BytesIO
import time
from datetime import datetime
from html import unescape
import re
from gtts import gTTS

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")





# Optional heavy libraries
USE_SENTTRANS = True
USE_UMAP = True
USE_SPEECHREC = True
USE_PYDUB = True
USE_KERAS = True

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    USE_SENTTRANS = False

try:
    import umap
except Exception:
    USE_UMAP = False

try:
    import speech_recognition as sr
except Exception:
    USE_SPEECHREC = False

try:
    from pydub import AudioSegment
except Exception:
    USE_PYDUB = False

try:
    # Keras/Tensorflow may be heavy - optional
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except Exception:
    USE_KERAS = False

# textblob fallback for sentiment if HF not available
try:
    from textblob import TextBlob
    USE_TEXTBLOB = True
except Exception:
    USE_TEXTBLOB = False

# chardet for encoding detection
try:
    import chardet
except Exception:
    chardet = None

# Set Streamlit page config
st.set_page_config(page_title="MindCare Pro AI Chat & Analytics", layout="wide",page_icon="🧠")
ROOT = os.getcwd()
DB_PATH = os.path.join(ROOT,"mindcare.db")

#Load CSS function
def load_css(file_name="style.css"):
    """Read CSS file and inject into Streamlit app"""
    css_path = Path(__file__).parent/file_name
    if css_path.exists():
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.markdown("style.css not found!")      
          # 3️⃣ Apply CSS
load_css("style.css")  # make sure your CSS file is in the same folder as app.py
st.markdown("""
<div class="bubble"></div>
<div class="bubble"></div>
<div class="bubble"></div>
<div class="bubble"></div>
<div class="bubble"></div>
""", unsafe_allow_html=True)
# ------------------------------
# Database / Auth helpers
# ------------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn

def init_db():
    conn = get_conn()
    c = conn.cursor()
    # users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT
        );
    """)
    # chat_history table
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            speaker TEXT,
            message TEXT,
            sentiment TEXT,
            emotion TEXT,
            intent TEXT
        );
    """)
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def create_user(username: str, password: str):
    conn = get_conn()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, datetime('now'))",
                  (username, hash_password(password)))
        conn.commit()
        return True, "Account created"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    finally:
        conn.close()

def verify_user(username: str, password: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if not row:
        return False
    stored_hash = row[0]
    return stored_hash == hash_password(password)

# chat DB helpers
def save_chat_to_db(speaker, message, sentiment, emotion, intent):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (timestamp, speaker, message, sentiment, emotion, intent) VALUES (?, ?, ?, ?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), speaker, message, sentiment, emotion, intent))
    conn.commit()
    conn.close()

def load_chat_history_df(limit=None):
    conn = get_conn()
    q = "SELECT * FROM chat_history ORDER BY id ASC"
    df = pd.read_sql_query(q, conn, parse_dates=["timestamp"])
    conn.close()
    if limit:
        return df.tail(limit)
    return df

# initialize DB
init_db()
# create default admin (if not exists)
create_user("admin", "admin123")

# ------------------------------
# Model / pipeline resources
# ------------------------------
@st.cache_resource(show_spinner=False)
def load_resources():
    resources = {}
    if USE_SENTTRANS:
        try:
            resources['embedder'] = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception:
            resources['embedder'] = None
    else:
        resources['embedder'] = None
    return resources

RES = load_resources()

# ------------------------------
# Inference helper (fallback)
# ------------------------------
def infer_sentiment(text: str):
    # prefer textblob if available for polarity fallback
    text = str(text)
    if USE_TEXTBLOB:
        pol = TextBlob(text).sentiment.polarity
        label = "positive" if pol > 0.05 else ("negative" if pol < -0.05 else "neutral")
        return label.capitalize(), float(pol)
    # simple rule fallback
    t = text.lower()
    if any(w in t for w in ["happy","good","great","love","joy","awesome"]):
        return "Positive", 0.9
    if any(w in t for w in ["sad","depress","unhappy","bad","angry"]):
        return "Negative", -0.8
    return "Neutral", 0.0

def infer_emotion(text: str):
    t = text.lower()
    if any(w in t for w in ["sad","depress","cry"]): return "sadness", 0.9
    if any(w in t for w in ["happy","joy","excited"]): return "joy", 0.9
    if any(w in t for w in ["angry","mad","furious"]): return "anger", 0.9
    if any(w in t for w in ["scared","fear","panic","anxious"]): return "fear", 0.9
    return "neutral", 0.5

def infer_intent(text: str):
    t = text.lower()
    if any(w in t for w in ["help","support","suicide","hurt"]): return "seeking_help", 0.95
    if any(w in t for w in ["how","what","why","when","where"]): return "question", 0.6
    if any(w in t for w in ["hi","hello","hey"]): return "greeting", 0.8
    return "statement", 0.4

# ------------------------------
# Utility helpers
# ------------------------------
def safe_read_csv(uploaded_file):
    b = uploaded_file.read()
    # detect encoding if chardet available
    enc = None
    if chardet is not None:
        try:
            r = chardet.detect(b)
            enc = r.get('encoding', None)
        except Exception:
            enc = None
    # try reading with detected encoding, fallback to utf-8 and latin1
    for enc_try in [enc, 'utf-8', 'latin1', 'cp1252']:
        if not enc_try:
            continue
        try:
            return pd.read_csv(io.BytesIO(b), encoding=enc_try)
        except Exception:
            continue
    # final fallback: try without specifying encoding
    try:
        return pd.read_csv(io.BytesIO(b))
    except Exception as e:
        raise e

def clean_text_basic(t, lowercase=True, remove_punc=True):
    x = str(t)
    x = unescape(x)
    if lowercase:
        x = x.lower()
    if remove_punc:
        x = re.sub(r'[^a-z0-9\s]', ' ', x)
    x = re.sub(r'\s+', ' ', x).strip()
    return x

def download_dataframe(df: pd.DataFrame, filename="data.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    return csv

# ------------------------------
# App UI : Authentication
# ------------------------------
st.sidebar.title("Account")
auth_choice = st.sidebar.selectbox("Action", ["Login", "Sign Up", "About"])

if auth_choice == "Sign Up":
    st.sidebar.subheader("Create account")
    new_user = st.sidebar.text_input("Username", key="su_user")
    new_pass = st.sidebar.text_input("Password", type="password", key="su_pass")
    if st.sidebar.button("Create Account"):
        ok, msg = create_user(new_user, new_pass)
        if ok:
            st.sidebar.success("Account created. Please login.")
        else:
            st.sidebar.error(msg)
    st.stop()
elif auth_choice == "About":
    st.sidebar.markdown("*MindCare Pro*\n- Demo capstone project\n- Text + Voice + Analytics")
    st.stop()

# Login
st.sidebar.subheader("Login")
username = st.sidebar.text_input("Username", key="login_user")
password = st.sidebar.text_input("Password", type="password", key="login_pass")
if st.sidebar.button("Login"):
    if verify_user(username, password):
        st.sidebar.success(f"Welcome, {username}")
        st.session_state['user'] = username
    else:
        st.sidebar.error("Invalid credentials")

if 'user' not in st.session_state:
    st.stop()

# ------------------------------
# Main layout: Tabs
# ------------------------------
st.title("🧠 MindCare Pro — Multimodal Mental Health Chatbot")
st.markdown("<div class='small-muted'>Secure login → Data → Preprocess → Feature engineering → Modeling → Evaluation → Deployment</div>", unsafe_allow_html=True)

tabs = st.tabs(["Home", "Data & EDA", "Preprocess & Features", "Modeling & Eval", "Chat (Text+Voice)", "Dashboard & Visuals", "Export & Deploy"])

# -------------------------
# Home
# -------------------------
with tabs[0]:
    st.header("Project Overview")
    st.markdown("""
    MindCare demonstrates a full pipeline:
    - Data collection, preprocessing, feature engineering (ML/DL)
    - Multi-task inference: sentiment, emotion, intent
    - Voice transcription support (upload -> SpeechRecognition)
    - Baseline (TF-IDF + Logistic), optional LSTM
    - Embeddings + UMAP/PCA
    - Dashboard: 20+ visualizations, export & metrics
    """)
    st.info("Admin default: username=admin password=admin123 (created automatically).")

# -------------------------
# Data & EDA
# -------------------------
with tabs[1]:
    st.header("📂 Data Collection & EDA")
    st.markdown("Upload a CSV with at least a message (or text) column. Optionally include sentiment, emotion, or intent.")
    uploaded = st.file_uploader("Upload dataset (CSV)", type=['csv'])
    if uploaded:
        try:
            df = safe_read_csv(uploaded)
            st.session_state['df_raw'] = df
            st.success(f"Loaded dataset with {len(df)} rows, {len(df.columns)} columns")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        if 'df_raw' not in st.session_state:
            st.info("No dataset uploaded. Use Chat tab to collect messages.")

    if 'df_raw' in st.session_state:
        df = st.session_state['df_raw'].copy()
        st.markdown("### Basic stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(df))
        text_col = None
        for c in df.columns:
            if c.lower() in ('text','message','utterance','content'):
                text_col = c
                break
        if text_col:
            col2.metric("Missing in text", int(df[text_col].isnull().sum()))
        else:
            col2.metric("Text column", "Not found")
        if 'sentiment' in df.columns:
            col3.metric("Unique labels", df['sentiment'].nunique())

        # text length distribution if text_col present
        if text_col:
            df['text_len'] = df[text_col].astype(str).apply(len)
            fig = px.histogram(df, x='text_len', nbins=50, title="Text length distribution")
            st.plotly_chart(fig, use_container_width=True)

            wc = WordCloud(width=800, height=300, background_color='white').generate(" ".join(df[text_col].astype(str).tolist()))
            st.image(wc.to_array(), use_container_width=True)

            if st.checkbox("Show top TF-IDF tokens (preview)"):
                v = TfidfVectorizer(max_features=50, stop_words='english')
                X = v.fit_transform(df[text_col].astype(str))
                freqs = dict(zip(v.get_feature_names_out(), X.sum(axis=0).A1))
                freq_df = pd.DataFrame(list(freqs.items()), columns=['token','score']).sort_values('score', ascending=False).head(30)
                st.plotly_chart(px.bar(freq_df, x='token', y='score', title="Top TF-IDF tokens"), use_container_width=True)
# -------------------------
# Preprocess & Features
# -------------------------
with tabs[2]:
    st.header("🧹 Preprocessing & Feature Engineering")

    if 'df_raw' not in st.session_state:
        st.warning("Upload dataset in the 'Data & EDA' tab first.")
    else:
        df = st.session_state['df_raw'].copy()
        st.subheader("Before Cleaning")
        st.dataframe(df.head())

        # --- Cleaning Settings ---
        lowercase = st.checkbox("Lowercase", True)
        remove_punct = st.checkbox("Remove punctuation", True)

        import re
        def clean_text(x):
            x = str(x)
            if lowercase:
                x = x.lower()
            if remove_punct:
                x = re.sub(r'[^a-zA-Z0-9\s]', ' ', x)
            return re.sub(r'\s+', ' ', x).strip()

        df['text_clean'] = df['text'].apply(clean_text)
        st.subheader("After Cleaning")
        st.dataframe(df[['text', 'text_clean']].head())

        # --- TF-IDF Feature Extraction ---
        from sklearn.feature_extraction.text import TfidfVectorizer  # ✅ add this import
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        if st.button("🔹 Create TF-IDF Features"):
            # ✅ define vect before using it
            vect = TfidfVectorizer(max_features=1000, stop_words='english')
            X = vect.fit_transform(df['text_clean'])

            # Combine with the dataframe for saving
            df_features = df.copy()
            df_features['text_vectorized'] = list(X.toarray())

            # Save to session state for next tab
            st.session_state['df_features'] = df_features
            st.session_state['vectorizer'] = vect

            st.success("✅ TF-IDF Features Created Successfully!")
            st.write("Feature Matrix Shape:", X.shape)
            st.write("Sample Tokens:", vect.get_feature_names_out()[:30])
            st.dataframe(df_features.head())

# -------------------------
# 🧠 Model Training & Evaluation (using df_features)
# -------------------------

with tabs[3]:
    st.header("🧠 Model Training & Evaluation")

    # --- Check if Preprocessed Data Exists ---
    if 'df_features' not in st.session_state:
        st.info("⚠️ Please preprocess data first in the 'Preprocessing' tab.")
    else:
        df_features = st.session_state['df_features'].copy()

        # --- Auto-detect sentiment column ---
        possible_cols = ['sentiment', 'label', 'emotion', 'target', 'class']
        sentiment_col = None
        for c in df_features.columns:
            if c.lower() in possible_cols:
                sentiment_col = c
                break

        # --- If not found, create dummy labels (for demo) ---
        if not sentiment_col:
            st.warning("⚠️ No sentiment/label column found — creating dummy labels for demo.")
            df_features['sentiment'] = np.random.choice(['positive', 'negative', 'neutral'], size=len(df_features))
            sentiment_col = 'sentiment'
        else:
            st.success(f"✅ Found sentiment column: {sentiment_col}")

        # --- Show data preview ---
        st.subheader("Data Preview for Training")
        st.dataframe(df_features.head())

        # --- Split Data ---
        X = df_features['text_clean'].astype(str).values
        y = df_features[sentiment_col].astype(str).values
        test_size = st.slider("Test size", 0.05, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # --- Train Model ---
        if st.button("🚀 Train TF-IDF + Logistic Regression Model"):
            with st.spinner("Training the model... Please wait..."):
                pipe = Pipeline([
                    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, stop_words='english')),
                    ("clf", LogisticRegression(max_iter=2000))
                ])
                pipe.fit(X_train, y_train)
                st.session_state['baseline_model'] = pipe
                st.success("✅ Model trained successfully!")

        # --- Evaluation ---
        if 'baseline_model' in st.session_state:
            pipe = st.session_state['baseline_model']
            y_pred = pipe.predict(X_test)

            # --- Accuracy Metric ---
            acc = accuracy_score(y_test, y_pred)
            st.metric("Model Accuracy", f"{acc:.3f}")

            # --- Confusion Matrix ---
            st.subheader("📊 Confusion Matrix")
            labels = sorted(list(set(y_test) | set(y_pred)))
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            fig_cm = px.imshow(
                cm, x=labels, y=labels, text_auto=True,
                color_continuous_scale='blues', title="Confusion Matrix"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # --- Classification Report ---
            st.subheader("📋 Classification Report")
            cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(cr).transpose())

            # --- ROC / AUC ---
            st.subheader("📈 ROC Curve & AUC")
            try:
                if len(set(y_test)) == 2:
                    y_score = pipe.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve((np.array(y_test) == 'positive').astype(int), y_score)
                    auc_val = roc_auc_score((np.array(y_test) == 'positive').astype(int), y_score)
                    st.success(f"ROC AUC (binary): {auc_val:.3f}")
                    fig_roc = px.line(x=fpr, y=tpr, title="ROC Curve", labels={'x': 'FPR', 'y': 'TPR'})
                    st.plotly_chart(fig_roc, use_container_width=True)
                else:
                    y_proba = pipe.predict_proba(X_test)
                    y_test_b = pd.get_dummies(pd.Series(y_test))
                    auc_val = roc_auc_score(y_test_b, y_proba, average='macro', multi_class='ovr')
                    st.success(f"ROC AUC (multi-class, macro): {auc_val:.3f}")
            except Exception as e:
                st.warning(f"⚠️ ROC/AUC skipped: {e}")

            # --- Cross Validation ---
            if st.button("🔁 5-Fold Cross Validation"):
                with st.spinner("Running 5-fold CV..."):
                    pipe_cv = Pipeline([
                        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, stop_words='english')),
                        ("clf", LogisticRegression(max_iter=2000))
                    ])
                    preds = cross_val_predict(pipe_cv, X, y, cv=5, method='predict')
                    acc_cv = accuracy_score(y, preds)
                    st.success(f"✅ Cross-validated Accuracy (5-fold): {acc_cv:.3f}")
                    st.dataframe(pd.DataFrame(classification_report(y, preds, output_dict=True)).transpose())

        
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Embedding, LSTM, Dense
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            st.markdown("#🔬 Optional: LSTM Sentiment Model (Demo)")
            if st.button("Train Small LSTM Demo"):
                with st.spinner("Training small LSTM model..."):
                    tok = Tokenizer(num_words=5000)
                    tok.fit_on_texts(df_features['text_clean'])
                    X_seq = tok.texts_to_sequences(df_features['text_clean'])
                    X_pad = pad_sequences(X_seq, maxlen=100)
                    y_enc = pd.get_dummies(df_features[sentiment_col]).values

                    model = Sequential([
                        Embedding(5000, 64, input_length=100),
                        LSTM(64),
                        Dense(y_enc.shape[1], activation='softmax')
                    ])
                    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                    history = model.fit(X_pad, y_enc, epochs=3, batch_size=64, validation_split=0.1, verbose=1)

                    st.session_state['lstm_model'] = model
                    st.success("✅ LSTM model trained successfully!")

                    # Plot training history
                    fig_lstm = px.line(
                        history.history, title="LSTM Training History",
                        labels={'index': 'Epoch', 'value': 'Score', 'variable': 'Metric'}
                    )
                    st.plotly_chart(fig_lstm, use_container_width=True)
        except ImportError:
            st.info("💡 TensorFlow not installed — LSTM demo skipped.")
            
# -------------------------
# Tab: Chat (Text + Voice)
# -------------------------
with tabs[4]:
    DB_PATH = "chat_history.db"

    def init_db():
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        speaker TEXT,
                        message TEXT,
                        sentiment TEXT,
                        emotion TEXT,
                        intent TEXT
                    )''')
        conn.commit()
        conn.close()

    def save_chat(speaker, message, sentiment, emotion, intent):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO chat_history (timestamp, speaker, message, sentiment, emotion, intent) VALUES (?,?,?,?,?,?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), speaker, message, sentiment, emotion, intent))
        conn.commit()
        conn.close()

    def load_chat_history():
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM chat_history ORDER BY id ASC", conn)
        conn.close()
        return df

    # Initialize database
    init_db()

    # --- Dummy Inference Functions (replace with your ML models if needed) ---
    def infer_sentiment(text):
        text = text.lower()

        if any(w in text for w in ["happy", "great", "love", "good", "joy"]):
            return "Positive", 0.92
        elif any(w in text for w in ["sad", "bad", "upset", "depressed", "angry"]):
            return "Negative", 0.87
        else:
            return "Neutral", 0.70

    def infer_emotion(text):
        text = text.lower()
        if "angry" in text:
            return "Anger", 0.85
        elif "fear" in text or "scared" in text:
            return "Fear", 0.83
        elif "happy" in text or "joy" in text:
            return "Joy", 0.90
        elif "sad" in text or "depressed" in text:
            return "Sadness", 0.88
        else:
            return "Neutral", 0.75

    def infer_intent(text):
        text = text.lower()
        if "help" in text:
            return "Seeking Support", 0.91
        elif "bye" in text:
            return "Goodbye", 0.85
        elif "thank" in text:
            return "Gratitude", 0.88
        else:
            return "General Chat", 0.80

    # --- AI Reply Function ---
    def analyze_message(message):
        sentiment_label, sentiment_score = infer_sentiment(message)
        emotion_label, emotion_score = infer_emotion(message)
        intent_label, intent_score = infer_intent(message)

        analysis = (
            f"🧩 Sentiment: {sentiment_label} ({sentiment_score:.2f})\n"
            f"🎭 Emotion: {emotion_label} ({emotion_score:.2f})\n"
            f"🎯 Intent: {intent_label} ({intent_score:.2f})"
        )

        # Basic AI response logic
        if emotion_label.lower() == "sadness":
            ai_reply = "I'm here for you. It’s okay to feel sad. Would you like to share more?,Okay 💙 here’s something deeper for you —Sometimes life feels like a storm that just won’t end, and it’s easy to forget how strong you’ve already been through everything before. But you are still here — breathing, trying, feeling — and that means hope is still alive inside you.You don’t have to be okay right now. You just have to keep going, one small step at a time. The sadness you feel doesn’t define you — it’s just a chapter, not the whole story. And someday soon, you’ll look back and realize that even in your quietest pain, you were growing stronger than ever. 🌙💫"
        elif emotion_label.lower() == "joy":
            ai_reply = "That's wonderful! I'm glad you're feeling positive!,Joy is the soft light that warms your heart after a long night of doubt. It reminds you that life still holds gentle surprises and reasons to smile. Each small laugh, each calm breath, is proof that happiness can return in the simplest moments. You don’t have to search for it — sometimes it finds you when you least expect it. Let today be a quiet celebration of how far you’ve come and how beautifully you’re beginning to glow again. ✨💛"
        elif emotion_label.lower() == "anger":
            ai_reply = "When anger rises, pause and breathe — you don’t have to let it rule your heart.Walk away from the heat, and give your mind a moment to cool.Speak gently, even when your feelings burn inside you.Peace always finds those who choose calm over chaos. 🌿"
        elif emotion_label.lower() == "fear":
            ai_reply = "Fear may whisper that you can’t, but your strength quietly answers that you can. Each time you face what scares you, you grow a little braver than before. Remember — courage doesn’t mean you’re never afraid; it means you keep going anyway. Breathe, believe, and trust that you have the power to handle whatever comes. The more you choose hope over fear, the lighter your heart becomes. You are stronger than the shadows that try to hold you back. 🌤️💫I’m here with you."
        else:
            ai_reply = "Thanks for sharing. Tell me more about your feelings."

        return analysis, ai_reply, sentiment_label, emotion_label, intent_label

    # --- Text-to-Speech ---
    def speak_text(text):
        tts = gTTS(text=text, lang='en')
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        st.audio(audio_fp, format="audio/mp3")

    # --- TEXT CHAT ---
    st.subheader("💬 Text Chat")
    user_input = st.text_input("Type your message:", key="user_input")

    if st.button("Send Text Message"):
        if user_input.strip():
            analysis, ai_reply, sent, emo, intent = analyze_message(user_input)
            st.info(analysis)
            speak_text(ai_reply)

            save_chat("You", user_input, sent, emo, intent)
            save_chat("AI", ai_reply, sent, emo, intent)

            st.success("✅ AI reply generated and chat saved!")
        else:
            st.warning("Please type a message before sending.")

    st.markdown("---")

    # --- VOICE CHAT ---
    st.subheader("🎙️ Voice Chat (Upload your voice file)")
    uploaded_audio = st.file_uploader("Upload audio (WAV or MP3)", type=["wav", "mp3"])

    if uploaded_audio is not None:
        with open(uploaded_audio.name, "wb") as f:
            f.write(uploaded_audio.getbuffer())
        audio_path = uploaded_audio.name

        if uploaded_audio.type == "audio/mpeg":
            sound = AudioSegment.from_mp3(audio_path)
            audio_path = "converted_audio.wav"
            sound.export(audio_path, format="wav")
            st.info("Converted MP3 to WAV format for recognition")

        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                st.success(f"🗣️ Recognized Speech: {text}")

                analysis, ai_reply, sent, emo, intent = analyze_message(text)
                st.info(analysis)
                speak_text(ai_reply)

                save_chat("You (Voice)", text, sent, emo, intent)
                save_chat("AI", ai_reply, sent, emo, intent)

        except Exception as e:
            st.error(f"Speech recognition failed: {e}")

    st.markdown("---")
    # --- CHAT HISTORY (Grouped by Date) ---
st.subheader("📜 Conversation History (Grouped by Date)")
chat_df = load_chat_history()

if not chat_df.empty:
    # Extract only date (no time)
    chat_df["date"] = pd.to_datetime(chat_df["timestamp"]).dt.date

    # Group by each date
    for date, group in chat_df.groupby("date"):
        st.markdown(f"### 🗓️ {date.strftime('%B %d, %Y')}")
        for _, row in group.iterrows():
            color = "#00FFFF" if "You" in row["speaker"] else "#FF66FF"
            st.markdown(
                f"<div style='color:{color}'><b>{row['speaker']}:</b> {row['message']} "
                f"<small style='color:gray;'>({row['timestamp'].split(' ')[1]})</small></div>",
                unsafe_allow_html=True,
            )
        st.markdown("---")
else:
    st.info("No previous chat history yet.")
    st.markdown("---")

    # --- ACCURACY + DOWNLOAD ---
    st.subheader("📊 Sentiment & Emotion Stats + Download")

    if not chat_df.empty:
        if st.button("Show Stats"):
            sentiment_counts = chat_df["sentiment"].value_counts(normalize=True) * 100
            emotion_counts = chat_df["emotion"].value_counts(normalize=True) * 100

            st.write("✅ *Sentiment Distribution (%):*")
            st.bar_chart(sentiment_counts)

            st.write("🎭 *Emotion Distribution (%):*")
            st.bar_chart(emotion_counts)

        # --- Download CSV Button ---
        csv_data = chat_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Chat History as CSV",
            data=csv_data,
            file_name="mindcare_chat_history.csv",
            mime="text/csv",
        )
    else:
        st.warning("No data available yet.")


# -------------------------
# Dashboard & Visuals
# -------------------------
with tabs[5]:
    st.header("🧩Advanced Visualizations Dashboard")
    df = load_chat_history_df()
    if df.empty:
        st.info("No chat data available. Use Chat tab to add messages or upload dataset.")
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['msg_len'] = df['message'].astype(str).apply(len)
        # sentiment_score numeric mapping
        df['sentiment_score'] = df['sentiment'].map({'Positive':1.0,'positive':1.0,'Negative':-1.0,'negative':-1.0,'Neutral':0.0,'neutral':0.0}).fillna(0.0)
        df['hour'] = df['timestamp'].dt.hour
        df['date'] = df['timestamp'].dt.date
        col1, col2, col3 = st.columns(3)
        col1.metric("Total messages", len(df))
        col2.metric("Unique speakers", df['speaker'].nunique())
        top_em = df['emotion'].mode().iloc[0] if not df['emotion'].mode().empty else "N/A"
        col3.metric("Top emotion", top_em)

        st.subheader("🎯 Sentiment distribution (pie)")
        st.plotly_chart(px.pie(df, names='sentiment', hole=0.3), use_container_width=True)

        st.subheader("📊 Emotion counts")
        emo_counts = df['emotion'].value_counts().reset_index()
        emo_counts.columns = ['emotion','count']
        st.plotly_chart(px.bar(emo_counts, x='emotion', y='count', color='emotion'), use_container_width=True)

        st.subheader("🎭Intent distribution")
        intent_counts = df[['intent']].value_counts().reset_index()
        intent_counts.columns = ['intent', 'count']

        st.plotly_chart(
                px.bar(intent_counts, x='intent', y='count', color='intent', title='Intent Frequency'),
            use_container_width=True
        )

        st.subheader("🎯Emotion vs sentiment scatter")
        emotion_map = {e:i for i,e in enumerate(df['emotion'].unique())}
        df['emotion_id'] = df['emotion'].map(emotion_map)
        fig5 = px.scatter(df, x='sentiment_score', y='emotion_id', color='emotion', hover_data=['message'])
        fig5.update_yaxes(tickmode='array', tickvals=list(emotion_map.values()), ticktext=list(emotion_map.keys()))
        st.plotly_chart(fig5, use_container_width=True)

        st.subheader("🎭 WordCloud (choose emotion)")
        emo_sel = st.selectbox("Emotion for wordcloud", options=df['emotion'].unique())
        texts = " ".join(df[df['emotion']==emo_sel]['message'].astype(str).tolist())
        if texts.strip():
            wc = WordCloud(width=900, height=300).generate(texts)
            st.image(wc.to_array(), use_container_width=True)
        else:
            st.info("No messages for selected emotion.")

        st.subheader("🎯Confusion matrix (upload CSV with true labels to compare)")
        gt_file = st.file_uploader("Upload CSV with ['message','true_sentiment']", type=['csv'], key="gt_cmp")
        if gt_file is not None:
            try:
                gt = pd.read_csv(gt_file)
                merged = pd.merge(gt, df, left_on='message', right_on='message', how='left').dropna(subset=['sentiment'])
                if merged.empty:
                    st.error("No matches between uploaded ground truth and chat messages")
                else:
                    y_true = merged['true_sentiment']
                    y_pred = merged['sentiment']
                    labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    st.plotly_chart(px.imshow(cm, x=labels, y=labels, text_auto=True, title="Confusion Matrix"), use_container_width=True)
                    st.dataframe(pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose())
            except Exception as e:
                st.error("Error processing GT file: " + str(e))

        st.subheader("🎯 KMeans cluster (on msg_len, hour, sentiment_score)")
        with st.expander("Clustering options"):
            k = st.slider("k clusters", 2, 10, 3, key="k_kmeans_dashboard")
            Xk = df[['msg_len','hour','sentiment_score']].fillna(0).values
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xk)
                coords = PCA(n_components=2).fit_transform(Xk)
                clust_df = pd.DataFrame(coords, columns=['x','y'])
                clust_df['cluster'] = kmeans.labels_.astype(str)
                clust_df['message'] = df['message'].values
                st.plotly_chart(px.scatter(clust_df, x='x', y='y', color='cluster', hover_data=['message']), use_container_width=True)
            except Exception as e:
                st.info("Clustering failed: " + str(e))

        st.subheader("📊Hourly volume & 10) Cumulative messages")
        hourly = df.groupby('hour').size().reset_index(name='count')
        st.plotly_chart(px.bar(hourly, x='hour', y='count', title="Messages by hour"), use_container_width=True)
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        df_sorted['cum'] = np.arange(1, len(df_sorted)+1)
        st.plotly_chart(px.area(df_sorted, x='timestamp', y='cum', title="Cumulative messages"), use_container_width=True)
        # 1️⃣ Sentiment Trend Over Time
        st.subheader("🕒 Sentiment Trend Over Time")
        fig = px.line(df, x="timestamp", y="sentiment", color="speaker",
                      markers=True, title="Sentiment Scores Over Time")
        st.plotly_chart(fig, use_container_width=True)

        # 2️⃣ Emotion Distribution
        st.subheader("🎭 Emotion Distribution")
        fig = px.pie(df, names="emotion", hole=0.3, title="Emotion Breakdown")
        st.plotly_chart(fig, use_container_width=True)

        # 3️⃣ Intent Distribution
        st.subheader("🎯 Intent Distribution")
        fig = px.bar(
        intent_counts,
        x='intent',
        y='count',
        color='intent',
        title='Intent Frequency Distribution'
        )

        # 4️⃣ Sentiment vs Emotion Heatmap
        st.subheader("🔥 Sentiment vs Emotion Correlation")
        heat_df = df.groupby(["sentiment", "emotion"]).size().reset_index(name="count")
        heat_pivot = heat_df.pivot(index="sentiment", columns="emotion", values="count").fillna(0)
        fig, ax = plt.subplots()
        sns.heatmap(heat_pivot, annot=True, fmt=".0f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # 5️⃣ Word Cloud
        st.subheader("☁️ Word Cloud of Chat Messages")
        text_data = " ".join(df["message"].astype(str))
        wc = WordCloud(width=900, height=400, background_color="black", colormap="plasma").generate(text_data)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

        # 6️⃣ Message Count by User
        st.subheader("👥 Messages per Speaker")
        speaker_counts = df['speaker'].value_counts().reset_index()

        # Step 2: Explicitly set unique column names
        speaker_counts.columns = ['speaker', 'count']

        # Step 3: Sort properly
        speaker_counts = speaker_counts.sort_values(by='count', ascending=False)

        # Step 4: Plot
        fig = px.bar(
            speaker_counts,
            x='speaker',
            y='count',
            color='speaker',
            text='count',
            title='Messages by Speaker'
        )
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)




        # 7️⃣ Sentiment Polarity Histogram
        st.subheader("📈 Sentiment Histogram")
        fig = px.histogram(df, x="sentiment", nbins=10, title="Sentiment Value Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # 8️⃣ Emotion Over Time
        st.subheader("⏳ Emotion Over Time")
        fig = px.area(df, x="timestamp", color="emotion", title="Emotion Over Time (Area Plot)")
        st.plotly_chart(fig, use_container_width=True)

        # 🔟 Word Frequency Bar
        st.subheader("🔠 Top Keywords Frequency")
        from collections import Counter
        tokens = " ".join(df["message"].astype(str)).lower().split()
        top_words = Counter(tokens).most_common(30)
        freq_df = pd.DataFrame(top_words, columns=["word", "count"])
        fig = px.bar(freq_df, x="word", y="count", title="Most Common Words")
        st.plotly_chart(fig, use_container_width=True)

        # 12️⃣ Daily Chat Activity
        st.subheader("📆 Daily Chat Activity")
        df["date"] = df["timestamp"].dt.date
        daily = df.groupby("date").size().reset_index(name="count")
        fig = px.line(daily, x="date", y="count", title="Daily Chat Activity")
        st.plotly_chart(fig, use_container_width=True)

        # 13️⃣ Hourly Activity Heatmap
        st.subheader("🕗 Hourly Chat Activity")
        df["hour"] = df["timestamp"].dt.hour
        hour_heat = df.groupby(["hour", "emotion"]).size().reset_index(name="count")
        heat_pivot2 = hour_heat.pivot(index="hour", columns="emotion", values="count").fillna(0)
        fig, ax = plt.subplots()
        sns.heatmap(heat_pivot2, cmap="coolwarm", annot=True, fmt=".0f", ax=ax)
        st.pyplot(fig)

        
        # 15️⃣ Real-Time Gauge (Happiness Index)
        st.subheader("🧭 Emotional Happiness Gauge")

        import plotly.graph_objects as go

        # Convert text sentiment labels to numeric scores
        sentiment_map = {
            "Negative": -1,
            "Neutral": 0,
            "Positive": 1
        }

        # Safely map the sentiments
        df["sentiment_score"] = df["sentiment"].map(sentiment_map)

        # Compute average sentiment score (ignore NaN)
        avg_sentiment = df["sentiment_score"].mean()

        # Handle case if no data available
        if pd.isna(avg_sentiment):
            avg_sentiment = 0

        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_sentiment,
            title={'text': "Average Sentiment"},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': "lightgreen"},
                'steps': [
                    {'range': [-1, -0.2], 'color': "red"},
                    {'range': [-0.2, 0.2], 'color': "yellow"},
                    {'range': [0.2, 1], 'color': "green"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)



        # 16️⃣ Emotion Intensity Radar
        st.subheader("🌈 Emotion Intensity Radar Chart")
        emo_counts = df["emotion"].value_counts().reset_index()
        emo_counts.columns = ["emotion", "count"]
        fig = px.line_polar(emo_counts, r="count", theta="emotion", line_close=True)
        st.plotly_chart(fig, use_container_width=True)

        # 17️⃣ WordCloud per Emotion
        st.subheader("💫 WordCloud per Emotion")
        emotions = df["emotion"].unique()
        for emo in emotions:
            emo_text = " ".join(df[df["emotion"] == emo]["message"].astype(str))
            wc = WordCloud(width=800, height=400, background_color="white").generate(emo_text)
            st.image(wc.to_array(), caption=f"WordCloud — {emo}", use_container_width=True)

        # 18️⃣ Conversation Volume per User over Time
        st.subheader("💬 Message Volume per Speaker")
        count_by_user = df.groupby(["speaker", "date"]).size().reset_index(name="count")
        fig = px.area(count_by_user, x="date", y="count", color="speaker",
                      title="Conversation Volume per Speaker")
        st.plotly_chart(fig, use_container_width=True)

        
        # 20️⃣ RNN/DL Output Statistics Simulation
        st.subheader("🧠 Deep Learning Model Performance (Simulated Metrics)")
        metrics = {
            "Accuracy": [0.93],
            "Precision": [0.91],
            "Recall": [0.89],
            "F1-Score": [0.90],
            "AIC": [450.7],
            "BIC": [470.3],
            "R2": [0.86]
        }
        metric_df = pd.DataFrame(metrics)
        st.dataframe(metric_df.style.highlight_max(axis=1, color="lightgreen"))

        st.success("✅ Dashboard updated dynamically from chat + voice interactions.")

# -------------------------
# Export & Deploy
# -------------------------
with tabs[6]:
    st.header("Export & Deployment")
    st.markdown("""
    - Download chat history CSV
    - Save baseline model
    - Deployment tips
    """)
    df = load_chat_history_df()
    if not df.empty:
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download chat history", data=csv_bytes, file_name="chat_history.csv", mime="text/csv")

    if 'baseline_model' in st.session_state:
        if st.button("Save baseline model (joblib)"):
            import joblib
            joblib.dump(st.session_state['baseline_model'], "baseline_model.joblib")
            st.success("Saved baseline_model.joblib")

    st.markdown("### Deployment tips")
    st.markdown("""
    - For production use a managed DB (Postgres), host heavy models on GPUs or use HF inference APIs.
    - Use FastAPI for real-time model serving + React/Streamlit frontend.
    - Always apply privacy & consent for mental health data.
    """)

# End of app
st.markdown("<hr><div class='small-muted'>MindCare Pro — demo version. Use responsibly. Not a replacement for clinical care.</div>", unsafe_allow_html=True)

