from flask import Flask, render_template_string, request, redirect, url_for
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.text as mpl_text
import seaborn as sns
import pandas as pd
import numpy as np
import re
import os
import io
import base64
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from collections import Counter
import pickle

# Ensure necessary NLTK data is available
try:
		nltk.data.find('tokenizers/punkt')
except LookupError:
		nltk.download('punkt', quiet=True)

try:
		nltk.data.find('corpora/stopwords')
except LookupError:
		nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Fallback Indonesian stopwords if not available via NLTK
DEFAULT_IND_STOPWORDS = {
		'yang','dan','di','ke','dari','ini','itu','untuk','pada','dengan','atau','sebagai',
		'adalah','saya','kamu','kita','dia','ya','nya','dengan','oleh','akan','tidak','tida',
		'ada','ini','itu','yg','nih','ga','gak','yg','supaya','biar','masa'
}

def get_stopwords(lang='indonesian'):
		try:
				sw = set(stopwords.words(lang))
				if len(sw) > 10:
						return sw
		except Exception:
				pass
		return DEFAULT_IND_STOPWORDS

STOPWORDS = get_stopwords('indonesian')

# ===================== NAIVE BAYES CLASSIFIERS =====================

# Indonesian Sentiment Lexicon (expanded)
POSITIVE_WORDS = {
    'bagus', 'baik', 'hebat', 'keren', 'mantap', 'suka', 'senang', 'gembira',
    'indah', 'cantik', 'sempurna', 'luar biasa', 'amazing', 'good', 'great',
    'best', 'terbaik', 'sukses', 'berhasil', 'menarik', 'recommended', 'recommend',
    'setuju', 'sip', 'oke', 'ok', 'bangga', 'semangat', 'termotivasi', 'inspiratif',
    'membanggakan', 'wow', 'wah', 'asik', 'asyik', 'jos', 'josss', 'top', 'mantul',
    'mantab', 'cakep', 'ganteng', 'ramah', 'helpful', 'membantu', 'bermanfaat',
    'berkualitas', 'professional', 'profesional', 'favorit', 'favourite', 'love',
    'cinta', 'sayang', 'blessed', 'bersyukur', 'terima kasih', 'terimakasih', 'thanks',
    'thank', 'appreciate', 'apresiasi', 'salut', 'respect', 'hormat', 'kagum',
    'impian', 'dream', 'harapan', 'hope', 'optimis', 'positif', 'positive'
}

NEGATIVE_WORDS = {
    'jelek', 'buruk', 'bodoh', 'tolol', 'goblok', 'benci', 'marah', 'kesal',
    'kecewa', 'sedih', 'gagal', 'sampah', 'busuk', 'worst', 'bad', 'terrible',
    'horrible', 'awful', 'disappointing', 'mengecewakan', 'menyesal', 'bohong',
    'tipu', 'scam', 'penipuan', 'palsu', 'fake', 'hoax', 'bullshit', 'bs',
    'anjing', 'anjir', 'bangsat', 'brengsek', 'kampret', 'kontol', 'memek',
    'tai', 'setan', 'iblis', 'jahat', 'jahanam', 'sialan', 'mampus', 'mati',
    'bego', 'idiot', 'stupid', 'dumb', 'lambat', 'lelet', 'malas', 'males',
    'males', 'capek', 'cape', 'bosan', 'bosen', 'males', 'ogah', 'males',
    'susah', 'ribet', 'repot', 'pusing', 'muak', 'jijik', 'kotor', 'jorok',
    'kasar', 'toxic', 'racun', 'merusak', 'hancur', 'rusak', 'parah', 'payah'
}

# Spam indicators (common spam patterns in Indonesian social media)``
SPAM_INDICATORS = {
    'klik', 'click', 'link', 'bio', 'promo', 'diskon', 'discount', 'gratis',
    'free', 'hadiah', 'prize', 'menang', 'winner', 'claim', 'klaim', 'daftar',
    'register', 'join', 'gabung', 'followers', 'follower', 'like', 'subscribe',
    'subs', 'cek', 'check', 'order', 'pesan', 'beli', 'buy', 'jual', 'sell',
    'murah', 'cheap', 'termurah', 'cheapest', 'terpercaya', 'trusted', 'reseller',
    'dropship', 'cod', 'wa', 'whatsapp', 'dm', 'inbox', 'contact', 'hubungi',
    'slot', 'gacor', 'jackpot', 'bonus', 'deposit', 'withdraw', 'betting', 'bet',
    'casino', 'poker', 'togel', 'lottery', 'judi', 'gambling', 'maxwin', 'rtp',
    'www', 'http', 'https', '.com', '.id', '.net', '.org', 'bit.ly', 'tinyurl',
    'money', 'uang', 'rupiah', 'dollar', 'income', 'penghasilan', 'cuan', 'profit',
    'investment', 'investasi', 'crypto', 'bitcoin', 'trading', 'forex'
}

def create_training_data():
    """Create synthetic training data for sentiment and spam classifiers."""
    # Sentiment training data (Indonesian)
    sentiment_texts = []
    sentiment_labels = []
    
    # Positive examples
    positive_templates = [
        "bagus banget", "keren abis", "mantap sekali", "suka banget",
        "luar biasa", "terbaik", "recommended banget", "sangat membantu",
        "semangat terus", "inspiratif sekali", "bangga sama", "love it",
        "amazing content", "great job", "keep it up", "sukses selalu",
        "terima kasih banyak", "sangat bermanfaat", "top markotop",
        "gokil abis", "josss gandos", "mantul kali", "cakep banget",
        "salut sama usahanya", "respect", "kagum", "wow keren",
        "semoga sukses", "good luck", "all the best", "proud of you"
    ]
    for text in positive_templates:
        sentiment_texts.append(text)
        sentiment_labels.append('positive')
    
    # Add variations with positive words
    for word in list(POSITIVE_WORDS)[:30]:
        sentiment_texts.append(f"{word} sekali ini")
        sentiment_labels.append('positive')
    
    # Negative examples
    negative_templates = [
        "jelek banget", "buruk sekali", "kecewa berat", "mengecewakan",
        "gak bagus", "tidak recommended", "sampah", "worst ever",
        "sangat buruk", "payah banget", "menyesal", "rugi",
        "bohong semua", "tipu tipu", "penipuan", "fake news",
        "bad service", "terrible experience", "sangat kecewa",
        "buang waktu", "tidak berguna", "ribet banget", "susah amat",
        "bosan", "males lihat", "jijik", "kotor banget", "parah sih"
    ]
    for text in negative_templates:
        sentiment_texts.append(text)
        sentiment_labels.append('negative')
    
    # Add variations with negative words
    for word in list(NEGATIVE_WORDS)[:30]:
        sentiment_texts.append(f"{word} banget sih")
        sentiment_labels.append('negative')
    
    # Spam training data
    spam_texts = []
    spam_labels = []
    
    # Spam examples
    spam_templates = [
        "klik link di bio", "cek profil untuk info", "dm untuk order",
        "promo diskon 50%", "gratis ongkir", "hadiah menanti",
        "daftar sekarang dapat bonus", "follow back ya", "like dan subscribe",
        "slot gacor hari ini", "maxwin jackpot", "bonus deposit 100%",
        "jual followers murah", "wa 08xxx untuk order", "hubungi admin",
        "reseller welcome", "dropship bisa", "cod seluruh indonesia",
        "investasi profit tinggi", "cuan setiap hari", "penghasilan jutaan",
        "www.linkspam.com", "bit.ly/promo", "kunjungi website kami",
        "trading forex profit", "crypto to the moon", "bitcoin gratis"
    ]
    for text in spam_templates:
        spam_texts.append(text)
        spam_labels.append('spam')
    
    # Not spam examples (normal comments)
    not_spam_templates = [
        "bagus videonya", "keren kontennya", "mantap", "semangat",
        "informatif sekali", "terima kasih infonya", "bermanfaat",
        "setuju sama pendapatnya", "nice content", "good job",
        "kapan upload lagi", "ditunggu video selanjutnya", "the best",
        "suka banget sama channel ini", "selalu menginspirasi",
        "kecewa sih sama hasilnya", "kurang bagus menurutku",
        "agak membingungkan", "coba dijelaskan lebih detail",
        "pertanyaan dong", "mau tanya", "gimana caranya",
        "siapa nama lagunya", "lokasi dimana ini", "kapan eventnya",
        "relate banget", "sama kayak aku", "bener banget ini"
    ]
    for text in not_spam_templates:
        spam_texts.append(text)
        spam_labels.append('not_spam')
    
    return sentiment_texts, sentiment_labels, spam_texts, spam_labels

def train_classifiers():
    """Train Naive Bayes classifiers for sentiment and spam detection."""
    sentiment_texts, sentiment_labels, spam_texts, spam_labels = create_training_data()
    
    # Sentiment classifier
    sentiment_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('classifier', MultinomialNB(alpha=0.1))
    ])
    sentiment_pipeline.fit(sentiment_texts, sentiment_labels)
    
    # Spam classifier
    spam_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('classifier', MultinomialNB(alpha=0.1))
    ])
    spam_pipeline.fit(spam_texts, spam_labels)
    
    return sentiment_pipeline, spam_pipeline

# Initialize classifiers
print("Training Naive Bayes classifiers...")
SENTIMENT_CLASSIFIER, SPAM_CLASSIFIER = train_classifiers()
print("Classifiers ready!")

def classify_sentiment(text: str) -> tuple:
    """Classify text sentiment using Naive Bayes.
    Returns (label, confidence).
    """
    if not text or not text.strip():
        return ('neutral', 0.0)
    
    text_clean = clean_text(text)
    if not text_clean:
        return ('neutral', 0.0)
    
    # Get prediction and probability
    prediction = SENTIMENT_CLASSIFIER.predict([text_clean])[0]
    proba = SENTIMENT_CLASSIFIER.predict_proba([text_clean])[0]
    confidence = max(proba)
    
    # If confidence is low, also check lexicon
    tokens = set(text_clean.lower().split())
    pos_count = len(tokens & POSITIVE_WORDS)
    neg_count = len(tokens & NEGATIVE_WORDS)
    
    # Hybrid approach: combine model prediction with lexicon
    if confidence < 0.6:
        if pos_count > neg_count:
            return ('positive', 0.5 + (pos_count / (pos_count + neg_count + 1)) * 0.3)
        elif neg_count > pos_count:
            return ('negative', 0.5 + (neg_count / (pos_count + neg_count + 1)) * 0.3)
        else:
            return ('neutral', 0.5)
    
    return (prediction, round(confidence, 2))

def classify_spam(text: str) -> tuple:
    """Classify if text is spam using Naive Bayes.
    Returns (label, confidence).
    """
    if not text or not text.strip():
        return ('not_spam', 0.0)
    
    text_clean = clean_text(text)
    if not text_clean:
        return ('not_spam', 0.0)
    
    # Get prediction and probability
    prediction = SPAM_CLASSIFIER.predict([text_clean])[0]
    proba = SPAM_CLASSIFIER.predict_proba([text_clean])[0]
    confidence = max(proba)
    
    # Also check spam indicators
    text_lower = text.lower()
    spam_score = sum(1 for indicator in SPAM_INDICATORS if indicator in text_lower)
    
    # Hybrid approach
    if spam_score >= 3:
        return ('spam', min(0.9, 0.6 + spam_score * 0.05))
    elif confidence < 0.6 and spam_score >= 1:
        return ('spam', 0.55)
    
    return (prediction, round(confidence, 2))

# ===================== END CLASSIFIERS =====================

# ===================== SVM CLASSIFIERS =====================
def train_svm_classifiers():
    sentiment_texts, sentiment_labels, spam_texts, spam_labels = create_training_data()
    
    # ---------- SENTIMENT ----------
    sentiment_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('svm', LinearSVC())
    ])

    sentiment_svm = CalibratedClassifierCV(
        estimator=sentiment_pipeline,
        cv=5
    )
    sentiment_svm.fit(sentiment_texts, sentiment_labels)

    # ---------- SPAM ----------
    spam_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('svm', LinearSVC())
    ])

    spam_svm = CalibratedClassifierCV(
        estimator=spam_pipeline,
        cv=5
    )
    spam_svm.fit(spam_texts, spam_labels)

    return sentiment_svm, spam_svm


print("Training SVM classifiers...")
SENTIMENT_SVM, SPAM_SVM = train_svm_classifiers()
print("SVM ready!")

def classify_sentiment_svm(text: str):
    if not text or not text.strip():
        return ('neutral', 0.0)

    text_clean = clean_text(text)
    if not text_clean:
        return ('neutral', 0.0)

    label = SENTIMENT_SVM.predict([text_clean])[0]
    proba = SENTIMENT_SVM.predict_proba([text_clean])[0]

    return (label, round(float(max(proba)), 2))


def classify_spam_svm(text: str):
    if not text or not text.strip():
        return ('not_spam', 0.0)

    text_clean = clean_text(text)
    if not text_clean:
        return ('not_spam', 0.0)

    label = SPAM_SVM.predict([text_clean])[0]
    proba = SPAM_SVM.predict_proba([text_clean])[0]

    return (label, round(float(max(proba)), 2))
# ===================== END SVM CLASSIFIERS =====================

# ===================== ALL MODELS (3 Algorithms × 2 Feature Extractors) =====================

def train_all_models():
    """
    Train all combinations of:
    - Algorithms: Naive Bayes, Decision Tree, SVM
    - Feature Extraction: BoW (CountVectorizer), TF-IDF (TfidfVectorizer)
    
    Returns dict of trained models with cross-validation scores and evaluation metrics.
    """
    sentiment_texts, sentiment_labels, spam_texts, spam_labels = create_training_data()
    
    models = {}
    
    # Define classifiers
    classifiers = {
        'Naive Bayes': lambda: MultinomialNB(alpha=0.1),
        'Decision Tree': lambda: DecisionTreeClassifier(max_depth=10, random_state=42),
        'SVM': lambda: LinearSVC(random_state=42, max_iter=2000)
    }
    
    # Train for both tasks: sentiment and spam
    for task_name, (texts, labels) in [('sentiment', (sentiment_texts, sentiment_labels)), 
                                        ('spam', (spam_texts, spam_labels))]:
        
        # Get unique labels for this task
        unique_labels = list(set(labels))
        
        for vec_name in ['BoW', 'TF-IDF']:
            for clf_name, clf_factory in classifiers.items():
                model_key = f"{task_name}_{clf_name}_{vec_name}"
                
                # Create fresh vectorizer and classifier instances
                vec = CountVectorizer(ngram_range=(1, 2), max_features=5000) if vec_name == 'BoW' else TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
                clf = clf_factory()
                
                # Create pipeline
                pipeline = Pipeline([
                    ('vectorizer', vec),
                    ('classifier', clf)
                ])
                
                # Fit the model
                pipeline.fit(texts, labels)
                
                # Cross-validation predictions for evaluation metrics
                try:
                    cv_predictions = cross_val_predict(pipeline, texts, labels, cv=3)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(labels, cv_predictions)
                    precision = precision_score(labels, cv_predictions, average='weighted', zero_division=0)
                    recall = recall_score(labels, cv_predictions, average='weighted', zero_division=0)
                    f1 = f1_score(labels, cv_predictions, average='weighted', zero_division=0)
                    conf_matrix = confusion_matrix(labels, cv_predictions, labels=unique_labels)
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(pipeline, texts, labels, cv=3, scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except Exception as e:
                    print(f"  Warning: {model_key} - {e}")
                    accuracy = precision = recall = f1 = cv_mean = cv_std = 0.0
                    conf_matrix = np.zeros((len(unique_labels), len(unique_labels)))
                
                # For SVM, wrap with CalibratedClassifierCV to get probabilities
                if clf_name == 'SVM':
                    # Refit for calibration
                    vec2 = CountVectorizer(ngram_range=(1, 2), max_features=5000) if vec_name == 'BoW' else TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
                    clf2 = LinearSVC(random_state=42, max_iter=2000)
                    pipeline2 = Pipeline([('vectorizer', vec2), ('classifier', clf2)])
                    
                    calibrated = CalibratedClassifierCV(estimator=pipeline2, cv=5)
                    calibrated.fit(texts, labels)
                    final_model = calibrated
                else:
                    final_model = pipeline
                
                models[model_key] = {
                    'model': final_model,
                    'task': task_name,
                    'algorithm': clf_name,
                    'feature_extraction': vec_name,
                    'cv_accuracy': round(cv_mean * 100, 2),
                    'cv_std': round(cv_std * 100, 2),
                    'accuracy': round(accuracy * 100, 2),
                    'precision': round(precision * 100, 2),
                    'recall': round(recall * 100, 2),
                    'f1_score': round(f1 * 100, 2),
                    'confusion_matrix': conf_matrix.tolist(),
                    'labels': unique_labels,
                    'training_samples': len(texts)
                }
                
                print(f"  Trained: {model_key} (Acc: {accuracy*100:.1f}%, P: {precision*100:.1f}%, R: {recall*100:.1f}%, F1: {f1*100:.1f}%)")
    
    return models

print("\nTraining ALL model combinations (3 algorithms × 2 feature extractors × 2 tasks)...")
ALL_MODELS = train_all_models()
print(f"All {len(ALL_MODELS)} models ready!\n")

def generate_confusion_matrix_heatmap(conf_matrix, labels, title):
    """Generate confusion matrix heatmap as base64 image."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return img_base64

def generate_accuracy_comparison_chart(task='sentiment'):
    """Generate accuracy comparison bar chart for all models of a task."""
    models_data = [(k, v) for k, v in ALL_MODELS.items() if v['task'] == task]
    
    algorithms = ['Naive Bayes', 'Decision Tree', 'SVM']
    bow_scores = []
    tfidf_scores = []
    
    for algo in algorithms:
        for key, info in models_data:
            if info['algorithm'] == algo:
                if info['feature_extraction'] == 'BoW':
                    bow_scores.append(info['accuracy'])
                else:
                    tfidf_scores.append(info['accuracy'])
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, bow_scores, width, label='BoW', color='#17a2b8')
    bars2 = ax.bar(x + width/2, tfidf_scores, width, label='TF-IDF', color='#ffc107')
    
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{task.title()} Classification - Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.set_ylim(0, 105)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return img_base64

def generate_bow_vs_tfidf_chart():
    """Generate comparison chart between BoW and TF-IDF across all models."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Calculate averages for BoW and TF-IDF
    bow_avgs = []
    tfidf_avgs = []
    
    for metric in metrics:
        bow_values = [v[metric] for v in ALL_MODELS.values() if v['feature_extraction'] == 'BoW']
        tfidf_values = [v[metric] for v in ALL_MODELS.values() if v['feature_extraction'] == 'TF-IDF']
        bow_avgs.append(np.mean(bow_values))
        tfidf_avgs.append(np.mean(tfidf_values))
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, bow_avgs, width, label='BoW (Bag of Words)', color='#17a2b8')
    bars2 = ax.bar(x + width/2, tfidf_avgs, width, label='TF-IDF', color='#ffc107')
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score (%)')
    ax.set_title('BoW vs TF-IDF - Average Performance Across All Models')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 105)
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return img_base64

def generate_all_metrics_chart(task='sentiment'):
    """Generate grouped bar chart comparing all metrics for each model."""
    models_data = [(k, v) for k, v in ALL_MODELS.items() if v['task'] == task]
    
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for key, info in sorted(models_data, key=lambda x: (x[1]['algorithm'], x[1]['feature_extraction'])):
        model_names.append(f"{info['algorithm']}\n({info['feature_extraction']})")
        accuracies.append(info['accuracy'])
        precisions.append(info['precision'])
        recalls.append(info['recall'])
        f1_scores.append(info['f1_score'])
    
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#28a745')
    bars2 = ax.bar(x - 0.5*width, precisions, width, label='Precision', color='#17a2b8')
    bars3 = ax.bar(x + 0.5*width, recalls, width, label='Recall', color='#ffc107')
    bars4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#dc3545')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score (%)')
    ax.set_title(f'{task.title()} Classification - All Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=8)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return img_base64

def classify_with_model(text: str, task: str, algorithm: str, feature_extraction: str) -> tuple:
    """
    Classify text using a specific model combination.
    Returns (label, confidence).
    """
    if not text or not text.strip():
        default_label = 'neutral' if task == 'sentiment' else 'not_spam'
        return (default_label, 0.0)
    
    text_clean = clean_text(text)
    if not text_clean:
        default_label = 'neutral' if task == 'sentiment' else 'not_spam'
        return (default_label, 0.0)
    
    model_key = f"{task}_{algorithm}_{feature_extraction}"
    
    if model_key not in ALL_MODELS:
        return ('unknown', 0.0)
    
    model_info = ALL_MODELS[model_key]
    model = model_info['model']
    
    # Predict
    label = model.predict([text_clean])[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba([text_clean])[0]
        confidence = float(max(proba))
    else:
        confidence = 1.0  # No probability available
    
    return (label, round(confidence, 2))

def get_model_comparison():
    """Get comparison data for all models."""
    comparison = []
    for key, info in ALL_MODELS.items():
        comparison.append({
            'model_key': key,
            'task': info['task'].title(),
            'algorithm': info['algorithm'],
            'feature_extraction': info['feature_extraction'],
            'cv_accuracy': info['cv_accuracy'],
            'cv_std': info['cv_std'],
            'accuracy': info['accuracy'],
            'precision': info['precision'],
            'recall': info['recall'],
            'f1_score': info['f1_score'],
            'training_samples': info['training_samples']
        })
    return comparison

def get_best_models():
    """Get the best model for each task based on F1-score."""
    best_sentiment = None
    best_spam = None
    best_sentiment_score = -1
    best_spam_score = -1
    
    for key, info in ALL_MODELS.items():
        if info['task'] == 'sentiment' and info['f1_score'] > best_sentiment_score:
            best_sentiment_score = info['f1_score']
            best_sentiment = {
                'key': key,
                'algorithm': info['algorithm'],
                'feature_extraction': info['feature_extraction'],
                'model': info['model'],
                'accuracy': info['accuracy'],
                'precision': info['precision'],
                'recall': info['recall'],
                'f1_score': info['f1_score']
            }
        elif info['task'] == 'spam' and info['f1_score'] > best_spam_score:
            best_spam_score = info['f1_score']
            best_spam = {
                'key': key,
                'algorithm': info['algorithm'],
                'feature_extraction': info['feature_extraction'],
                'model': info['model'],
                'accuracy': info['accuracy'],
                'precision': info['precision'],
                'recall': info['recall'],
                'f1_score': info['f1_score']
            }
    
    return best_sentiment, best_spam

# Get best models globally
BEST_SENTIMENT_MODEL, BEST_SPAM_MODEL = get_best_models()
print(f"Best Sentiment Model: {BEST_SENTIMENT_MODEL['algorithm']} + {BEST_SENTIMENT_MODEL['feature_extraction']} (F1: {BEST_SENTIMENT_MODEL['f1_score']}%)")
print(f"Best Spam Model: {BEST_SPAM_MODEL['algorithm']} + {BEST_SPAM_MODEL['feature_extraction']} (F1: {BEST_SPAM_MODEL['f1_score']}%)")

def predict_with_best_model(text: str):
    """
    Predict sentiment and spam using the best models.
    Returns dict with predictions and confidence.
    """
    if not text or not text.strip():
        return {
            'sentiment': {'label': 'neutral', 'confidence': 0.0},
            'spam': {'label': 'not_spam', 'confidence': 0.0},
            'cleaned_text': ''
        }
    
    text_clean = clean_text(text)
    if not text_clean:
        return {
            'sentiment': {'label': 'neutral', 'confidence': 0.0},
            'spam': {'label': 'not_spam', 'confidence': 0.0},
            'cleaned_text': ''
        }
    
    # Sentiment prediction
    sent_model = BEST_SENTIMENT_MODEL['model']
    sent_label = sent_model.predict([text_clean])[0]
    if hasattr(sent_model, 'predict_proba'):
        sent_proba = sent_model.predict_proba([text_clean])[0]
        sent_conf = float(max(sent_proba))
    else:
        sent_conf = 1.0
    
    # Spam prediction
    spam_model = BEST_SPAM_MODEL['model']
    spam_label = spam_model.predict([text_clean])[0]
    if hasattr(spam_model, 'predict_proba'):
        spam_proba = spam_model.predict_proba([text_clean])[0]
        spam_conf = float(max(spam_proba))
    else:
        spam_conf = 1.0
    
    return {
        'sentiment': {'label': sent_label, 'confidence': round(sent_conf, 4)},
        'spam': {'label': spam_label, 'confidence': round(spam_conf, 4)},
        'cleaned_text': text_clean
    }

# ===================== END ALL MODELS =====================

EMOJI_PATTERN = re.compile("["
													 "\U0001F600-\U0001F64F"  # emoticons
													 "\U0001F300-\U0001F5FF"  # symbols & pictographs
													 "\U0001F680-\U0001F6FF"  # transport & map symbols
													 "\U0001F1E0-\U0001F1FF"  # flags (iOS)
													 "\u2600-\u26FF\u2700-\u27BF]+", flags=re.UNICODE)

def clean_text(text: str) -> str:
		if pd.isna(text):
				return ""
		# convert to str
		text = str(text)
		# remove urls
		text = re.sub(r'http\S+|www\.\S+', ' ', text)
		# remove emojis
		text = EMOJI_PATTERN.sub(' ', text)
		# remove mentions and hashtags
		text = re.sub(r'[@#]\w+', ' ', text)
		# remove non-letter characters (keep unicode letters)
		text = re.sub(r'[^\w\s]', ' ', text)
		# remove digits
		text = re.sub(r'\d+', ' ', text)
		# collapse whitespace
		text = re.sub(r'\s+', ' ', text).strip()
		return text.lower()

def tokenize_and_filter(text: str):
		if not text:
				return []
		try:
				tokens = word_tokenize(text)
		except Exception:
				# fallback simple split
				tokens = text.split()
		# normalize tokens
		tokens = [t.lower() for t in tokens if len(t) > 1]
		# remove stopwords
		tokens = [t for t in tokens if t not in STOPWORDS]
		return tokens


def df_to_html_with_links(df: pd.DataFrame, max_rows: int | None = None) -> str:
	"""Return HTML for DataFrame where any column containing 'url' is rendered as a clickable link.
	If max_rows is provided, use head(max_rows).
	"""
	if max_rows:
		df = df.head(max_rows)
	else:
		df = df.copy()
	# work on a copy to avoid mutating original
	df = df.copy()
	for col in df.columns:
		if 'url' in str(col).lower():
			# replace with anchor tags; keep text as the url itself per request
			df[col] = df[col].fillna('').apply(lambda u: f'<a href="{u}" target="_blank" rel="noopener noreferrer">url</a>' if str(u).strip() else '')
	# render without escaping so anchors are active
	return df.to_html(classes='table table-sm table-striped', index=False, escape=False)

INDEX_HTML = '''
<!doctype html>
<html lang="en">
	<head>
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<title>NLP Preprocessing</title>
		<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
	</head>
	<body class="bg-light">
		<div class="container py-4">
			<h1 class="mb-3">NLP Preprocessing - Simple Web UI</h1>
			<p class="text-muted">Pilih kolom teks dari dataset `dataNew.csv` untuk memproses (comments, caption, atau kolom lain).</p>

			{% if error %}
				<div class="alert alert-danger">{{ error }}</div>
			{% endif %}

			<form method="post" action="/preprocess" class="row g-3 mb-4">
				<div class="col-md-4">
					<label class="form-label">CSV Path</label>
					<input name="csv_path" class="form-control" value="dataNew.csv">
				</div>
				<div class="col-md-4">
					<label class="form-label">Text Column</label>
					<input name="text_col" class="form-control" value="comments">
				</div>
				<div class="col-md-2">
					<label class="form-label">Rows to show</label>
					<input name="n_rows" type="number" min="1" max="100000" class="form-control" value="{{ rows_to_show or 20 }}">
				</div>
				<div class="col-md-2 align-self-end">
					<button class="btn btn-primary w-100">Preprocess</button>
				</div>
			</form>

			<div class="mb-3">
				<form method="get" action="/show_all" class="d-inline">
					<button class="btn btn-outline-secondary">Show All Data</button>
				</form>
				<a href="/features" class="btn btn-outline-info ms-2">Feature Extraction (BoW/TF-IDF)</a>
				<a href="/models" class="btn btn-outline-success ms-2">Model Comparison (All 12 Models)</a>
				<a href="/models/evaluation" class="btn btn-outline-warning ms-2">Model Evaluation & Metrics</a>
				<a href="/predict" class="btn btn-primary ms-2">🔮 Predict (Best Model)</a>
			</div>

			{% if table_html %}
				<h5>Data Preview (original)</h5>
				<div class="table-responsive">{{ table_html | safe }}</div>
			{% endif %}

			<hr>
			<footer class="text-muted small">If NLTK data is missing, server will attempt to download it once.</footer>
		</div>
	</body>
</html>
'''

RESULT_HTML = '''
<!doctype html>
<html lang="en">
	<head>
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1">
		<title>Preprocessing Result</title>
		<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
		<style>
			.badge-positive { background-color: #198754; }
			.badge-negative { background-color: #dc3545; }
			.badge-neutral { background-color: #6c757d; }
			.badge-spam { background-color: #fd7e14; }
			.badge-not_spam { background-color: #20c997; }
		</style>
	</head>
	<body class="bg-light">
		<div class="container py-4">
			<h1 class="mb-3">Preprocessing & Classification Result</h1>
			<a href="/" class="btn btn-secondary mb-3">&larr; Back</a>

			<a href="/comments?csv_path={{ csv_path }}&text_col={{ text_col }}"
				class="btn btn-outline-primary mb-3 ms-2">
				View Per-Comment Analysis
			</a>
   
			<a href="/svm?csv_path={{ csv_path }}&text_col={{ text_col }}"
				class="btn btn-outline-success mb-3 ms-2">
				SVM Classification
			</a>



			{% if stats %}
			<div class="row mb-4">
				<div class="col-md-6">
					<div class="card">
						<div class="card-header"><strong>Sentiment Analysis (Naive Bayes)</strong></div>
						<div class="card-body">
							<div class="d-flex justify-content-around text-center">
								<div>
									<h3 class="text-success">{{ stats.positive }}</h3>
									<small>Positive</small>
								</div>
								<div>
									<h3 class="text-danger">{{ stats.negative }}</h3>
									<small>Negative</small>
								</div>
								<div>
									<h3 class="text-secondary">{{ stats.neutral }}</h3>
									<small>Neutral</small>
								</div>
							</div>
						</div>
					</div>
				</div>
				<div class="col-md-6">
					<div class="card">
						<div class="card-header"><strong>Spam Detection (Naive Bayes)</strong></div>
						<div class="card-body">
							<div class="d-flex justify-content-around text-center">
								<div>
									<h3 class="text-warning">{{ stats.spam }}</h3>
									<small>Spam</small>
								</div>
								<div>
									<h3 class="text-info">{{ stats.not_spam }}</h3>
									<small>Not Spam</small>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
			{% endif %}
   
			{% if svm_stats %}
			<hr>

			<h3>Support Vector Machine (SVM) Result</h3>

			<table border="1" cellpadding="8">
				<tr>
					<th>Task</th>
					<th>Prediction</th>
					<th>Confidence</th>
				</tr>
				<tr>
					<td>Sentiment</td>
					<td>{{ svm_stats.sentiment }}</td>
					<td>{{ svm_stats.sentiment_conf }}%</td>
				</tr>
				<tr>
					<td>Spam</td>
					<td>{{ svm_stats.spam }}</td>
					<td>{{ svm_stats.spam_conf }}%</td>
				</tr>
			</table>
			{% endif %}

			<h5>Processed (first {{ n_rows }} rows)</h5>
			<div class="d-flex justify-content-between align-items-center mb-2">
				<div>Page {{ page }} / {{ total_pages }}</div>
				<div>
					<form method="get" action="/preprocess" class="d-inline">
						<input type="hidden" name="csv_path" value="{{ csv_path }}">
						<input type="hidden" name="text_col" value="{{ text_col }}">
						<input type="hidden" name="page_size" value="{{ page_size }}">
						<button name="page" value="{{ 1 }}" class="btn btn-sm btn-outline-secondary">First</button>
						{% if page > 1 %}
						<button name="page" value="{{ page-1 }}" class="btn btn-sm btn-outline-secondary">Prev</button>
						{% endif %}
						{% if page < total_pages %}
						<button name="page" value="{{ page+1 }}" class="btn btn-sm btn-outline-secondary">Next</button>
						{% endif %}
						<button name="page" value="{{ total_pages }}" class="btn btn-sm btn-outline-secondary">Last</button>
					</form>
				</div>
			</div>
			<div class="table-responsive">{{ result_table | safe }}</div>

			<hr>
		</div>
	</body>
</html>
'''

COMMENT_HTML = '''
<!doctype html>
<html>
<head>
  <title>Per Comment Classification</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    .badge-positive { background:#198754 }
    .badge-negative { background:#dc3545 }
    .badge-neutral { background:#6c757d }
    .badge-spam { background:#fd7e14 }
    .badge-not_spam { background:#20c997 }
  </style>
</head>
<body class="bg-light">
<div class="container py-4">

<h2>Naive Bayes Per Comment Classification</h2>

<a href="/" class="btn btn-secondary mb-3">&larr; Back</a>

<a href="/svm?csv_path={{ csv_path }}&text_col={{ text_col }}"
   class="btn btn-outline-primary mb-3 ms-2">
   SVM View
</a>

<div class="d-flex justify-content-between mb-2">
  <div>Page {{ page }} / {{ total_pages }}</div>
  <div>
    <a class="btn btn-sm btn-outline-secondary"
       href="/comments?csv_path={{ csv_path }}&text_col={{ text_col }}&page=1&page_size={{ page_size }}">First</a>

    {% if page > 1 %}
    <a class="btn btn-sm btn-outline-secondary"
       href="/comments?csv_path={{ csv_path }}&text_col={{ text_col }}&page={{ page-1 }}&page_size={{ page_size }}">Prev</a>
    {% endif %}

    {% if page < total_pages %}
    <a class="btn btn-sm btn-outline-secondary"
       href="/comments?csv_path={{ csv_path }}&text_col={{ text_col }}&page={{ page+1 }}&page_size={{ page_size }}">Next</a>
    {% endif %}
  </div>
</div>

<div class="table-responsive">
  {{ table_html | safe }}
</div>

</div>
</body>
</html>
'''

SVM_HTML = '''
<!doctype html>
<html>
<head>
  <title>SVM Comment Classification</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    .badge-positive { background:#198754 }
    .badge-negative { background:#dc3545 }
    .badge-neutral { background:#6c757d }
    .badge-spam { background:#fd7e14 }
    .badge-not_spam { background:#20c997 }
  </style>
</head>
<body class="bg-light">
<div class="container py-4">

<h2>SVM Per-Comment Classification</h2>

<a href="/" class="btn btn-secondary mb-3">&larr; Back</a>

<a href="/comments?csv_path={{ csv_path }}&text_col={{ text_col }}"
   class="btn btn-outline-primary mb-3 ms-2">
   Naive Bayes View
</a>

<div class="d-flex justify-content-between mb-2">
  <div>Page {{ page }} / {{ total_pages }}</div>
  <div>
    <a class="btn btn-sm btn-outline-secondary"
       href="/svm?csv_path={{ csv_path }}&text_col={{ text_col }}&page=1&page_size={{ page_size }}">First</a>

    {% if page > 1 %}
    <a class="btn btn-sm btn-outline-secondary"
       href="/svm?csv_path={{ csv_path }}&text_col={{ text_col }}&page={{ page-1 }}&page_size={{ page_size }}">Prev</a>
    {% endif %}

    {% if page < total_pages %}
    <a class="btn btn-sm btn-outline-secondary"
       href="/svm?csv_path={{ csv_path }}&text_col={{ text_col }}&page={{ page+1 }}&page_size={{ page_size }}">Next</a>
    {% endif %}
  </div>
</div>

<div class="table-responsive">
  {{ table_html | safe }}
</div>

</div>
</body>
</html>
'''

FEATURES_HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Feature Extraction - {{ method }}</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    .feature-matrix { font-size: 0.85rem; }
    .feature-matrix th { position: sticky; top: 0; background: #f8f9fa; }
    .vocab-list { max-height: 300px; overflow-y: auto; }
    .highlight { background-color: #fff3cd; }
  </style>
</head>
<body class="bg-light">
<div class="container-fluid py-4">

<h2>Feature Extraction Output</h2>
<p class="text-muted">Raw vectorized representation (not trained yet)</p>

<a href="/" class="btn btn-secondary mb-3">&larr; Back</a>

<div class="row mb-4">
  <div class="col-12">
    <div class="card">
      <div class="card-header">
        <strong>Select Method & Data</strong>
      </div>
      <div class="card-body">
        <form method="get" action="/features" class="row g-3">
          <div class="col-md-3">
            <label class="form-label">CSV Path</label>
            <input name="csv_path" class="form-control" value="{{ csv_path }}">
          </div>
          <div class="col-md-2">
            <label class="form-label">Text Column</label>
            <input name="text_col" class="form-control" value="{{ text_col }}">
          </div>
          <div class="col-md-2">
            <label class="form-label">Method</label>
            <select name="method" class="form-select">
              <option value="bow" {% if method == 'bow' %}selected{% endif %}>Bag of Words (BoW)</option>
              <option value="tfidf" {% if method == 'tfidf' %}selected{% endif %}>TF-IDF</option>
            </select>
          </div>
          <div class="col-md-2">
            <label class="form-label">Max Features</label>
            <input name="max_features" type="number" min="10" max="1000" class="form-control" value="{{ max_features }}">
          </div>
          <div class="col-md-1">
            <label class="form-label">N-gram</label>
            <select name="ngram" class="form-select">
              <option value="1" {% if ngram == 1 %}selected{% endif %}>1</option>
              <option value="2" {% if ngram == 2 %}selected{% endif %}>1-2</option>
              <option value="3" {% if ngram == 3 %}selected{% endif %}>1-3</option>
            </select>
          </div>
          <div class="col-md-2 align-self-end">
            <button class="btn btn-primary w-100">Extract Features</button>
          </div>
        </form>
      </div>
    </div>
  </div>
</div>

{% if error %}
  <div class="alert alert-danger">{{ error }}</div>
{% endif %}

{% if vocab %}
<div class="row mb-4">
  <div class="col-md-4">
    <div class="card">
      <div class="card-header"><strong>Vocabulary ({{ vocab|length }} features)</strong></div>
      <div class="card-body vocab-list">
        <table class="table table-sm">
          <thead><tr><th>Index</th><th>Term</th></tr></thead>
          <tbody>
          {% for idx, term in vocab %}
            <tr><td>{{ idx }}</td><td><code>{{ term }}</code></td></tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>
  <div class="col-md-8">
    <div class="card">
      <div class="card-header">
        <strong>{{ method_name }} Matrix</strong>
        <span class="text-muted">({{ matrix_shape.0 }} documents × {{ matrix_shape.1 }} features)</span>
      </div>
      <div class="card-body">
        <p class="small text-muted mb-2">
          {% if method == 'bow' %}
            Values = word counts per document
          {% else %}
            Values = TF-IDF weights (higher = more important in that document)
          {% endif %}
        </p>
        <div class="table-responsive" style="max-height: 500px; overflow: auto;">
          {{ matrix_html | safe }}
        </div>
      </div>
    </div>
  </div>
</div>

<div class="row mb-4">
  <div class="col-12">
    <div class="card">
      <div class="card-header"><strong>Sample Documents (first {{ sample_docs|length }})</strong></div>
      <div class="card-body">
        <table class="table table-sm table-striped">
          <thead><tr><th>Doc #</th><th>Original Text</th><th>Cleaned Text</th></tr></thead>
          <tbody>
          {% for doc in sample_docs %}
            <tr>
              <td>{{ doc.idx }}</td>
              <td style="max-width:300px; overflow:hidden; text-overflow:ellipsis;">{{ doc.original[:100] }}{% if doc.original|length > 100 %}...{% endif %}</td>
              <td style="max-width:300px; overflow:hidden; text-overflow:ellipsis;">{{ doc.cleaned[:100] }}{% if doc.cleaned|length > 100 %}...{% endif %}</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>
{% endif %}

</div>
</body>
</html>
'''

MODELS_HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Comparison</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    .badge-positive { background:#198754 }
    .badge-negative { background:#dc3545 }
    .badge-neutral { background:#6c757d }
    .badge-spam { background:#fd7e14 }
    .badge-not_spam { background:#20c997 }
    .best-score { background-color: #d4edda !important; font-weight: bold; }
    .model-card { transition: transform 0.2s; }
    .model-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
  </style>
</head>
<body class="bg-light">
<div class="container-fluid py-4">

<h2>Model Comparison</h2>
<p class="text-muted">All trained models: 3 Algorithms × 2 Feature Extractors × 2 Tasks = 12 Models</p>

<a href="/" class="btn btn-secondary mb-3">&larr; Back</a>
<a href="/models/test" class="btn btn-primary mb-3 ms-2">Test Models with Custom Text</a>
<a href="/models/evaluation" class="btn btn-success mb-3 ms-2">Model Evaluation & Metrics</a>

<!-- Model Comparison Tables -->
<div class="row mb-4">
  <div class="col-12">
    <div class="card">
      <div class="card-header"><strong>Sentiment Classification Models</strong></div>
      <div class="card-body">
        <table class="table table-sm table-striped table-hover">
          <thead class="table-dark">
            <tr>
              <th>Algorithm</th>
              <th>Feature Extraction</th>
              <th>CV Accuracy (%)</th>
              <th>Std Dev</th>
              <th>Training Samples</th>
            </tr>
          </thead>
          <tbody>
          {% for model in sentiment_models %}
            <tr class="{% if model.is_best %}best-score{% endif %}">
              <td><strong>{{ model.algorithm }}</strong></td>
              <td><span class="badge {% if model.feature_extraction == 'BoW' %}bg-info{% else %}bg-warning text-dark{% endif %}">{{ model.feature_extraction }}</span></td>
              <td>{{ model.cv_accuracy }}%</td>
              <td>±{{ model.cv_std }}%</td>
              <td>{{ model.training_samples }}</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<div class="row mb-4">
  <div class="col-12">
    <div class="card">
      <div class="card-header"><strong>Spam Detection Models</strong></div>
      <div class="card-body">
        <table class="table table-sm table-striped table-hover">
          <thead class="table-dark">
            <tr>
              <th>Algorithm</th>
              <th>Feature Extraction</th>
              <th>CV Accuracy (%)</th>
              <th>Std Dev</th>
              <th>Training Samples</th>
            </tr>
          </thead>
          <tbody>
          {% for model in spam_models %}
            <tr class="{% if model.is_best %}best-score{% endif %}">
              <td><strong>{{ model.algorithm }}</strong></td>
              <td><span class="badge {% if model.feature_extraction == 'BoW' %}bg-info{% else %}bg-warning text-dark{% endif %}">{{ model.feature_extraction }}</span></td>
              <td>{{ model.cv_accuracy }}%</td>
              <td>±{{ model.cv_std }}%</td>
              <td>{{ model.training_samples }}</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<!-- Summary Cards -->
<div class="row mb-4">
  <div class="col-md-6">
    <div class="card model-card border-success">
      <div class="card-header bg-success text-white"><strong>Best Sentiment Model</strong></div>
      <div class="card-body">
        <h5>{{ best_sentiment.algorithm }} + {{ best_sentiment.feature_extraction }}</h5>
        <p class="mb-0">Accuracy: <strong>{{ best_sentiment.cv_accuracy }}%</strong> (±{{ best_sentiment.cv_std }}%)</p>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card model-card border-warning">
      <div class="card-header bg-warning"><strong>Best Spam Model</strong></div>
      <div class="card-body">
        <h5>{{ best_spam.algorithm }} + {{ best_spam.feature_extraction }}</h5>
        <p class="mb-0">Accuracy: <strong>{{ best_spam.cv_accuracy }}%</strong> (±{{ best_spam.cv_std }}%)</p>
      </div>
    </div>
  </div>
</div>

<!-- Algorithm Explanation -->
<div class="row">
  <div class="col-12">
    <div class="card">
      <div class="card-header"><strong>Model Details</strong></div>
      <div class="card-body">
        <div class="row">
          <div class="col-md-4">
            <h6>Algorithms</h6>
            <ul class="small">
              <li><strong>Naive Bayes</strong>: Probabilistic classifier, fast, works well with text</li>
              <li><strong>Decision Tree</strong>: Rule-based, interpretable, max_depth=10</li>
              <li><strong>SVM</strong>: Linear SVC, good for high-dimensional text data</li>
            </ul>
          </div>
          <div class="col-md-4">
            <h6>Feature Extraction</h6>
            <ul class="small">
              <li><strong>BoW (Bag of Words)</strong>: Raw word counts using CountVectorizer</li>
              <li><strong>TF-IDF</strong>: Term frequency-inverse document frequency weights</li>
            </ul>
          </div>
          <div class="col-md-4">
            <h6>Evaluation</h6>
            <ul class="small">
              <li><strong>CV Accuracy</strong>: 3-fold cross-validation mean</li>
              <li><strong>Std Dev</strong>: Variation across folds</li>
              <li>Green row = best model for that task</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

</div>
</body>
</html>
'''

MODELS_TEST_HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Test Models</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    .badge-positive { background:#198754 }
    .badge-negative { background:#dc3545 }
    .badge-neutral { background:#6c757d }
    .badge-spam { background:#fd7e14 }
    .badge-not_spam { background:#20c997 }
  </style>
</head>
<body class="bg-light">
<div class="container py-4">

<h2>Test All Models</h2>
<p class="text-muted">Compare predictions from all 12 model combinations</p>

<a href="/models" class="btn btn-secondary mb-3">&larr; Back to Comparison</a>

<div class="card mb-4">
  <div class="card-header"><strong>Enter Text to Classify</strong></div>
  <div class="card-body">
    <form method="post" action="/models/test">
      <div class="mb-3">
        <textarea name="text" class="form-control" rows="3" placeholder="Enter Indonesian text to classify...">{{ test_text or '' }}</textarea>
      </div>
      <button class="btn btn-primary">Classify with All Models</button>
    </form>
  </div>
</div>

{% if results %}
<div class="row">
  <div class="col-md-6">
    <div class="card mb-4">
      <div class="card-header bg-primary text-white"><strong>Sentiment Classification Results</strong></div>
      <div class="card-body">
        <table class="table table-sm table-striped">
          <thead>
            <tr><th>Algorithm</th><th>Features</th><th>Prediction</th><th>Confidence</th></tr>
          </thead>
          <tbody>
          {% for r in results.sentiment %}
            <tr>
              <td>{{ r.algorithm }}</td>
              <td><span class="badge {% if r.feature_extraction == 'BoW' %}bg-info{% else %}bg-warning text-dark{% endif %}">{{ r.feature_extraction }}</span></td>
              <td><span class="badge badge-{{ r.label }}">{{ r.label }}</span></td>
              <td>{{ (r.confidence * 100)|round(1) }}%</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>
  <div class="col-md-6">
    <div class="card mb-4">
      <div class="card-header bg-warning"><strong>Spam Detection Results</strong></div>
      <div class="card-body">
        <table class="table table-sm table-striped">
          <thead>
            <tr><th>Algorithm</th><th>Features</th><th>Prediction</th><th>Confidence</th></tr>
          </thead>
          <tbody>
          {% for r in results.spam %}
            <tr>
              <td>{{ r.algorithm }}</td>
              <td><span class="badge {% if r.feature_extraction == 'BoW' %}bg-info{% else %}bg-warning text-dark{% endif %}">{{ r.feature_extraction }}</span></td>
              <td><span class="badge badge-{{ r.label }}">{{ r.label }}</span></td>
              <td>{{ (r.confidence * 100)|round(1) }}%</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
  </div>
</div>

<div class="card">
  <div class="card-header"><strong>Cleaned Text</strong></div>
  <div class="card-body">
    <code>{{ cleaned_text }}</code>
  </div>
</div>
{% endif %}

</div>
</body>
</html>
'''

EVALUATION_HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model Evaluation</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    .metric-card { text-align: center; }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .best-model { border: 3px solid #28a745 !important; }
    .chart-container { background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
  </style>
</head>
<body class="bg-light">
<div class="container-fluid py-4">

<h2>Model Evaluation & Metrics</h2>
<p class="text-muted">Comprehensive evaluation with Accuracy, Precision, Recall, F1-Score, and Confusion Matrix</p>

<a href="/models" class="btn btn-secondary mb-3">&larr; Back to Models</a>

<!-- Task Selection -->
<ul class="nav nav-tabs mb-4" id="taskTabs" role="tablist">
  <li class="nav-item" role="presentation">
    <button class="nav-link active" id="sentiment-tab" data-bs-toggle="tab" data-bs-target="#sentiment" type="button" role="tab">Sentiment Classification</button>
  </li>
  <li class="nav-item" role="presentation">
    <button class="nav-link" id="spam-tab" data-bs-toggle="tab" data-bs-target="#spam" type="button" role="tab">Spam Detection</button>
  </li>
  <li class="nav-item" role="presentation">
    <button class="nav-link" id="comparison-tab" data-bs-toggle="tab" data-bs-target="#comparison" type="button" role="tab">BoW vs TF-IDF</button>
  </li>
</ul>

<div class="tab-content" id="taskTabsContent">
  <!-- Sentiment Tab -->
  <div class="tab-pane fade show active" id="sentiment" role="tabpanel">
    <h4>Sentiment Classification Models</h4>
    
    <!-- Metrics Table -->
    <div class="card mb-4">
      <div class="card-header"><strong>Evaluation Metrics</strong></div>
      <div class="card-body">
        <table class="table table-sm table-striped table-hover">
          <thead class="table-dark">
            <tr>
              <th>Algorithm</th>
              <th>Features</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
            </tr>
          </thead>
          <tbody>
          {% for m in sentiment_models %}
            <tr class="{% if m.is_best %}table-success{% endif %}">
              <td><strong>{{ m.algorithm }}</strong></td>
              <td><span class="badge {% if m.feature_extraction == 'BoW' %}bg-info{% else %}bg-warning text-dark{% endif %}">{{ m.feature_extraction }}</span></td>
              <td>{{ m.accuracy }}%</td>
              <td>{{ m.precision }}%</td>
              <td>{{ m.recall }}%</td>
              <td>{{ m.f1_score }}%</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
    
    <!-- Charts Row -->
    <div class="row mb-4">
      <div class="col-12">
        <div class="card">
          <div class="card-header"><strong>Accuracy Comparison</strong></div>
          <div class="card-body chart-container text-center">
            <img src="data:image/png;base64,{{ sentiment_accuracy_chart }}" class="img-fluid" alt="Sentiment Accuracy Comparison">
          </div>
        </div>
      </div>
    </div>
    
    <div class="row mb-4">
      <div class="col-12">
        <div class="card">
          <div class="card-header"><strong>All Metrics Comparison</strong></div>
          <div class="card-body chart-container text-center">
            <img src="data:image/png;base64,{{ sentiment_metrics_chart }}" class="img-fluid" alt="Sentiment All Metrics">
          </div>
        </div>
      </div>
    </div>
    
    <!-- Confusion Matrices -->
    <h5 class="mt-4">Confusion Matrices</h5>
    <div class="row">
      {% for m in sentiment_models %}
      <div class="col-md-4 mb-3">
        <div class="card {% if m.is_best %}best-model{% endif %}">
          <div class="card-header">
            <strong>{{ m.algorithm }}</strong> ({{ m.feature_extraction }})
            {% if m.is_best %}<span class="badge bg-success ms-2">Best</span>{% endif %}
          </div>
          <div class="card-body text-center">
            <img src="data:image/png;base64,{{ m.cm_image }}" class="img-fluid" alt="Confusion Matrix">
          </div>
        </div>
      </div>
      {% endfor %}
    </div>
  </div>
  
  <!-- Spam Tab -->
  <div class="tab-pane fade" id="spam" role="tabpanel">
    <h4>Spam Detection Models</h4>
    
    <!-- Metrics Table -->
    <div class="card mb-4">
      <div class="card-header"><strong>Evaluation Metrics</strong></div>
      <div class="card-body">
        <table class="table table-sm table-striped table-hover">
          <thead class="table-dark">
            <tr>
              <th>Algorithm</th>
              <th>Features</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1-Score</th>
            </tr>
          </thead>
          <tbody>
          {% for m in spam_models %}
            <tr class="{% if m.is_best %}table-success{% endif %}">
              <td><strong>{{ m.algorithm }}</strong></td>
              <td><span class="badge {% if m.feature_extraction == 'BoW' %}bg-info{% else %}bg-warning text-dark{% endif %}">{{ m.feature_extraction }}</span></td>
              <td>{{ m.accuracy }}%</td>
              <td>{{ m.precision }}%</td>
              <td>{{ m.recall }}%</td>
              <td>{{ m.f1_score }}%</td>
            </tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    </div>
    
    <!-- Charts Row -->
    <div class="row mb-4">
      <div class="col-12">
        <div class="card">
          <div class="card-header"><strong>Accuracy Comparison</strong></div>
          <div class="card-body chart-container text-center">
            <img src="data:image/png;base64,{{ spam_accuracy_chart }}" class="img-fluid" alt="Spam Accuracy Comparison">
          </div>
        </div>
      </div>
    </div>
    
    <div class="row mb-4">
      <div class="col-12">
        <div class="card">
          <div class="card-header"><strong>All Metrics Comparison</strong></div>
          <div class="card-body chart-container text-center">
            <img src="data:image/png;base64,{{ spam_metrics_chart }}" class="img-fluid" alt="Spam All Metrics">
          </div>
        </div>
      </div>
    </div>
    
    <!-- Confusion Matrices -->
    <h5 class="mt-4">Confusion Matrices</h5>
    <div class="row">
      {% for m in spam_models %}
      <div class="col-md-4 mb-3">
        <div class="card {% if m.is_best %}best-model{% endif %}">
          <div class="card-header">
            <strong>{{ m.algorithm }}</strong> ({{ m.feature_extraction }})
            {% if m.is_best %}<span class="badge bg-success ms-2">Best</span>{% endif %}
          </div>
          <div class="card-body text-center">
            <img src="data:image/png;base64,{{ m.cm_image }}" class="img-fluid" alt="Confusion Matrix">
          </div>
        </div>
      </div>
      {% endfor %}
    </div>
  </div>
  
  <!-- BoW vs TF-IDF Comparison Tab -->
  <div class="tab-pane fade" id="comparison" role="tabpanel">
    <h4>BoW vs TF-IDF Feature Extraction Comparison</h4>
    
    <div class="row mb-4">
      <div class="col-12">
        <div class="card">
          <div class="card-header"><strong>Average Performance Across All Models</strong></div>
          <div class="card-body chart-container text-center">
            <img src="data:image/png;base64,{{ bow_vs_tfidf_chart }}" class="img-fluid" alt="BoW vs TF-IDF Comparison">
          </div>
        </div>
      </div>
    </div>
    
    <div class="row">
      <div class="col-md-6">
        <div class="card">
          <div class="card-header bg-info text-white"><strong>Bag of Words (BoW) Summary</strong></div>
          <div class="card-body">
            <table class="table table-sm">
              <tr><td>Average Accuracy</td><td><strong>{{ bow_stats.accuracy }}%</strong></td></tr>
              <tr><td>Average Precision</td><td><strong>{{ bow_stats.precision }}%</strong></td></tr>
              <tr><td>Average Recall</td><td><strong>{{ bow_stats.recall }}%</strong></td></tr>
              <tr><td>Average F1-Score</td><td><strong>{{ bow_stats.f1_score }}%</strong></td></tr>
            </table>
          </div>
        </div>
      </div>
      <div class="col-md-6">
        <div class="card">
          <div class="card-header bg-warning"><strong>TF-IDF Summary</strong></div>
          <div class="card-body">
            <table class="table table-sm">
              <tr><td>Average Accuracy</td><td><strong>{{ tfidf_stats.accuracy }}%</strong></td></tr>
              <tr><td>Average Precision</td><td><strong>{{ tfidf_stats.precision }}%</strong></td></tr>
              <tr><td>Average Recall</td><td><strong>{{ tfidf_stats.recall }}%</strong></td></tr>
              <tr><td>Average F1-Score</td><td><strong>{{ tfidf_stats.f1_score }}%</strong></td></tr>
            </table>
          </div>
        </div>
      </div>
    </div>
    
    <div class="row mt-4">
      <div class="col-12">
        <div class="card">
          <div class="card-header"><strong>Winner: {{ winner }}</strong></div>
          <div class="card-body">
            <p>Based on average F1-Score across all models, <strong>{{ winner }}</strong> performs better for this dataset.</p>
            <ul>
              <li><strong>BoW (Bag of Words)</strong>: Simple word count, good for interpretability</li>
              <li><strong>TF-IDF</strong>: Weighted by term importance, better for handling common words</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</div>
</body>
</html>
'''

PREDICT_HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Predict - Select Model</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    .prediction-card { transition: transform 0.2s; }
    .prediction-card:hover { transform: scale(1.02); }
    .badge-positive { background:#198754; font-size: 1.2rem; }
    .badge-negative { background:#dc3545; font-size: 1.2rem; }
    .badge-neutral { background:#6c757d; font-size: 1.2rem; }
    .badge-spam { background:#fd7e14; font-size: 1.2rem; }
    .badge-not_spam { background:#20c997; font-size: 1.2rem; }
    .confidence-bar { height: 25px; border-radius: 5px; }
    .model-info { background: #f8f9fa; border-radius: 8px; padding: 15px; }
    .result-section { animation: fadeIn 0.5s ease-in; }
    .best-option { font-weight: bold; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
  </style>
</head>
<body class="bg-light">
<div class="container py-4">

<h2>🔮 Predict with Selected Model</h2>
<p class="text-muted">Choose your preferred model or use the best performing model (pre-selected)</p>

<a href="/" class="btn btn-secondary mb-3">&larr; Back</a>
<a href="/models/evaluation" class="btn btn-outline-info mb-3 ms-2">View All Model Metrics</a>

<!-- Input Form -->
<div class="card mb-4">
  <div class="card-header bg-primary text-white"><strong>Enter Text & Select Models</strong></div>
  <div class="card-body">
    <form method="post" action="/predict">
      <div class="mb-3">
        <label class="form-label"><strong>Text to Classify</strong></label>
        <textarea name="text" class="form-control" rows="4" placeholder="Masukkan teks dalam Bahasa Indonesia untuk dianalisis...&#10;&#10;Contoh: Videonya bagus banget, sangat menginspirasi!">{{ input_text or '' }}</textarea>
      </div>
      
      <div class="row mb-3">
        <div class="col-md-6">
          <label class="form-label"><strong>😊 Sentiment Model</strong></label>
          <select name="sentiment_model" class="form-select">
            {% for model in sentiment_models %}
            <option value="{{ model.key }}" {% if model.key == selected_sentiment_model %}selected{% endif %}>
              {{ model.algorithm }} + {{ model.feature_extraction }} (F1: {{ model.f1_score }}%){% if model.is_best %} ⭐ BEST{% endif %}
            </option>
            {% endfor %}
          </select>
          <div class="form-text">⭐ indicates best performing model</div>
        </div>
        <div class="col-md-6">
          <label class="form-label"><strong>📧 Spam Model</strong></label>
          <select name="spam_model" class="form-select">
            {% for model in spam_models %}
            <option value="{{ model.key }}" {% if model.key == selected_spam_model %}selected{% endif %}>
              {{ model.algorithm }} + {{ model.feature_extraction }} (F1: {{ model.f1_score }}%){% if model.is_best %} ⭐ BEST{% endif %}
            </option>
            {% endfor %}
          </select>
          <div class="form-text">⭐ indicates best performing model</div>
        </div>
      </div>
      
      <button class="btn btn-primary btn-lg">🔍 Predict</button>
      <button type="button" class="btn btn-outline-secondary ms-2" onclick="document.querySelector('textarea').value = ''">Clear Text</button>
    </form>
  </div>
</div>

{% if result %}
<!-- Prediction Results -->
<div class="result-section">
  <h4 class="mb-3">Prediction Results</h4>
  
  <div class="row">
    <!-- Sentiment Result -->
    <div class="col-md-6 mb-3">
      <div class="card prediction-card h-100 {% if result.sentiment.label == 'positive' %}border-success{% elif result.sentiment.label == 'negative' %}border-danger{% else %}border-secondary{% endif %}">
        <div class="card-header">
          <strong>😊 Sentiment Analysis</strong>
          <span class="badge bg-secondary float-end">{{ result.sentiment.model_name }}</span>
        </div>
        <div class="card-body text-center">
          <div class="mb-3">
            <span class="badge badge-{{ result.sentiment.label }} p-3">
              {% if result.sentiment.label == 'positive' %}👍 POSITIVE
              {% elif result.sentiment.label == 'negative' %}👎 NEGATIVE
              {% else %}😐 NEUTRAL{% endif %}
            </span>
          </div>
          <div class="mb-2">
            <strong>Confidence: {{ (result.sentiment.confidence * 100)|round(1) }}%</strong>
          </div>
          <div class="progress confidence-bar">
            <div class="progress-bar {% if result.sentiment.label == 'positive' %}bg-success{% elif result.sentiment.label == 'negative' %}bg-danger{% else %}bg-secondary{% endif %}" 
                 role="progressbar" 
                 style="width: {{ (result.sentiment.confidence * 100)|round(1) }}%">
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Spam Result -->
    <div class="col-md-6 mb-3">
      <div class="card prediction-card h-100 {% if result.spam.label == 'spam' %}border-warning{% else %}border-info{% endif %}">
        <div class="card-header">
          <strong>📧 Spam Detection</strong>
          <span class="badge bg-secondary float-end">{{ result.spam.model_name }}</span>
        </div>
        <div class="card-body text-center">
          <div class="mb-3">
            <span class="badge badge-{{ result.spam.label }} p-3">
              {% if result.spam.label == 'spam' %}🚫 SPAM
              {% else %}✅ NOT SPAM{% endif %}
            </span>
          </div>
          <div class="mb-2">
            <strong>Confidence: {{ (result.spam.confidence * 100)|round(1) }}%</strong>
          </div>
          <div class="progress confidence-bar">
            <div class="progress-bar {% if result.spam.label == 'spam' %}bg-warning{% else %}bg-info{% endif %}" 
                 role="progressbar" 
                 style="width: {{ (result.spam.confidence * 100)|round(1) }}%">
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
  
  <!-- Text Analysis -->
  <div class="card mt-3">
    <div class="card-header"><strong>Text Analysis</strong></div>
    <div class="card-body">
      <div class="row">
        <div class="col-md-6">
          <h6>Original Text:</h6>
          <div class="p-2 bg-light rounded" style="word-wrap: break-word;">{{ input_text }}</div>
        </div>
        <div class="col-md-6">
          <h6>Cleaned Text (Preprocessed):</h6>
          <div class="p-2 bg-light rounded" style="word-wrap: break-word;"><code>{{ result.cleaned_text }}</code></div>
        </div>
      </div>
    </div>
  </div>
</div>
{% endif %}

<!-- Example Texts -->
<div class="card mt-4">
  <div class="card-header"><strong>Try These Examples</strong></div>
  <div class="card-body">
    <div class="row">
      <div class="col-md-4">
        <h6 class="text-success">Positive Examples:</h6>
        <ul class="small">
          <li><a href="#" onclick="setExample('Videonya bagus banget, sangat menginspirasi!')">Videonya bagus banget, sangat menginspirasi!</a></li>
          <li><a href="#" onclick="setExample('Keren abis kontennya, terus berkarya ya!')">Keren abis kontennya, terus berkarya ya!</a></li>
          <li><a href="#" onclick="setExample('Mantap sekali, recommended banget!')">Mantap sekali, recommended banget!</a></li>
        </ul>
      </div>
      <div class="col-md-4">
        <h6 class="text-danger">Negative Examples:</h6>
        <ul class="small">
          <li><a href="#" onclick="setExample('Jelek banget videonya, buang waktu')">Jelek banget videonya, buang waktu</a></li>
          <li><a href="#" onclick="setExample('Kecewa berat sama hasilnya, mengecewakan')">Kecewa berat sama hasilnya, mengecewakan</a></li>
          <li><a href="#" onclick="setExample('Tidak recommended, buruk sekali')">Tidak recommended, buruk sekali</a></li>
        </ul>
      </div>
      <div class="col-md-4">
        <h6 class="text-warning">Spam Examples:</h6>
        <ul class="small">
          <li><a href="#" onclick="setExample('Klik link di bio untuk dapat hadiah gratis!')">Klik link di bio untuk dapat hadiah gratis!</a></li>
          <li><a href="#" onclick="setExample('Slot gacor hari ini, maxwin jackpot!')">Slot gacor hari ini, maxwin jackpot!</a></li>
          <li><a href="#" onclick="setExample('DM untuk order, promo diskon 50%!')">DM untuk order, promo diskon 50%!</a></li>
        </ul>
      </div>
    </div>
  </div>
</div>

<script>
function setExample(text) {
  document.querySelector('textarea[name="text"]').value = text;
  event.preventDefault();
}
</script>

</div>
</body>
</html>
'''


@app.route('/', methods=['GET'])
def index():
	csv_path = os.path.join(os.getcwd(), 'dataNew.csv')
	table_html = None
	error = None
	if os.path.exists(csv_path):
		try:
			df = pd.read_csv(csv_path, nrows=20)
			table_html = df_to_html_with_links(df, max_rows=20)
		except Exception as e:
			error = f"Error reading CSV: {e}"
	else:
		error = f"File not found: {csv_path}"

	return render_template_string(INDEX_HTML, table_html=table_html, error=error)



@app.route('/show_all', methods=['GET'])
def show_all():
	csv_path = os.path.join(os.getcwd(), 'dataNew.csv')
	# ensure file exists
	if not os.path.exists(csv_path):
		return render_template_string(INDEX_HTML, table_html=None, error=f"File not found: {csv_path}")
	try:
		df = pd.read_csv(csv_path)
		table_html = df_to_html_with_links(df)
		# when showing all data, update the form's rows_to_show to the number of rows
		rows = int(df.shape[0]) if hasattr(df, 'shape') else None
		return render_template_string(INDEX_HTML, table_html=table_html, error=None, rows_to_show=rows)
	except Exception as e:
		return render_template_string(INDEX_HTML, table_html=None, error=f"Error reading CSV: {e}")


@app.route('/features', methods=['GET'])
def features_view():
    """Show raw feature extraction output (BoW or TF-IDF) without training."""
    csv_path = request.args.get('csv_path', 'dataNew.csv')
    text_col = request.args.get('text_col', 'comments')
    method = request.args.get('method', 'bow')  # 'bow' or 'tfidf'
    
    try:
        max_features = int(request.args.get('max_features', 50))
    except:
        max_features = 50
    
    try:
        ngram = int(request.args.get('ngram', 1))
    except:
        ngram = 1
    
    ngram_range = (1, ngram)
    
    # Default response if no data yet
    context = {
        'csv_path': csv_path,
        'text_col': text_col,
        'method': method,
        'max_features': max_features,
        'ngram': ngram,
        'method_name': 'Bag of Words (Count)' if method == 'bow' else 'TF-IDF',
        'vocab': None,
        'matrix_html': None,
        'matrix_shape': (0, 0),
        'sample_docs': [],
        'error': None
    }
    
    if not os.path.exists(csv_path):
        context['error'] = f"File not found: {csv_path}"
        return render_template_string(FEATURES_HTML, **context)
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        context['error'] = f"Error reading CSV: {e}"
        return render_template_string(FEATURES_HTML, **context)
    
    if text_col not in df.columns:
        context['error'] = f"Column '{text_col}' not found. Available: {list(df.columns)}"
        return render_template_string(FEATURES_HTML, **context)
    
    # Get texts and clean them
    texts = df[text_col].astype(str).fillna('').tolist()
    cleaned_texts = [clean_text(t) for t in texts]
    
    # Filter out empty texts
    valid_data = [(i, orig, clean) for i, (orig, clean) in enumerate(zip(texts, cleaned_texts)) if clean.strip()]
    
    if not valid_data:
        context['error'] = "No valid text data found after cleaning."
        return render_template_string(FEATURES_HTML, **context)
    
    indices, originals, cleaned = zip(*valid_data)
    cleaned = list(cleaned)
    
    # Apply feature extraction (NOT TRAINED - just transform)
    if method == 'bow':
        vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features
        )
    else:  # tfidf
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features
        )
    
    # Fit and transform the data
    feature_matrix = vectorizer.fit_transform(cleaned)
    
    # Get vocabulary (sorted by index)
    vocab = sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])
    vocab_list = [(idx, term) for term, idx in vocab]
    
    # Convert sparse matrix to dense for display (limit rows for performance)
    max_display_rows = min(20, feature_matrix.shape[0])
    dense_matrix = feature_matrix[:max_display_rows].toarray()
    
    # Round TF-IDF values for readability
    if method == 'tfidf':
        dense_matrix = np.round(dense_matrix, 3)
    
    # Create DataFrame for display
    feature_names = vectorizer.get_feature_names_out()
    matrix_df = pd.DataFrame(
        dense_matrix,
        columns=feature_names,
        index=[f"Doc {indices[i]}" for i in range(max_display_rows)]
    )
    
    # Highlight non-zero values
    def highlight_nonzero(val):
        if val > 0:
            return 'background-color: #d4edda'
        return ''
    
    matrix_html = matrix_df.style.applymap(highlight_nonzero).to_html()
    
    # Sample documents
    sample_docs = [
        {'idx': indices[i], 'original': originals[i], 'cleaned': cleaned[i]}
        for i in range(max_display_rows)
    ]
    
    context.update({
        'vocab': vocab_list,
        'matrix_html': matrix_html,
        'matrix_shape': feature_matrix.shape,
        'sample_docs': sample_docs
    })
    
    return render_template_string(FEATURES_HTML, **context)


@app.route('/models', methods=['GET'])
def models_view():
    """Show comparison of all trained models."""
    comparison = get_model_comparison()
    
    # Separate by task
    sentiment_models = [m for m in comparison if m['task'] == 'Sentiment']
    spam_models = [m for m in comparison if m['task'] == 'Spam']
    
    # Find best models
    best_sentiment = max(sentiment_models, key=lambda x: x['cv_accuracy'])
    best_spam = max(spam_models, key=lambda x: x['cv_accuracy'])
    
    # Mark best models
    for m in sentiment_models:
        m['is_best'] = (m['cv_accuracy'] == best_sentiment['cv_accuracy'])
    for m in spam_models:
        m['is_best'] = (m['cv_accuracy'] == best_spam['cv_accuracy'])
    
    return render_template_string(MODELS_HTML,
        sentiment_models=sentiment_models,
        spam_models=spam_models,
        best_sentiment=best_sentiment,
        best_spam=best_spam
    )


@app.route('/models/test', methods=['GET', 'POST'])
def models_test():
    """Test all models with custom text."""
    test_text = ''
    results = None
    cleaned_text = ''
    
    if request.method == 'POST':
        test_text = request.form.get('text', '')
        
        if test_text.strip():
            cleaned_text = clean_text(test_text)
            
            results = {'sentiment': [], 'spam': []}
            
            algorithms = ['Naive Bayes', 'Decision Tree', 'SVM']
            features = ['BoW', 'TF-IDF']
            
            for algo in algorithms:
                for feat in features:
                    # Sentiment
                    sent_label, sent_conf = classify_with_model(test_text, 'sentiment', algo, feat)
                    results['sentiment'].append({
                        'algorithm': algo,
                        'feature_extraction': feat,
                        'label': sent_label,
                        'confidence': sent_conf
                    })
                    
                    # Spam
                    spam_label, spam_conf = classify_with_model(test_text, 'spam', algo, feat)
                    results['spam'].append({
                        'algorithm': algo,
                        'feature_extraction': feat,
                        'label': spam_label,
                        'confidence': spam_conf
                    })
    
    return render_template_string(MODELS_TEST_HTML,
        test_text=test_text,
        results=results,
        cleaned_text=cleaned_text
    )


@app.route('/models/evaluation', methods=['GET'])
def models_evaluation():
    """Show detailed model evaluation with metrics and visualizations."""
    
    # Prepare data for sentiment models
    sentiment_models = []
    for key, info in ALL_MODELS.items():
        if info['task'] == 'sentiment':
            cm_image = generate_confusion_matrix_heatmap(
                np.array(info['confusion_matrix']),
                info['labels'],
                f"{info['algorithm']} ({info['feature_extraction']})"
            )
            sentiment_models.append({
                'algorithm': info['algorithm'],
                'feature_extraction': info['feature_extraction'],
                'accuracy': info['accuracy'],
                'precision': info['precision'],
                'recall': info['recall'],
                'f1_score': info['f1_score'],
                'confusion_matrix': info['confusion_matrix'],
                'labels': info['labels'],
                'cm_image': cm_image,
                'is_best': False
            })
    
    # Prepare data for spam models
    spam_models = []
    for key, info in ALL_MODELS.items():
        if info['task'] == 'spam':
            cm_image = generate_confusion_matrix_heatmap(
                np.array(info['confusion_matrix']),
                info['labels'],
                f"{info['algorithm']} ({info['feature_extraction']})"
            )
            spam_models.append({
                'algorithm': info['algorithm'],
                'feature_extraction': info['feature_extraction'],
                'accuracy': info['accuracy'],
                'precision': info['precision'],
                'recall': info['recall'],
                'f1_score': info['f1_score'],
                'confusion_matrix': info['confusion_matrix'],
                'labels': info['labels'],
                'cm_image': cm_image,
                'is_best': False
            })
    
    # Mark best models
    if sentiment_models:
        best_sentiment = max(sentiment_models, key=lambda x: x['f1_score'])
        for m in sentiment_models:
            if m['f1_score'] == best_sentiment['f1_score']:
                m['is_best'] = True
    
    if spam_models:
        best_spam = max(spam_models, key=lambda x: x['f1_score'])
        for m in spam_models:
            if m['f1_score'] == best_spam['f1_score']:
                m['is_best'] = True
    
    # Generate charts
    sentiment_accuracy_chart = generate_accuracy_comparison_chart('sentiment')
    spam_accuracy_chart = generate_accuracy_comparison_chart('spam')
    sentiment_metrics_chart = generate_all_metrics_chart('sentiment')
    spam_metrics_chart = generate_all_metrics_chart('spam')
    bow_vs_tfidf_chart = generate_bow_vs_tfidf_chart()
    
    # Calculate BoW vs TF-IDF stats
    bow_models = [v for v in ALL_MODELS.values() if v['feature_extraction'] == 'BoW']
    tfidf_models = [v for v in ALL_MODELS.values() if v['feature_extraction'] == 'TF-IDF']
    
    bow_stats = {
        'accuracy': round(np.mean([m['accuracy'] for m in bow_models]), 2),
        'precision': round(np.mean([m['precision'] for m in bow_models]), 2),
        'recall': round(np.mean([m['recall'] for m in bow_models]), 2),
        'f1_score': round(np.mean([m['f1_score'] for m in bow_models]), 2)
    }
    
    tfidf_stats = {
        'accuracy': round(np.mean([m['accuracy'] for m in tfidf_models]), 2),
        'precision': round(np.mean([m['precision'] for m in tfidf_models]), 2),
        'recall': round(np.mean([m['recall'] for m in tfidf_models]), 2),
        'f1_score': round(np.mean([m['f1_score'] for m in tfidf_models]), 2)
    }
    
    winner = 'BoW' if bow_stats['f1_score'] >= tfidf_stats['f1_score'] else 'TF-IDF'
    
    return render_template_string(EVALUATION_HTML,
        sentiment_models=sentiment_models,
        spam_models=spam_models,
        sentiment_accuracy_chart=sentiment_accuracy_chart,
        spam_accuracy_chart=spam_accuracy_chart,
        sentiment_metrics_chart=sentiment_metrics_chart,
        spam_metrics_chart=spam_metrics_chart,
        bow_vs_tfidf_chart=bow_vs_tfidf_chart,
        bow_stats=bow_stats,
        tfidf_stats=tfidf_stats,
        winner=winner
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Predict using the best models."""
    input_text = ''
    result = None
    
    # Build model lists for dropdowns
    sentiment_models = []
    spam_models = []
    
    if ALL_MODELS:
        for key, model_info in ALL_MODELS.items():
            parts = key.split('_')
            task = parts[0]
            algorithm = parts[1]
            feature_extraction = parts[2]
            
            model_entry = {
                'key': key,
                'algorithm': algorithm,
                'feature_extraction': feature_extraction,
                'f1_score': round(model_info['f1_score'], 2),
                'is_best': False
            }
            
            if task == 'sentiment':
                if BEST_SENTIMENT_MODEL and key == f"sentiment_{BEST_SENTIMENT_MODEL['algorithm']}_{BEST_SENTIMENT_MODEL['feature_extraction']}":
                    model_entry['is_best'] = True
                sentiment_models.append(model_entry)
            else:
                if BEST_SPAM_MODEL and key == f"spam_{BEST_SPAM_MODEL['algorithm']}_{BEST_SPAM_MODEL['feature_extraction']}":
                    model_entry['is_best'] = True
                spam_models.append(model_entry)
        
        # Sort by F1 score descending
        sentiment_models.sort(key=lambda x: x['f1_score'], reverse=True)
        spam_models.sort(key=lambda x: x['f1_score'], reverse=True)
    
    # Get default selected models (best ones)
    selected_sentiment_model = ''
    selected_spam_model = ''
    
    if BEST_SENTIMENT_MODEL:
        selected_sentiment_model = f"sentiment_{BEST_SENTIMENT_MODEL['algorithm']}_{BEST_SENTIMENT_MODEL['feature_extraction']}"
    if BEST_SPAM_MODEL:
        selected_spam_model = f"spam_{BEST_SPAM_MODEL['algorithm']}_{BEST_SPAM_MODEL['feature_extraction']}"
    
    if request.method == 'POST':
        input_text = request.form.get('text', '')
        selected_sentiment_model = request.form.get('sentiment_model', selected_sentiment_model)
        selected_spam_model = request.form.get('spam_model', selected_spam_model)
        
        if input_text.strip() and ALL_MODELS:
            # Parse selected models
            sent_parts = selected_sentiment_model.split('_')
            spam_parts = selected_spam_model.split('_')
            
            sent_algorithm = sent_parts[1]
            sent_feature = sent_parts[2]
            spam_algorithm = spam_parts[1]
            spam_feature = spam_parts[2]
            
            # Classify with selected models (returns tuple: label, confidence)
            sent_label, sent_conf = classify_with_model(input_text, 'sentiment', sent_algorithm, sent_feature)
            spam_label, spam_conf = classify_with_model(input_text, 'spam', spam_algorithm, spam_feature)
            
            result = {
                'text': input_text,
                'cleaned_text': clean_text(input_text),
                'sentiment': {
                    'label': sent_label,
                    'confidence': sent_conf,
                    'model_name': f"{sent_algorithm} + {sent_feature}"
                },
                'spam': {
                    'label': spam_label,
                    'confidence': spam_conf,
                    'model_name': f"{spam_algorithm} + {spam_feature}"
                }
            }
    
    return render_template_string(PREDICT_HTML,
        input_text=input_text,
        result=result,
        sentiment_models=sentiment_models,
        spam_models=spam_models,
        selected_sentiment_model=selected_sentiment_model,
        selected_spam_model=selected_spam_model
    )


def classify_text_svm(text):
    text = str(text)

    sent = SENTIMENT_SVM.predict([text])[0]
    sent_conf = SENTIMENT_SVM.predict_proba([text]).max()

    spam = SPAM_SVM.predict([text])[0]
    spam_conf = SPAM_SVM.predict_proba([text]).max()

    return (sent, sent_conf, spam, spam_conf)

@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
	# POST: receive form -> redirect to GET with query params for pagination
	if request.method == 'POST':
		csv_path = request.form.get('csv_path', 'dataNew.csv')
		text_col = request.form.get('text_col', 'comments')
		try:
			page_size = int(request.form.get('n_rows', 20))
		except Exception:
			page_size = 20
		# start at page 1
		return redirect(url_for('preprocess', csv_path=csv_path, text_col=text_col, page=1, page_size=page_size))

	# GET: show paginated preprocessing results
	csv_path = request.args.get('csv_path', 'dataNew.csv')
	text_col = request.args.get('text_col', 'comments')
	try:
		page = int(request.args.get('page', 1))
	except Exception:
		page = 1
	try:
		page_size = int(request.args.get('page_size', request.args.get('n_rows', 20)))
	except Exception:
		page_size = 20

	if not os.path.exists(csv_path):
		return render_template_string(INDEX_HTML, table_html=None, error=f"CSV not found: {csv_path}")

	try:
		df = pd.read_csv(csv_path)
	except Exception as e:
		return render_template_string(INDEX_HTML, table_html=None, error=f"Error reading CSV: {e}")

	if text_col not in df.columns:
		return render_template_string(INDEX_HTML, table_html=None, error=f"Column '{text_col}' not found in CSV. Available columns: {', '.join(df.columns)}")

	# Apply preprocessing for entire column, then paginate
	texts = df[text_col].astype(str).fillna('')

	def process_segments(text):
		# split on pipe | to get segments, strip spaces, ignore empty
		segments = [s.strip() for s in str(text).split('|')]
		segments = [s for s in segments if s]
		if not segments:
			return ('', '')
		cleaned_segs = [clean_text(s) for s in segments]
		token_segs = [tokenize_and_filter(s) for s in cleaned_segs]
		# join cleaned segments with ' | ' and tokens per segment joined by spaces
		cleaned_join = ' | '.join(cleaned_segs)
		tokens_join = ' | '.join([' '.join(t) for t in token_segs])
		return (cleaned_join, tokens_join)

	processed = texts.map(process_segments)
	cleaned = processed.map(lambda x: x[0])
	tokens = processed.map(lambda x: x[1])

	# Apply classification
	def classify_text(text):
		sentiment_result = classify_sentiment(text)
		spam_result = classify_spam(text)
		return (sentiment_result, spam_result)
	
	classifications = texts.map(classify_text)
	sentiment_labels = classifications.map(lambda x: x[0][0])
	sentiment_conf = classifications.map(lambda x: x[0][1])
	spam_labels = classifications.map(lambda x: x[1][0])
	spam_conf = classifications.map(lambda x: x[1][1])

	result_df = pd.DataFrame({
		'original': texts,
		'cleaned_segments': cleaned,
		'tokens_per_segment': tokens,
		'sentiment': sentiment_labels,
		'sentiment_conf': sentiment_conf,
		'spam': spam_labels,
		'spam_conf': spam_conf
	})
	
	# Calculate statistics for all data
	stats = {
		'positive': int((sentiment_labels == 'positive').sum()),
		'negative': int((sentiment_labels == 'negative').sum()),
		'neutral': int((sentiment_labels == 'neutral').sum()),
		'spam': int((spam_labels == 'spam').sum()),
		'not_spam': int((spam_labels == 'not_spam').sum())
	}

	# SVM classification
	svm_results = texts.map(classify_text_svm)

	svm_sentiment = svm_results.map(lambda x: x[0])
	svm_sentiment_conf = svm_results.map(lambda x: x[1])
	svm_spam = svm_results.map(lambda x: x[2])
	svm_spam_conf = svm_results.map(lambda x: x[3])

 
	svm_stats = {
            "sentiment": svm_sentiment,
            "sentiment_conf": round(svm_sentiment_conf * 100, 2),
            "spam": svm_spam,
            "spam_conf": round(svm_spam_conf * 100, 2),
    }
 
	total = int(result_df.shape[0])
	total_pages = max(1, (total + page_size - 1) // page_size)
	if page < 1:
		page = 1
	if page > total_pages:
		page = total_pages
	start = (page - 1) * page_size
	end = start + page_size
	slice_df = result_df.iloc[start:end].copy()

	# Format sentiment and spam columns with badges
	def format_sentiment(row):
		label = row['sentiment']
		conf = row['sentiment_conf']
		return f'<span class="badge badge-{label}">{label}</span> <small>({conf:.0%})</small>'
	
	def format_spam(row):
		label = row['spam']
		conf = row['spam_conf']
		return f'<span class="badge badge-{label}">{label.replace("_", " ")}</span> <small>({conf:.0%})</small>'
	
	slice_df['sentiment'] = slice_df.apply(format_sentiment, axis=1)
	slice_df['spam'] = slice_df.apply(format_spam, axis=1)
	slice_df = slice_df.drop(columns=['sentiment_conf', 'spam_conf'])

	# If text_col looks like URL, convert original column to links
	if 'url' in text_col.lower():
		slice_df['original'] = slice_df['original'].fillna('').apply(lambda u: f'<a href="{u}" target="_blank" rel="noopener noreferrer">{u}</a>' if str(u).strip() else '')
	
	result_table = slice_df.to_html(classes='table table-sm table-striped', index=False, escape=False)

	# render result with pagination context
	return render_template_string(RESULT_HTML,
		result_table=result_table,
		n_rows=page_size,
		page=page,
		total_pages=total_pages,
		csv_path=csv_path,
		text_col=text_col,
		page_size=page_size,
		stats=stats,
		svm_stats=svm_stats
  	)
 
#  -------------------- PER COMMENT VIEW =====================
def explode_comments(df: pd.DataFrame, text_col: str):
    """
    Turns:
      post_row -> "comment1 | comment2 | comment3"
    Into:
      multiple rows, one per comment
    """
    rows = []

    for idx, row in df.iterrows():
        raw_text = str(row.get(text_col, ""))
        segments = [s.strip() for s in raw_text.split('|') if s.strip()]

        for c in segments:
            rows.append({
                "post_index": idx,
                "comment": c
            })

    return pd.DataFrame(rows)

@app.route('/comments', methods=['GET'])
def comments_view():
    csv_path = request.args.get('csv_path', 'dataNew.csv')
    text_col = request.args.get('text_col', 'comments')

    try:
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 25))
    except Exception:
        page, page_size = 1, 25

    if not os.path.exists(csv_path):
        return "CSV not found", 404

    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        return f"Column {text_col} not found", 400

    # 🔥 explode comments
    exploded = explode_comments(df, text_col)

    if exploded.empty:
        return "No comments found", 200

    # Clean + classify PER COMMENT
    exploded['cleaned'] = exploded['comment'].apply(clean_text)

    exploded[['sentiment', 'sentiment_conf']] = exploded['cleaned'].apply(
        lambda x: pd.Series(classify_sentiment(x))
    )

    exploded[['spam', 'spam_conf']] = exploded['cleaned'].apply(
        lambda x: pd.Series(classify_spam(x))
    )

    # Pagination
    total = len(exploded)
    total_pages = max(1, (total + page_size - 1) // page_size)

    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size

    view_df = exploded.iloc[start:end].copy()

    # Badges
    def sentiment_badge(row):
        return f'<span class="badge badge-{row.sentiment}">{row.sentiment}</span> ({row.sentiment_conf:.0%})'

    def spam_badge(row):
        return f'<span class="badge badge-{row.spam}">{row.spam.replace("_"," ")}</span> ({row.spam_conf:.0%})'

    view_df['sentiment'] = view_df.apply(sentiment_badge, axis=1)
    view_df['spam'] = view_df.apply(spam_badge, axis=1)

    view_df = view_df[['post_index', 'comment', 'sentiment', 'spam']]

    table_html = view_df.to_html(
        classes="table table-sm table-striped",
        index=False,
        escape=False
    )

    return render_template_string(COMMENT_HTML,
        table_html=table_html,
        page=page,
        total_pages=total_pages,
        csv_path=csv_path,
        text_col=text_col,
        page_size=page_size
    )

@app.route('/svm', methods=['GET'])
def svm_view():
    csv_path = request.args.get('csv_path', 'dataNew.csv')
    text_col = request.args.get('text_col', 'comments')

    try:
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 25))
    except Exception:
        page, page_size = 1, 25

    if not os.path.exists(csv_path):
        return "CSV not found", 404

    df = pd.read_csv(csv_path)

    if text_col not in df.columns:
        return "Column not found", 400

    exploded = explode_comments(df, text_col)

    exploded['cleaned'] = exploded['comment'].apply(clean_text)

    exploded[['sentiment', 'sentiment_conf']] = exploded['cleaned'].apply(
        lambda x: pd.Series(classify_sentiment_svm(x))
    )

    exploded[['spam', 'spam_conf']] = exploded['cleaned'].apply(
        lambda x: pd.Series(classify_spam_svm(x))
    )

    total = len(exploded)
    total_pages = max(1, (total + page_size - 1) // page_size)

    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size

    view_df = exploded.iloc[start:end].copy()

    def sentiment_badge(row):
        return f'<span class="badge badge-{row.sentiment}">{row.sentiment}</span> ({row.sentiment_conf:.0%})'

    def spam_badge(row):
        return f'<span class="badge badge-{row.spam}">{row.spam.replace("_"," ")}</span> ({row.spam_conf:.0%})'

    view_df['sentiment'] = view_df.apply(sentiment_badge, axis=1)
    view_df['spam'] = view_df.apply(spam_badge, axis=1)

    view_df = view_df[['post_index', 'comment', 'sentiment', 'spam']]

    table_html = view_df.to_html(
        classes='table table-sm table-striped',
        index=False,
        escape=False
    )

    return render_template_string(SVM_HTML,
        table_html=table_html,
        page=page,
        total_pages=total_pages,
        csv_path=csv_path,
        text_col=text_col,
        page_size=page_size
    )


if __name__ == '__main__':
		# Use port 5000 by default. Run with `python main.py`.
		app.run(host='0.0.0.0', port=5500, debug=True)

