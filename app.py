# app.py

import streamlit as st
import re
import math
import numpy as np

# Import advanced libraries
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances_argmin_min
from keybert import KeyBERT
from transformers import pipeline, AutoTokenizer
from newspaper import Article
import spacy
from spacy import displacy
import textstat
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px

# --- LAZY MODEL LOADING ---
# Instead of loading all models at once, we load them on demand and cache them.
# This makes the app startup much faster.

def get_model(model_name):
    if model_name not in st.session_state:
        st.session_state[model_name] = pipeline(model_name.split(':')[0], model=model_name.split(':')[1]) if ':' in model_name else pipeline(model_name)
    return st.session_state[model_name]

def get_sentence_transformer(model_name):
    if model_name not in st.session_state:
        st.session_state[model_name] = SentenceTransformer(model_name)
    return st.session_state[model_name]

def get_tokenizer(model_name):
    if f"tokenizer_{model_name}" not in st.session_state:
        st.session_state[f"tokenizer_{model_name}"] = AutoTokenizer.from_pretrained(model_name)
    return st.session_state[f"tokenizer_{model_name}"]
    
def get_spacy_model(model_name):
    if model_name not in st.session_state:
        st.session_state[model_name] = spacy.load(model_name)
    return st.session_state[model_name]

# Initialize session state for caching results
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'last_text' not in st.session_state:
    st.session_state.last_text = ""

# --- PROCESSING FUNCTIONS ---

def get_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        st.error(f"Error: Could not fetch article. Details: {e}")
        return None

def preprocess_text(text):
    text = text.replace('\n', ' ').replace('\r', '')
    text = re.sub(r'\s+', ' ', text)
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def generate_extractive_summary(text, ratio, model):
    sentences = preprocess_text(text)
    if len(sentences) < 4: return "Text too short for extractive summary."
    embeddings = model.encode(sentences)
    num_clusters = math.ceil(len(sentences) * ratio)
    if num_clusters < 2 and len(sentences) > 1: num_clusters = 2
    if num_clusters >= len(sentences): num_clusters = len(sentences) - 1
    clustering_model = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
    clustering_model.fit(embeddings)
    closest, _ = pairwise_distances_argmin_min(clustering_model.cluster_centers_, embeddings)
    summary_sentences = [sentences[i] for i in sorted(closest)]
    return " ".join(summary_sentences)

def generate_abstractive_summary(text, summarizer, tokenizer):
    max_chunk_length = 1024
    sentences = preprocess_text(text)
    if len(sentences) < 5: return "Text too short to summarize."
    sentence_tokens = [tokenizer.tokenize(s) for s in sentences]
    chunks, current_chunk_tokens = [], []
    for tokens in sentence_tokens:
        if len(current_chunk_tokens) + len(tokens) <= max_chunk_length:
            current_chunk_tokens.extend(tokens)
        else:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk_tokens))
            current_chunk_tokens = tokens
    if current_chunk_tokens: chunks.append(tokenizer.convert_tokens_to_string(current_chunk_tokens))
    try:
        with st.spinner(f"Summarizing {len(chunks)} chunk(s)..."):
            chunk_summaries = summarizer(chunks, max_length=150, min_length=30, do_sample=False)
        return " ".join([summary['summary_text'] for summary in chunk_summaries])
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return "Failed to generate abstractive summary."

def extract_keywords(text, model):
    kw_model = KeyBERT(model=model)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    return [kw[0] for kw in keywords]

def analyze_entities(text, model):
    doc = model(text)
    return displacy.render(doc, style="ent", jupyter=False)

def analyze_sentiment(text, model):
    result = model(text[:512])
    return result[0]['label'], result[0]['score']

def classify_topic(text, model):
    candidate_labels = ["Technology", "Politics", "Sports", "Business", "Entertainment", "Science", "Health", "World News"]
    result = model(text[:512], candidate_labels)
    return result['labels'][0], result['scores'][0]

def analyze_emotion(text, model):
    result = model(text[:512])
    return result[0][0]['label'].capitalize(), result[0][0]['score']

def calculate_readability(text):
    score = textstat.flesch_reading_ease(text)
    if score >= 90: grade = "Very Easy (5th Grade)"
    elif score >= 80: grade = "Easy (6th Grade)"
    elif score >= 70: grade = "Fairly Easy (7th Grade)"
    elif score >= 60: grade = "Standard (8th-9th Grade)"
    elif score >= 50: grade = "Fairly Difficult (10th-12th Grade)"
    elif score >= 30: grade = "Difficult (College)"
    else: grade = "Very Confusing (Graduate)"
    return f"{score:.1f} ({grade})"

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def compare_texts(text_a, text_b, model):
    embeddings = model.encode([text_a, text_b])
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    sentences_a, sentences_b = preprocess_text(text_a), preprocess_text(text_b)
    embeddings_a, embeddings_b = model.encode(sentences_a), model.encode(sentences_b)
    similarities = cosine_similarity(embeddings_a, embeddings_b)
    unique_to_a = [sentences_a[i] for i, row in enumerate(similarities) if np.max(row) < 0.6]
    unique_to_b = [sentences_b[i] for i, col in enumerate(similarities.T) if np.max(col) < 0.6]
    return similarity_score, unique_to_a, unique_to_b

def simplify_text(text, model):
    prompt = "simplify: " + text
    result = model(prompt, max_length=len(text.split())*2, min_length=len(text.split())//2)
    return result[0]['generated_text']

def analyze_arguments(text, model):
    sentences = preprocess_text(text)
    candidate_labels = ["claim", "premise", "evidence", "non-argumentative"]
    results = model(sentences, candidate_labels)
    claims = [res['sequence'] for res in results if res['labels'][0] in ["claim", "premise"]]
    return claims

def analyze_toxicity(text, model):
    results = model(text[:512])
    # Find the 'toxic' score, or the highest score if 'toxic' isn't a label
    toxic_score = 0
    for res in results:
        if res['label'] == 'toxic':
            toxic_score = res['score']
            break
    if toxic_score > 0.5:
        return "High", toxic_score
    elif toxic_score > 0.2:
        return "Moderate", toxic_score
    else:
        return "Low", toxic_score
        
def visualize_sentiment_timeline(text, model):
    sentences = preprocess_text(text)
    if len(sentences) < 3: return None
    results = model(sentences)
    scores = [res['score'] if res['label'] == 'POSITIVE' else -res['score'] for res in results]
    
    fig = px.line(x=range(len(scores)), y=scores, labels={'x': 'Sentence Number', 'y': 'Sentiment Score'}, title='Sentiment Timeline')
    fig.update_layout(yaxis_range=[-1,1])
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    return fig

# --- STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("🏆 Elite AI Content Analyzer Suite")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")
accuracy_mode = st.sidebar.toggle("Enable High-Accuracy Mode", help="Uses larger, more accurate models. May be slower.")
mode = "accurate" if accuracy_mode else "fast"

# --- Main UI Tabs ---
main_tab, advanced_tools_tab = st.tabs(["Single Text Analysis", "Advanced Tools"])

with main_tab:
    st.header("Analyze a Single Piece of Content")
    input_option = st.radio("Input method:", ["From URL", "Paste Text"], horizontal=True, key="main_input_option")
    
    input_text = ""
    if input_option == "Paste Text":
        input_text = st.text_area("Paste your text here:", height=250, key="main_text_area")
    else:
        url = st.text_input("Enter article URL:", key="url_input")
        if url:
            with st.spinner("Fetching article..."):
                input_text = get_text_from_url(url)
                st.text_area("Extracted Text:", value=input_text, height=250, disabled=True, key="fetched_text")

    if st.button("Analyze Content", key="analyze_button"):
        if not input_text:
            st.warning("Please provide text or a valid URL.")
        else:
            st.session_state.last_text = input_text
            with st.spinner("Performing comprehensive analysis... Please wait."):
                # Dynamically select model names based on mode
                extractive_model_name = 'all-mpnet-base-v2' if mode == 'accurate' else 'all-MiniLM-L6-v2'
                abstractive_model_name = 'facebook/bart-large-cnn' if mode == 'accurate' else 'sshleifer/distilbart-cnn-12-6'

                results = {
                    "sentiment": analyze_sentiment(input_text, get_model('sentiment-analysis:distilbert-base-uncased-finetuned-sst-2-english')),
                    "topic": classify_topic(input_text, get_model('zero-shot-classification:facebook/bart-large-mnli')),
                    "emotion": analyze_emotion(input_text, get_model('text-classification:j-hartmann/emotion-english-distilroberta-base')),
                    "toxicity": analyze_toxicity(input_text, get_model('text-classification:distilbert-base-uncased-finetuned-sst-2-english')),
                    "readability": calculate_readability(input_text),
                    "keywords": extract_keywords(input_text, get_sentence_transformer(extractive_model_name)),
                    "entity_html": analyze_entities(input_text, get_spacy_model('en_core_web_sm')),
                    "wordcloud_fig": generate_wordcloud(input_text),
                    "arguments": analyze_arguments(input_text, get_model('zero-shot-classification:facebook/bart-large-mnli')),
                    "sentiment_timeline_fig": visualize_sentiment_timeline(input_text, get_model('sentiment-analysis:distilbert-base-uncased-finetuned-sst-2-english'))
                }
                st.session_state.analysis_results = results

    if st.session_state.analysis_results and st.session_state.last_text == input_text:
        results = st.session_state.analysis_results
        
        summary_type = st.selectbox("Summarization Type:", ["Abstractive", "Extractive"], key="summary_type_select")
        
        extractive_model_name = 'all-mpnet-base-v2' if mode == 'accurate' else 'all-MiniLM-L6-v2'
        abstractive_model_name = 'facebook/bart-large-cnn' if mode == 'accurate' else 'sshleifer/distilbart-cnn-12-6'

        if summary_type == "Extractive":
            ratio = st.slider("Summary Length (Ratio)", 0.1, 0.5, 0.2, 0.05, key="ratio_slider")
            summary = generate_extractive_summary(st.session_state.last_text, ratio, get_sentence_transformer(extractive_model_name))
        else:
            if 'abstractive_summary' not in results or results.get('summary_mode') != mode:
                 results['abstractive_summary'] = generate_abstractive_summary(st.session_state.last_text, get_model(f'summarization:{abstractive_model_name}'), get_tokenizer(abstractive_model_name))
                 results['summary_mode'] = mode
            summary = results['abstractive_summary']

        st.subheader("📝 Generated Summary")
        st.success(summary)
        
        st.subheader("📊 Comprehensive Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Overall Sentiment", value=results['sentiment'][0], delta=f"{results['sentiment'][1]:.2%}")
            st.metric(label="Predicted Topic", value=results['topic'][0], delta=f"{results['topic'][1]:.2%}")
        with col2:
            st.metric(label="Dominant Emotion", value=results['emotion'][0], delta=f"{results['emotion'][1]:.2%}")
            st.metric(label="Toxicity Level", value=results['toxicity'][0], delta=f"{results['toxicity'][1]:.2%}")
        with col3:
             st.metric(label="Readability Score", value=results['readability'])

        st.subheader("🔑 Keywords")
        st.info(", ".join(results['keywords']))
        
        tab_viz, tab_args, tab_ents = st.tabs(["Visualizations", "Argument Analysis", "Named Entities"])
        with tab_viz:
            st.write("#### ☁️ Word Cloud")
            st.pyplot(results['wordcloud_fig'])
            if results['sentiment_timeline_fig']:
                st.write("####📈 Sentiment Timeline")
                st.plotly_chart(results['sentiment_timeline_fig'], use_container_width=True)
        with tab_args:
             if results['arguments']:
                for arg in results['arguments']: st.write(f"- {arg}")
             else:
                st.write("No strong claims or premises were identified.")
        with tab_ents:
             st.markdown(results['entity_html'], unsafe_allow_html=True)

        report_text = f"SUMMARY: {summary}\n\nKEYWORDS: {', '.join(results['keywords'])}\n\nANALYTICS:\n- Sentiment: {results['sentiment'][0]} ({results['sentiment'][1]:.2%})\n- Topic: {results['topic'][0]} ({results['topic'][1]:.2%})\n- Emotion: {results['emotion'][0]} ({results['emotion'][1]:.2%})\n- Readability: {results['readability']}"
        st.download_button("Download Full Report", report_text, "analysis_report.txt")

with advanced_tools_tab:
    st.header("Advanced Text Tools")
    tool_choice = st.selectbox("Choose a tool:", ["Comparative Analysis", "Text Simplifier"])
    extractive_model_name = 'all-mpnet-base-v2' if mode == 'accurate' else 'all-MiniLM-L6-v2'

    if tool_choice == "Comparative Analysis":
        colA, colB = st.columns(2)
        with colA: text_a = st.text_area("Paste Text A here:", height=300, key="text_a")
        with colB: text_b = st.text_area("Paste Text B here:", height=300, key="text_b")
        if st.button("Compare Texts", key="compare_button"):
            if not text_a or not text_b: st.warning("Please paste text into both boxes.")
            else:
                with st.spinner("Performing comparative analysis..."):
                    score, unique_a, unique_b = compare_texts(text_a, text_b, get_sentence_transformer(extractive_model_name))
                st.metric(label="Semantic Similarity Score", value=f"{score:.2%}")
                exp_a = st.expander("Unique Points in Text A")
                with exp_a:
                    if unique_a: [st.write(f"- {s}") for s in unique_a]
                    else: st.write("No distinct points found.")
                exp_b = st.expander("Unique Points in Text B")
                with exp_b:
                    if unique_b: [st.write(f"- {s}") for s in unique_b]
                    else: st.write("No distinct points found.")
    
    elif tool_choice == "Text Simplifier":
        st.subheader("Simplify Complex Text")
        simplify_text_input = st.text_area("Paste the complex text you want to simplify:", height=200)
        if st.button("Simplify"):
            if not simplify_text_input: st.warning("Please enter text to simplify.")
            else:
                with st.spinner("Rewriting text..."):
                    simplified_text = simplify_text(simplify_text_input, get_model('text2text-generation:t5-base'))
                st.success("Simplified Version:")
                st.markdown(f"> {simplified_text}")

