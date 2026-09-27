import os
import sys
import json
import streamlit as st
import pandas as pd
import altair as alt
import speech_recognition as sr

# Ensure root directory is accessible
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from engines.spell_checker import SpellCheckerEngine
from engines.grammar_checker import GrammarCheckerEngine
from engines.next_word_predictor import NextWordPredictor
from engines.text_enhancer import TextEnhancerEngine

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Mini Grammarly | AI Writing Assistant",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Grammarly-inspired CSS styling
st.markdown("""
<style>
    /* Global Styles */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f5132;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 2px;
    }
    .sub-header {
        color: #4b5563;
        font-size: 1.05rem;
        margin-bottom: 20px;
    }
    .badge-pill {
        background-color: #10b981;
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .suggestion-card {
        background-color: #ffffff;
        border-left: 4px solid #ef4444;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-top: 1px solid #f3f4f6;
        border-right: 1px solid #f3f4f6;
        border-bottom: 1px solid #f3f4f6;
    }
    .suggestion-card.grammar {
        border-left-color: #3b82f6;
    }
    .suggestion-card.style {
        border-left-color: #8b5cf6;
    }
    .next-word-chip {
        background: #f0fdf4;
        border: 1px solid #86efac;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.95rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached Engine Initializers
# ---------------------------------------------------------
@st.cache_resource
def load_engines():
    spell_eng = SpellCheckerEngine()
    grammar_eng = GrammarCheckerEngine()
    next_word_eng = NextWordPredictor()
    enhancer_eng = TextEnhancerEngine()
    return spell_eng, grammar_eng, next_word_eng, enhancer_eng

spell_engine, grammar_engine, next_word_predictor, enhancer_engine = load_engines()

# Load benchmark results
benchmark_file = os.path.join(os.path.dirname(__file__), "data", "benchmark_results.json")
benchmark_data = {}
if os.path.exists(benchmark_file):
    try:
        with open(benchmark_file, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)
    except Exception:
        pass

# ---------------------------------------------------------
# Key-Versioned State Management
# (Guarantees 100% reliable instant updates for all buttons)
# ---------------------------------------------------------
if "editor_version" not in st.session_state:
    st.session_state["editor_version"] = 0

if "editor_content" not in st.session_state:
    st.session_state["editor_content"] = "I has a car and I am goin to school."

def set_text(new_text):
    """Sets editor text and increments version key to force immediate UI update."""
    st.session_state["editor_content"] = new_text
    st.session_state["editor_version"] += 1
    st.rerun()

# ---------------------------------------------------------
# Sidebar: Model Config & 10 Sample Sentences
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/grammarly.png", width=64)
    st.title("Mini Grammarly")
    st.caption("AI Writing Assistant & Deep Learning Suite")
    st.markdown("---")

    st.subheader("🤖 Sequence Model Selection")
    available_models = next_word_predictor.get_available_models()
    if not available_models:
        available_models = ["Transformer", "GRU", "LSTM", "RNN"]
    
    selected_model = st.selectbox(
        "Active Next-Word Architecture:",
        options=available_models,
        index=0 if "Transformer" in available_models else 0,
        help="Select which deep sequence model powers the next-word prediction."
    )
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        top_k = st.slider("Top-K Words:", min_value=1, max_value=8, value=4)
    with col_sb2:
        temperature = st.slider("Temperature:", min_value=0.2, max_value=2.0, value=1.0, step=0.1)

    st.markdown("---")
    st.subheader("📌 10 Demonstrative Test Sentences")
    st.caption("Click any test case to load it into the editor canvas:")
    
    test_sentences = [
        ("1. PDF Task Specimen", "I has a car and I am goin to school."),
        ("2. Next-Word Prediction Prompt", "I am going to"),
        ("3. Subject-Verb Agreement", "He have finished his homework and they is ready."),
        ("4. Verb Form / Auxiliary", "She is walk to university every day."),
        ("5. Article Misuse (a vs an)", "He ate a apple and bought an car."),
        ("6. Preposition Misuse", "She is very interested on artificial intelligence."),
        ("7. Past Tense with 'Did'", "We did went to the conference yesterday."),
        ("8. Double Negatives", "They don't know nothing about the new project."),
        ("9. Comparative Redundancy", "This model is more better and more faster."),
        ("10. Roman Urdu Test Case", "mein school ja raha hoon")
    ]
    
    for idx, (label, sentence) in enumerate(test_sentences):
        if st.button(label, key=f"sidebar_test_btn_{idx}", use_container_width=True):
            set_text(sentence)

    st.markdown("---")
    st.info("💡 **Academic Prototype**: Built with PyTorch, TensorFlow, NLTK, SymSpell, and TextStat demonstrating RNN → LSTM → GRU → Transformer evolution.")

# ---------------------------------------------------------
# Top Navigation & Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">✍️ Mini Grammarly <span class="badge-pill">Deep Learning Edition</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time intelligent spelling, grammar detection, sentence tone rewriting, and multi-architecture next-word prediction.</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Writing Canvas
# ---------------------------------------------------------
col_main, col_stats = st.columns([7, 3])

with col_main:
    # Text Area with dynamic key versioning to ensure 100% responsiveness
    active_key = f"text_canvas_v{st.session_state['editor_version']}"
    user_input = st.text_area(
        "Enter or paste your English text below:",
        value=st.session_state["editor_content"],
        height=160,
        placeholder="Type your sentences here... (e.g., I has a car and I am goin to school.)",
        key=active_key
    )
    # Sync typed text back to editor_content
    st.session_state["editor_content"] = user_input
    current_text = st.session_state["editor_content"]

    # Editor Action Bar
    btn_col1, btn_col2, btn_col3 = st.columns([4, 3, 3])
    
    with btn_col1:
        if st.button("⚡ One-Click Auto-Fix", use_container_width=True, type="primary"):
            fixed_spell = spell_engine.auto_correct(current_text)
            fixed_all = grammar_engine.auto_correct(fixed_spell)
            set_text(fixed_all)
            
    with btn_col2:
        if st.button("🧹 Clear Canvas", use_container_width=True):
            set_text("")
            
    with btn_col3:
        if st.button("🔄 Reset to PDF Sample", use_container_width=True):
            set_text("I has a car and I am goin to school.")

    # Voice Input Expander
    with st.expander("🎙️ Voice Input (Record Audio with Microphone)", expanded=False):
        st.caption("Click the microphone below to record. Your speech will be transcribed and can be added directly to the editor.")
        audio_record = st.audio_input("Record audio:", key=f"voice_input_{st.session_state['editor_version']}")
        if audio_record is not None:
            try:
                r = sr.Recognizer()
                with sr.AudioFile(audio_record) as source:
                    audio_data = r.record(source)
                    transcribed_text = r.recognize_google(audio_data)
                    st.success(f"**Transcribed Speech**: \"{transcribed_text}\"")
                    
                    v_col1, v_col2 = st.columns(2)
                    with v_col1:
                        if st.button("📥 Append to Canvas", key="btn_append_voice", use_container_width=True):
                            cur = current_text.strip()
                            new_t = f"{cur} {transcribed_text}".strip() if cur else transcribed_text
                            set_text(new_t)
                    with v_col2:
                        if st.button("📝 Replace Entire Canvas", key="btn_replace_voice", use_container_width=True):
                            set_text(transcribed_text)
            except Exception as e:
                st.warning(f"Could not transcribe audio. Error: {e}. Please ensure clear audio.")

    # ---------------------------------------------------------
    # Next-Word Prediction Chips
    # ---------------------------------------------------------
    st.markdown(f"##### 🔮 Next-Word Predictions (`{selected_model}` Architecture)")
    clean_text = current_text.strip()
    
    if clean_text:
        predictions = next_word_predictor.predict(
            clean_text, 
            model_name=selected_model, 
            top_k=top_k, 
            temperature=temperature
        )
        
        if predictions:
            chip_cols = st.columns(len(predictions))
            for i, p in enumerate(predictions):
                w = p["word"]
                pct = p["percentage"]
                with chip_cols[i]:
                    if st.button(f"+ {w} ({pct})", key=f"chip_btn_{i}_{w}_{st.session_state['editor_version']}", use_container_width=True):
                        cur = current_text.rstrip()
                        if cur:
                            new_t = f"{cur} {w} "
                        else:
                            new_t = f"{w} "
                        set_text(new_t)
        else:
            st.caption("Keep typing to see neural next-word completions...")
    else:
        st.caption("Enter a phrase above (e.g. `I am going to`) to generate next-word predictions.")

# Right column: Live Analytics Snapshot
with col_stats:
    st.markdown("##### 📈 Live Writing Metrics")
    readability_stats = enhancer_engine.analyze_readability(current_text)
    
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{readability_stats['word_count']}</div>
            <div class="metric-label">Words</div>
        </div>
        """, unsafe_allow_html=True)
    with c_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{readability_stats['character_count']}</div>
            <div class="metric-label">Characters</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    c_m3, c_m4 = st.columns(2)
    with c_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{readability_stats['reading_ease']}</div>
            <div class="metric-label">Reading Ease</div>
        </div>
        """, unsafe_allow_html=True)
    with c_m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{readability_stats['grade_level']}</div>
            <div class="metric-label">Grade Level</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.caption(f"📖 **Clarity Rating**: {readability_stats['reading_ease_label']}")
    st.caption(f"⏱️ **Est. Reading Time**: {readability_stats['reading_time_sec']}s")

st.markdown("---")

# ---------------------------------------------------------
# Feature Hub Tabs
# ---------------------------------------------------------
tab_inspect, tab_rewrite, tab_models, tab_analytics, tab_vocab, tab_urdu = st.tabs([
    "🔍 Suggestions & Corrections",
    "✍️ Tone Rewriter",
    "🧠 Deep Models Benchmark",
    "📊 In-Depth Analytics",
    "💎 Vocabulary Booster",
    "🌐 Urdu-English Support"
])

# ---------------------------------------------------------
# TAB 1: Suggestions & Error Inspector
# ---------------------------------------------------------
with tab_inspect:
    spell_errors = spell_engine.check(current_text)
    grammar_errors = grammar_engine.check(current_text)
    total_issues = len(spell_errors) + len(grammar_errors)

    col_h1, col_h2 = st.columns([7, 3])
    with col_h1:
        if total_issues == 0 and current_text.strip():
            st.success("🎉 Excellent! No grammar or spelling errors detected.")
        elif total_issues > 0:
            st.warning(f"Found **{total_issues}** writing issues ({len(spell_errors)} spelling, {len(grammar_errors)} grammar).")
        else:
            st.info("Write or paste text to inspect for errors.")

    with col_h2:
        if total_issues > 0:
            if st.button("Accept All Suggestions", type="primary", key=f"btn_accept_all_{st.session_state['editor_version']}", use_container_width=True):
                fixed_s = spell_engine.auto_correct(current_text)
                fixed_g = grammar_engine.auto_correct(fixed_s)
                set_text(fixed_g)

    # Display Grammar Issues
    if grammar_errors:
        st.markdown("#### 📘 Grammar Corrections")
        for i, err in enumerate(grammar_errors):
            card_class = "suggestion-card grammar"
            orig = err["original"]
            rep = err["replacement"]
            cat = err.get("category", "Grammar")
            expl = err["explanation"]
            
            c_card, c_act = st.columns([8, 2])
            with c_card:
                st.markdown(f"""
                <div class="{card_class}">
                    <span style="background:#dbeafe; color:#1e40af; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:4px;">{cat.upper()}</span>
                    <div style="margin-top:6px; font-size:1.05rem;">
                        Change <span style="text-decoration:line-through; color:#ef4444; font-weight:600;">{orig}</span> to 
                        <span style="color:#059669; font-weight:700;">{rep}</span>
                    </div>
                    <div style="color:#6b7280; font-size:0.85rem; margin-top:4px;">{expl}</div>
                </div>
                """, unsafe_allow_html=True)
            with c_act:
                st.write("")
                if st.button(f"Accept '{rep}'", key=f"btn_accept_g_{i}_{st.session_state['editor_version']}", use_container_width=True):
                    cur = current_text
                    if cur[err["start"]:err["end"]] == orig:
                        chars = list(cur)
                        chars[err["start"]:err["end"]] = list(rep)
                        set_text("".join(chars))
                    else:
                        set_text(cur.replace(orig, rep, 1))

    # Display Spelling Issues
    if spell_errors:
        st.markdown("#### 📕 Spelling Corrections")
        for i, err in enumerate(spell_errors):
            orig = err["original"]
            rep = err["replacement"]
            expl = err["explanation"]
            
            c_card, c_act = st.columns([8, 2])
            with c_card:
                st.markdown(f"""
                <div class="suggestion-card">
                    <span style="background:#fee2e2; color:#991b1b; font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:4px;">SPELLING</span>
                    <div style="margin-top:6px; font-size:1.05rem;">
                        Replace <span style="text-decoration:line-through; color:#ef4444; font-weight:600;">{orig}</span> with 
                        <span style="color:#059669; font-weight:700;">{rep}</span>
                    </div>
                    <div style="color:#6b7280; font-size:0.85rem; margin-top:4px;">{expl}</div>
                </div>
                """, unsafe_allow_html=True)
            with c_act:
                st.write("")
                if st.button(f"Accept '{rep}'", key=f"btn_accept_s_{i}_{st.session_state['editor_version']}", use_container_width=True):
                    cur = current_text
                    if cur[err["start"]:err["end"]] == orig:
                        chars = list(cur)
                        chars[err["start"]:err["end"]] = list(rep)
                        set_text("".join(chars))
                    else:
                        set_text(cur.replace(orig, rep, 1))

# ---------------------------------------------------------
# TAB 2: Tone Rewriter
# ---------------------------------------------------------
with tab_rewrite:
    st.markdown("#### ✍️ AI Sentence Rewriter (Tone Adaptation)")
    st.caption("Instantly rewrite your text for professional, casual, concise, or academic contexts.")
    
    if current_text.strip():
        tones = ["Formal / Professional", "Casual / Friendly", "Concise / Direct", "Academic / Elaborate"]
        tone_cols = st.columns(2)
        
        for idx, t in enumerate(tones):
            t_key = t.split()[0].lower()
            rewritten_text = enhancer_engine.rewrite_sentence(current_text, tone=t_key)
            target_col = tone_cols[idx % 2]
            
            with target_col:
                st.markdown(f"**{t}**")
                st.info(rewritten_text if rewritten_text else "No rewrite available.")
                if st.button(f"Use {t.split()[0]} Version", key=f"btn_rewrite_apply_{idx}_{st.session_state['editor_version']}", use_container_width=True):
                    set_text(rewritten_text)
    else:
        st.info("Please enter text above to view stylistic rewrites.")

# ---------------------------------------------------------
# TAB 3: Model Comparison & Benchmarks
# ---------------------------------------------------------
with tab_models:
    st.markdown("#### 🧠 Deep Learning Architecture Comparison (RNN vs LSTM vs GRU vs Transformer)")
    st.caption("Empirical evaluation metrics trained on a common linguistic corpus with identical vocabulary and hyperparameter budgets.")
    
    if benchmark_data:
        # Comparison Table
        table_rows = []
        for model_name, metrics in benchmark_data.items():
            table_rows.append({
                "Architecture": model_name,
                "Parameters": f"{metrics['parameter_count']:,}",
                "Val Loss": metrics['val_loss'],
                "Perplexity (PPL)": metrics['perplexity'],
                "Top-1 Acc (%)": f"{metrics['top1_accuracy']}%",
                "Top-5 Acc (%)": f"{metrics['top5_accuracy']}%",
                "Latency (ms)": f"{metrics['inference_latency_ms']} ms",
                "Train Time (s)": f"{metrics['training_time_sec']}s"
            })
        
        df_bench = pd.DataFrame(table_rows)
        st.dataframe(df_bench, use_container_width=True, hide_index=True)

        st.markdown("---")
        chart_col1, chart_col2 = st.columns(2)
        
        # Loss Chart
        with chart_col1:
            st.markdown("##### 📉 Training & Validation Loss Comparison")
            chart_df = pd.DataFrame([
                {"Model": m, "Val Loss": metrics['val_loss'], "Perplexity": metrics['perplexity']}
                for m, metrics in benchmark_data.items()
            ])
            bar_loss = alt.Chart(chart_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X('Model:N', title='Model Architecture'),
                y=alt.Y('Val Loss:Q', title='Validation Cross-Entropy Loss'),
                color=alt.Color('Model:N', scale=alt.Scale(scheme='set2'))
            ).properties(height=280)
            st.altair_chart(bar_loss, use_container_width=True)

        # Latency vs Accuracy Chart
        with chart_col2:
            st.markdown("##### ⚡ Inference Latency (ms per token)")
            lat_df = pd.DataFrame([
                {"Model": m, "Latency (ms)": metrics['inference_latency_ms']}
                for m, metrics in benchmark_data.items()
            ])
            bar_lat = alt.Chart(lat_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                x=alt.X('Model:N', title='Model Architecture'),
                y=alt.Y('Latency (ms):Q', title='Inference Latency (ms)'),
                color=alt.Color('Model:N', scale=alt.Scale(scheme='tableau10'))
            ).properties(height=280)
            st.altair_chart(bar_lat, use_container_width=True)

        st.markdown("---")
        st.markdown("##### 🧪 Live Side-by-Side Next-Word Prediction Comparison")
        st.write("Compare how all 4 sequence models predict the next words for your current text:")
        
        if current_text.strip():
            all_preds = next_word_predictor.compare_all_models(current_text, top_k=3)
            p_cols = st.columns(len(all_preds))
            for i, (m_name, m_preds) in enumerate(all_preds.items()):
                with p_cols[i]:
                    st.markdown(f"**{m_name}**")
                    if m_preds:
                        for p in m_preds:
                            st.write(f"- `{p['word']}` ({p['percentage']})")
                    else:
                        st.caption("No prediction")
    else:
        st.info("Run `python scripts/train_models.py` to populate live benchmark results.")

# ---------------------------------------------------------
# TAB 4: In-Depth Readability & Analytics
# ---------------------------------------------------------
with tab_analytics:
    st.markdown("#### 📊 Detailed Text Analytics & Linguistic Statistics")
    
    if current_text.strip():
        stats = enhancer_engine.analyze_readability(current_text)
        
        a_col1, a_col2, a_col3 = st.columns(3)
        with a_col1:
            st.metric("Flesch Reading Ease", f"{stats['reading_ease']} / 100", help="Higher means easier to read. 60-70 is standard.")
            st.metric("Flesch-Kincaid Grade", f"Grade {stats['grade_level']}", help="US school grade level required to understand the text.")
        with a_col2:
            st.metric("Sentence Count", stats['sentence_count'])
            st.metric("Avg Sentence Length", f"{stats['avg_sentence_len']} words")
        with a_col3:
            st.metric("Total Word Count", stats['word_count'])
            st.metric("Reading Time", f"{stats['reading_time_sec']} seconds")
            
        st.markdown("---")
        st.markdown("##### 📚 Reading Ease Scale Reference")
        scale_data = pd.DataFrame([
            {"Score Range": "90–100", "Level": "5th Grade", "Description": "Very easy to read, simple conversational English."},
            {"Score Range": "80–89", "Level": "6th Grade", "Description": "Easy to read, plain English."},
            {"Score Range": "70–79", "Level": "7th Grade", "Description": "Fairly easy to read."},
            {"Score Range": "60–69", "Level": "8th–9th Grade", "Description": "Standard English, easily understood by general audience."},
            {"Score Range": "50–59", "Level": "10th–12th Grade", "Description": "Fairly difficult."},
            {"Score Range": "30–49", "Level": "College", "Description": "Difficult, academic or technical articles."},
            {"Score Range": "0–29", "Level": "College Graduate", "Description": "Very difficult, scientific research papers."}
        ])
        st.dataframe(scale_data, use_container_width=True, hide_index=True)
    else:
        st.info("Enter text above to compute readability indices.")

# ---------------------------------------------------------
# TAB 5: Vocabulary Booster
# ---------------------------------------------------------
with tab_vocab:
    st.markdown("#### 💎 Vocabulary Booster & Synonym Suggestions")
    st.caption("Identifies overused, repetitive, or basic vocabulary and recommends sophisticated alternatives.")
    
    if current_text.strip():
        vocab_suggestions = enhancer_engine.get_synonym_suggestions(current_text)
        if vocab_suggestions:
            for s in vocab_suggestions:
                orig_word = s["original"]
                upgrades = s["upgrades"]
                cat = s["category"]
                
                with st.expander(f"Word Upgrade: '{orig_word}' ({cat})", expanded=True):
                    st.write(f"Consider replacing **'{orig_word}'** with one of these enhanced words:")
                    chips_html = " ".join([f"<span class='next-word-chip'>{u}</span>" for u in upgrades])
                    st.markdown(chips_html, unsafe_allow_html=True)
        else:
            st.success("Your text demonstrates good vocabulary diversity! No repetitive basic words flagged.")
    else:
        st.info("Enter text to inspect vocabulary enhancements.")

# ---------------------------------------------------------
# TAB 6: Urdu-English Support
# ---------------------------------------------------------
with tab_urdu:
    st.markdown("#### 🌐 Urdu-English Translation & Multilingual Support")
    st.caption("Assists Pakistani and bilingual students by translating Roman Urdu phrases into standard English.")
    
    urdu_input = st.text_input(
        "Enter Roman Urdu sentence (e.g. `mein school ja raha hoon aur mere paas gaari hai`):",
        value=""
    )
    
    if st.button("Translate to English", key="btn_translate_urdu_tab"):
        if urdu_input.strip():
            translated = enhancer_engine.translate_urdu(urdu_input)
            st.success(f"**English Translation**: {translated}")
            if st.button("Load Translation into Main Canvas", key="btn_load_urdu_trans_tab"):
                set_text(translated)
        else:
            st.warning("Please enter a Roman Urdu phrase.")
            
    st.markdown("---")
    st.markdown("##### Common Examples")
    c_ex1, c_ex2 = st.columns(2)
    with c_ex1:
        st.code("mein school ja raha hoon -> I am going to school\nmere paas gaari hai -> I have a car")
    with c_ex2:
        st.code("yeh bohot acha hai -> this is very good\nwoh university ja rahi hai -> she is going to university")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9ca3af; font-size: 0.85rem;'>"
    "Mini Grammarly AI Writing Assistant | Built for Academic NLP & Deep Learning Coursework"
    "</div>",
    unsafe_allow_html=True
)
