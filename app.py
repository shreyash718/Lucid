# streamlit_app.py
"""
Lucid — Streamlit app: PDF -> Summary + MCQ Trainer

Features:
 - Upload PDF (or use local path), extract sections by heading (blue/large/numbered).
 - Summarize sections (Gemini via langchain) with configurable summary length.
 - After reading a summary, user may take 5 MCQs generated from that summary.
 - Progress (correct out of 5 per paragraph) stored in progress.json and shown in a bar chart.
 - Download all summaries as TXT.

Run:
    streamlit run streamlit_app.py
"""

import streamlit as st
import fitz
import re
import json
import os
import time
import ast
from collections import Counter
from typing import List, Dict

# LangChain / Gemini interface (user must have creds configured)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# ---------- Page config ----------
st.set_page_config(page_title="Lucid", layout="wide")
st.title("🌟 Lucid — Your Smart Tutor 🤖📚")

# ---------- Config / LLM ----------
MODEL_NAME = "gemini-2.5-flash"
MODEL_TEMP = 0.2
model = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=MODEL_TEMP)
parser = StrOutputParser()

PROGRESS_FILE = "progress.json"
OUTPUT_SUMMARIES = "summaries.txt"

BLUE_MIN = 100
BLUE_RATIO = 1.25
SIZE_MULTIPLIER = 1.15
MIN_SECTION_TEXT_LEN = 100
SAMPLE_N = 6
SUMMARY_TARGET_LINES = 10
LLM_MAX_RETRIES = 2
LLM_RETRY_SLEEP = 1.5

# ---------- Helpers ----------
def rgb_from_int(color_int: int):
    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255
    return r, g, b

def is_blueish(r:int,g:int,b:int, blue_min=BLUE_MIN, ratio=BLUE_RATIO) -> bool:
    avg_rg = (r+g)/2.0 if (r+g)>0 else 0.0
    return (b >= blue_min) and (avg_rg == 0 or (b/(avg_rg+1e-9)) >= ratio)

def clean_join(lines: List[str]) -> str:
    s = " ".join(lines)
    s = re.sub(r'(\w)-\s+(\w)', r'\1\2', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def likely_page_number_or_footer(text: str) -> bool:
    t = text.strip()
    if not t: return True
    if re.fullmatch(r'\d{1,4}', t): return True
    if re.fullmatch(r'\d{4}(-\d{2,4})?', t): return True
    return False

# ---------- PDF extraction ----------
def extract_sections_by_heading(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages_lines = []
    for pno, page in enumerate(doc):
        page_dict = page.get_text("dict")
        lines_on_page = []
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = []
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    color = span.get("color", 0)
                    r,g,b = rgb_from_int(color)
                    size = span.get("size", 0) or 0.0
                    bbox = span.get("bbox", None)
                    font = span.get("font", "")
                    spans.append({"text": text.strip(), "r": r, "g": g, "b": b, "size": size, "bbox": bbox, "font": font})
                if not spans:
                    continue
                y0 = min((s['bbox'][1] if s['bbox'] else 0) for s in spans)
                y1 = max((s['bbox'][3] if s['bbox'] else 0) for s in spans)
                avg_size = sum(s['size'] for s in spans if s['size']) / max(1, sum(1 for s in spans if s['size']))
                full_text = " ".join(s['text'] for s in spans)
                lines_on_page.append({"spans": spans, "text": full_text, "y0": y0, "y1": y1, "avg_size": avg_size, "pno": pno})
        lines_on_page.sort(key=lambda x: x["y0"])
        pages_lines.append(lines_on_page)

    # detect repeated headers/footers
    top_lines = []
    bottom_lines = []
    for pl in pages_lines:
        top_lines.extend([ln["text"].strip() for ln in pl[:SAMPLE_N] if ln["text"].strip()])
        bottom_lines.extend([ln["text"].strip() for ln in pl[-SAMPLE_N:] if ln["text"].strip()])
    top_counts = Counter(top_lines)
    bottom_counts = Counter(bottom_lines)
    num_pages = max(1, len(pages_lines))
    def frequent_set(counter, frac=0.4):
        thresh = max(1, int(num_pages * frac))
        return {k for k,v in counter.items() if v >= thresh}
    frequent_tops = frequent_set(top_counts)
    frequent_bottoms = frequent_set(bottom_counts)

    all_sizes = [ln["avg_size"] for pl in pages_lines for ln in pl if ln["avg_size"] > 0]
    typical_size = (sum(all_sizes)/len(all_sizes)) if all_sizes else 10.0
    section_number_re = re.compile(r'^\s*\d+(\.\d+)*\b')

    sections = []
    current_heading = None
    current_lines = []
    current_page = None

    for pno, plines in enumerate(pages_lines):
        i = 0
        while i < len(plines):
            ln = plines[i]
            text = ln["text"].strip()
            if not text:
                i += 1
                continue
            if text in frequent_tops or text in frequent_bottoms or likely_page_number_or_footer(text):
                i += 1
                continue

            blueish = any(is_blueish(s['r'], s['g'], s['b']) for s in ln["spans"])
            larger = ln["avg_size"] >= (typical_size * SIZE_MULTIPLIER)
            starts_section_num = bool(section_number_re.match(text))
            is_upper = (len(text) <= 120 and text == text.upper() and len(text.split()) <= 8)

            if blueish or larger or starts_section_num or is_upper:
                heading_parts = [text]
                j = i + 1
                while j < len(plines):
                    ln2 = plines[j]
                    if abs(ln2["y0"] - ln["y0"]) < 12 or (ln2["y0"] - ln["y0"]) < 18:
                        text2 = ln2["text"].strip()
                        blue2 = any(is_blueish(s['r'], s['g'], s['b']) for s in ln2["spans"])
                        large2 = ln2["avg_size"] >= (typical_size * SIZE_MULTIPLIER)
                        upper2 = (len(text2) <= 120 and text2 == text2.upper() and len(text2.split()) <= 8)
                        num2 = bool(section_number_re.match(text2))
                        if blue2 or large2 or upper2 or num2:
                            heading_parts.append(text2)
                            j += 1
                            continue
                    break
                heading_text = clean_join(heading_parts)
                if current_heading is not None and current_lines:
                    sections.append({"heading": current_heading, "text": clean_join(current_lines), "page_no": current_page + 1})
                current_heading = heading_text
                current_lines = []
                current_page = pno
                i = j
                continue
            else:
                if current_heading is None:
                    current_heading = "INTRO"
                    current_page = pno
                current_lines.append(text)
                i += 1

    if current_heading is not None and current_lines:
        sections.append({"heading": current_heading, "text": clean_join(current_lines), "page_no": current_page + 1})

    filtered = []
    for s in sections:
        txt = s["text"].strip()
        if len(txt) < 40:
            if filtered:
                filtered[-1]["text"] = clean_join([filtered[-1]["text"], txt])
            else:
                filtered.append(s)
        else:
            filtered.append(s)
    return filtered

def merge_short_sections(sections: List[Dict], min_length: int = MIN_SECTION_TEXT_LEN) -> List[Dict]:
    if not sections:
        return sections
    merged = []
    buffer = None
    for sec in sections:
        txt = sec["text"].strip()
        hdr = sec["heading"].strip()
        if buffer is None:
            buffer = {"heading": hdr, "text": txt, "page_no": sec.get("page_no", None)}
            continue
        if len(txt) < min_length:
            buffer["text"] = clean_join([buffer["text"], txt])
        else:
            merged.append(buffer)
            buffer = {"heading": hdr, "text": txt, "page_no": sec.get("page_no", None)}
    if buffer is not None:
        merged.append(buffer)
    if len(merged) > 1 and len(merged[0]["text"]) < min_length:
        merged[1]["text"] = clean_join([merged[0]["text"], merged[1]["text"]])
        merged = merged[1:]
    return merged

# ---------- LLM prompt helpers ----------
def make_summary_prompt(target_lines: int = SUMMARY_TARGET_LINES) -> PromptTemplate:
    txt = (
        "You are an expert educational summarizer.\n\n"
        "Given the paragraph below, produce a structured, clear numbered summary.\n"
        f"- Produce roughly {target_lines} numbered lines (if paragraph is short, produce fewer lines).\n"
        "- Number each line starting from 1.\n"
        "- Use clear academic English and paraphrase; do not copy long verbatim passages.\n\n"
        "Paragraph:\n{paragraph}\n\nNow write the numbered summary:"
    )
    return PromptTemplate(input_variables=["paragraph"], template=txt)

def summariser_with_retries(paragraph_text: str, model, target_lines: int = SUMMARY_TARGET_LINES):
    prompt = make_summary_prompt(target_lines)
    chain = prompt | model | parser
    attempt = 0
    while attempt <= LLM_MAX_RETRIES:
        try:
            out = chain.invoke({"paragraph": paragraph_text})
            return (out if isinstance(out, str) else str(out)).strip()
        except Exception as e:
            attempt += 1
            if attempt > LLM_MAX_RETRIES:
                return f"[ERROR] summarization failed after {LLM_MAX_RETRIES}: {e}"
            time.sleep(LLM_RETRY_SLEEP)

def make_mcq_prompt() -> PromptTemplate:
    template_text = (
        "You are an exam-writer. Given the summary below, generate 5 multiple-choice questions "
        "(each with 4 answer choices) that test understanding of the content.\n\n"
        "Return a JSON array of 5 objects with fields: \"question\" (string), "
        "\"choices\" (array of 4 strings), \"answer\" (index 0-3 of the correct choice), "
        "\"explanation\" (short explanation).\n\n"
        "Example output (literal JSON example shown for format):\n"
        "[{{\"question\":\"...\",\"choices\":[\"a\",\"b\",\"c\",\"d\"],\"answer\":1,\"explanation\":\"...\"}},\n"
        " {{\"question\":\"...\",\"choices\":[\"a\",\"b\",\"c\",\"d\"],\"answer\":2,\"explanation\":\"...\"}}, ...]\n\n"
        "Summary:\n{summary}\n\n"
        "Now produce the JSON array EXACTLY (no extra commentary, no surrounding text)."
    )
    return PromptTemplate(input_variables=["summary"], template=template_text)

def generate_mcqs_from_summary(summary_text: str):
    prompt = make_mcq_prompt()
    chain = prompt | model | parser
    attempt = 0
    while attempt <= LLM_MAX_RETRIES:
        try:
            out = chain.invoke({"summary": summary_text})
            out_s = out if isinstance(out, str) else str(out)
            try:
                data = json.loads(out_s)
            except Exception:
                m = re.search(r'(\[.*\])', out_s, re.S)
                if m:
                    candidate = m.group(1)
                    try:
                        data = json.loads(candidate)
                    except Exception:
                        data = ast.literal_eval(candidate)
                else:
                    data = ast.literal_eval(out_s)
            mcqs = []
            for it in data:
                if not isinstance(it, dict):
                    continue
                q = it.get("question") or it.get("q") or ""
                choices = it.get("choices") or it.get("options") or []
                if not isinstance(choices, list):
                    continue
                while len(choices) < 4:
                    choices.append("")
                choices = [str(c).strip() for c in choices[:4]]
                answer = it.get("answer")
                if isinstance(answer, str):
                    lm = {"A":0,"B":1,"C":2,"D":3,"a":0,"b":1,"c":2,"d":3}
                    answer = lm.get(answer.strip(), None)
                if answer is None:
                    for key in ("correct","ans","correct_index"):
                        if key in it:
                            answer = it.get(key)
                            break
                explanation = it.get("explanation") or it.get("explain") or ""
                mcqs.append({
                    "question": str(q).strip(),
                    "choices": choices,
                    "answer": int(answer) if isinstance(answer, int) and 0<=answer<4 else None,
                    "explanation": str(explanation).strip()
                })
            return mcqs
        except Exception as e:
            attempt += 1
            if attempt > LLM_MAX_RETRIES:
                st.error(f"Could not generate MCQs: {e}")
                return []
            time.sleep(LLM_RETRY_SLEEP)

# ---------- persistence & UI helpers ----------
def load_progress() -> Dict[int,int]:
    if os.path.exists(PROGRESS_FILE):
        try:
            raw = json.load(open(PROGRESS_FILE, "r", encoding="utf-8"))
            return {int(k): int(v) for k,v in raw.items()}
        except Exception:
            return {}
    return {}

def save_progress(progress: Dict[int,int]):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as fh:
        json.dump({str(k): v for k,v in progress.items()}, fh, indent=2)

def append_summary_to_file(idx: int, heading: str, page_no: int, summary: str):
    with open(OUTPUT_SUMMARIES, "a", encoding="utf-8") as fh:
        fh.write(f"\n--- Summary Para {idx} | Heading: {heading} | Page: {page_no} ---\n\n")
        fh.write(summary + "\n\n")

def read_summaries_text() -> str:
    if os.path.exists(OUTPUT_SUMMARIES):
        return open(OUTPUT_SUMMARIES, "r", encoding="utf-8").read()
    return ""

def render_progress_chart(progress: Dict[int,int]):
    if not progress:
        st.info("No progress yet — take quizzes to populate the learning curve.")
        return
    items = sorted(progress.items(), key=lambda x:int(x[0]))
    x = [f"Para {k}" for k,_ in items]
    y = [v for _,v in items]
    fig = go.Figure([go.Bar(x=x, y=y, text=y, textposition="auto")])
    fig.update_yaxes(range=[0,5], title_text="Correct answers (out of 5)")
    fig.update_layout(title_text="Learning progress by paragraph", height=350, margin=dict(l=20,r=20,t=50,b=20))
    st.plotly_chart(fig, use_container_width=True)

# ---------- Sidebar & session init ----------
# initialize widget-backed keys BEFORE creating widgets
if "ui_summary_lines" not in st.session_state:
    st.session_state.ui_summary_lines = SUMMARY_TARGET_LINES
if "ui_auto_advance" not in st.session_state:
    st.session_state.ui_auto_advance = False

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    pdf_path_input = st.text_input("Or enter local PDF path", "")

    st.write("---")
    st.write("Settings")
    # create widgets (Streamlit will store values into st.session_state using the given keys)
    st.number_input("Summary lines", min_value=3, max_value=200, value=st.session_state.ui_summary_lines, key="ui_summary_lines")
    st.checkbox("Auto-advance after summarize", value=st.session_state.ui_auto_advance, key="ui_auto_advance")

    st.write("---")
    st.markdown("### Summaries")
    summaries_text = read_summaries_text()
    if summaries_text:
        st.download_button("Download summaries (TXT)", data=summaries_text, file_name="summaries.txt", mime="text/plain")
    else:
        st.caption("No summaries yet.")

    st.write("---")
    if st.button("Reset progress & summaries"):
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
        if os.path.exists(OUTPUT_SUMMARIES):
            os.remove(OUTPUT_SUMMARIES)
        for k in ["sections","merged","index","last_summary","last_para_idx","progress","mcqs","mcq_answers","mcq_taken"]:
            if k in st.session_state:
                del st.session_state[k]
        # use current stable API to rerun app
        try:
            st.rerun()
        except Exception:
            # fallback: just stop (user will manually refresh)
            st.stop()

# session defaults
if "sections" not in st.session_state: st.session_state.sections = []
if "merged" not in st.session_state: st.session_state.merged = []
if "index" not in st.session_state: st.session_state.index = 0
if "last_summary" not in st.session_state: st.session_state.last_summary = None
if "last_para_idx" not in st.session_state: st.session_state.last_para_idx = None
if "progress" not in st.session_state: st.session_state.progress = load_progress()
if "mcqs" not in st.session_state: st.session_state.mcqs = []
if "mcq_answers" not in st.session_state: st.session_state.mcq_answers = {}
if "mcq_taken" not in st.session_state: st.session_state.mcq_taken = {}

# ---------- Extract button ----------
if uploaded_file is not None or pdf_path_input.strip():
    if uploaded_file is not None:
        tmp_path = os.path.join(".", uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_path = tmp_path
    else:
        pdf_path = pdf_path_input.strip()

    if st.button("Extract Sections"):
        with st.spinner("Extracting sections from PDF..."):
            try:
                secs = extract_sections_by_heading(pdf_path)
                merged = merge_short_sections(secs, min_length=MIN_SECTION_TEXT_LEN)
                st.session_state.sections = secs
                st.session_state.merged = merged
                st.session_state.index = 0
                st.session_state.last_summary = None
                st.session_state.last_para_idx = None
                st.session_state.mcqs = []
                st.session_state.mcq_answers = {}
                st.session_state.mcq_taken = {}
                st.success(f"Extracted {len(merged)} merged sections.")
            except Exception as e:
                st.error(f"Extraction failed: {e}")

# ---------- Main layout ----------
left, right = st.columns([2,1])
with right:
    st.header("Progress")
    render_progress_chart(st.session_state.progress)
    st.write("---")
    st.header("Session")
    st.write(f"Current section: {st.session_state.index + 1} / {len(st.session_state.merged)}")
    st.write(f"Auto-advance: {'On' if st.session_state.ui_auto_advance else 'Off'}")
    st.write(f"Summary lines: {st.session_state.ui_summary_lines}")

with left:
    if not st.session_state.merged:
        st.info("Upload a PDF and click 'Extract Sections' to start.")
    else:
        idx = st.session_state.index
        total = len(st.session_state.merged)
        st.markdown(f"### Section {idx+1} / {total}")

        prev_pending = (
            st.session_state.last_summary is not None
            and st.session_state.last_para_idx is not None
            and (not st.session_state.mcq_taken.get(st.session_state.last_para_idx, False))
        )

        # If prev pending and MCQs not generated -> prompt to take/skip
        if prev_pending and (not st.session_state.mcqs):
            st.info(f"You have a summary from Para {st.session_state.last_para_idx} that requires MCQs before moving on.")
            c1, c2 = st.columns(2)
            with c1:
                take_prev_now = st.button("Take MCQs for previous summary now", key=f"take_prev_now_{idx}")
            with c2:
                skip_prev = st.button("Skip MCQs for previous summary", key=f"skip_prev_{idx}")

            if skip_prev:
                st.session_state.mcq_taken[st.session_state.last_para_idx] = True
                st.success("Skipped MCQs for previous summary. You may proceed.")

            if take_prev_now:
                with st.spinner("Generating MCQs for previous summary..."):
                    mcqs = generate_mcqs_from_summary(st.session_state.last_summary)
                    if not mcqs:
                        st.error("Failed to generate MCQs for the previous summary. You can skip or try again.")
                    else:
                        st.session_state.mcqs = mcqs
                        st.session_state.mcq_answers = {str(i): None for i in range(len(mcqs))}

        # If MCQs exist -> render quiz form
        if st.session_state.mcqs:
            st.subheader(f"MCQ Quiz for Para {st.session_state.last_para_idx}")
            with st.form(key=f"mcq_form_{st.session_state.last_para_idx}"):
                for i, q in enumerate(st.session_state.mcqs, start=1):
                    choices = q.get("choices", [])
                    label = f"Q{i}: {q.get('question')}"
                    options = [f"{chr(65+j)}. {choices[j]}" for j in range(len(choices))]
                    sel = st.radio(label, options=options, key=f"mcq_{st.session_state.last_para_idx}_{i}")
                    if sel:
                        st.session_state.mcq_answers[str(i-1)] = ord(sel[0]) - 65
                submit_mcq = st.form_submit_button("Submit MCQ Answers")
                if submit_mcq:
                    correct = 0
                    for i, q in enumerate(st.session_state.mcqs):
                        chosen = st.session_state.mcq_answers.get(str(i))
                        correct_idx = q.get("answer")
                        if chosen is not None and chosen == correct_idx:
                            correct += 1
                    para_idx = st.session_state.last_para_idx or 0
                    st.session_state.progress[para_idx] = int(correct)
                    save_progress(st.session_state.progress)
                    st.session_state.mcq_taken[para_idx] = True
                    st.success(f"Quiz submitted. You got {correct}/{len(st.session_state.mcqs)} correct.")
                    with st.expander("Show explanations"):
                        for qi, q in enumerate(st.session_state.mcqs, start=1):
                            ans_idx = q.get("answer")
                            expl = q.get("explanation") or "No explanation provided."
                            if ans_idx is None:
                                st.write(f"Q{qi}: correct answer unknown. Explanation: {expl}")
                            else:
                                st.write(f"Q{qi}: Correct = {chr(65+ans_idx)}. Explanation: {expl}")
                    st.session_state.mcqs = []
                    st.session_state.mcq_answers = {}
                    # auto-advance
                    if st.session_state.ui_auto_advance:
                        if st.session_state.index < len(st.session_state.merged)-1:
                            st.session_state.index += 1

        # If no pending MCQs, show preview & controls (capture clicks then apply)
        if not st.session_state.mcqs:
            sec = st.session_state.merged[idx]
            heading = sec["heading"]
            text = sec["text"]
            page_no = sec.get("page_no")
            st.subheader(heading)
            st.write(f"Page: {page_no}")
            st.write(text[:2000])
            st.write("---")

            col_s1, col_s2, col_s3 = st.columns([1,1,2])
            with col_s1:
                summarize_clicked = st.button("Summarize this section", key=f"summarize_{idx}")
            with col_s2:
                skip_clicked = st.button("Skip this section", key=f"skip_{idx}")
            with col_s3:
                next_clicked = st.button("Next section", key=f"next_{idx}")

            # process summarize click
            if summarize_clicked:
                st.info("Summarizing (LLM)...")
                summary_lines = st.session_state.ui_summary_lines
                summary = summariser_with_retries(text, model, target_lines=summary_lines)
                st.session_state.last_summary = summary
                st.session_state.last_para_idx = st.session_state.index + 1
                append_summary_to_file(st.session_state.last_para_idx, heading, page_no, summary)
                st.success("Summary generated and saved.")
                st.subheader("Summary (preview)")
                st.write(summary[:4000])

                # immediate quiz options
                st.markdown("**Take a 5-question quiz for this summary now?**")
                c1, c2 = st.columns(2)
                with c1:
                    take_now = st.button("Take MCQs now", key=f"take_now_{st.session_state.last_para_idx}")
                with c2:
                    take_later = st.button("Take MCQs later (ask before next section)", key=f"take_later_{st.session_state.last_para_idx}")

                if take_later:
                    st.session_state.mcq_taken[st.session_state.last_para_idx] = False
                    st.info("You will be asked MCQs for this summary before moving on.")
                    if st.session_state.ui_auto_advance:
                        if st.session_state.index < len(st.session_state.merged)-1:
                            st.session_state.index += 1

                if take_now:
                    with st.spinner("Generating MCQs for this summary..."):
                        mcqs = generate_mcqs_from_summary(st.session_state.last_summary)
                        if not mcqs:
                            st.error("Could not generate MCQs for this summary. Try again later.")
                            st.session_state.mcq_taken[st.session_state.last_para_idx] = False
                        else:
                            st.session_state.mcqs = mcqs
                            st.session_state.mcq_answers = {str(i): None for i in range(len(mcqs))}
                            st.session_state.mcq_taken[st.session_state.last_para_idx] = False

            # process next / skip clicks single-run
            if next_clicked or skip_clicked:
                prev_pending_local = (
                    st.session_state.last_summary is not None
                    and st.session_state.last_para_idx is not None
                    and (not st.session_state.mcq_taken.get(st.session_state.last_para_idx, False))
                )
                if prev_pending_local and (not st.session_state.mcqs):
                    with st.spinner("You must handle MCQs for previous summary first. Generating MCQs..."):
                        mcqs = generate_mcqs_from_summary(st.session_state.last_summary)
                        if not mcqs:
                            st.error("Couldn't generate MCQs for previous summary. You can skip them to proceed.")
                        else:
                            st.session_state.mcqs = mcqs
                            st.session_state.mcq_answers = {str(i): None for i in range(len(mcqs))}
                else:
                    if st.session_state.index < len(st.session_state.merged) - 1:
                        st.session_state.index += 1
                    else:
                        st.info("This is the last section.")

# ---------- Footer ----------
st.write("---")
st.caption("Built with PyMuPDF + Gemini (via langchain) + Streamlit. Monitor model usage & quota.")
