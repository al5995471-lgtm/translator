import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import anthropic
import ast
import base64
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import os
import re
import time
import requests
import numpy as np
import pandas as pd

try:
    import altair as alt
except Exception:
    alt = None

st.set_page_config(page_title="المترجم العلمي للمحاضرات", page_icon="🔬")
st.title("🔬 أدوات المحاضرات العلمية")

provider = st.radio("اختر المزوّد:", ["Gemini", "Claude"], horizontal=True)
KEY_NAMES = {"Gemini": "GEMINI_API_KEY", "Claude": "ANTHROPIC_API_KEY"}
KEY_NAME = KEY_NAMES[provider]

try:
    api_key = st.secrets.get(KEY_NAME, "")
except Exception:
    api_key = ""
if not api_key:
    api_key = st.text_input(f"أدخل مفتاح {provider} API بتاعك:", type="password")

PAGES_PER_REQUEST = 2
WORKERS = 2
MAX_CHARS = 40000
FONT_DIR = "fonts"
FONT_FILE = "Amiri-Regular.ttf"
FONT_URLS = [
    "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/amiri/Amiri-Regular.ttf",
]
CLAUDE_FALLBACK = ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"]
SRC_TYPES = ["pdf", "png", "jpg", "jpeg"]

SYSTEM_PROMPT = """
أنت مترجم أكاديمي متخصص في الكيمياء والأحياء وعلوم الرياضة.
ستستلم لكل سلايد: صورته + النص المستخرج منه (قد يكون مبعثرًا).
قواعد الترجمة:
1. ترجم بالمعنى إلى عربية علمية سليمة وسلسة، وليس ترجمة حرفية.
2. اكتب المصطلح العلمي بالعربي وبعده بين قوسين المصطلح الإنجليزي، مرة واحدة فقط عند أول ظهور في الصفحة.
3. اترك المعادلات والرموز والوحدات بالإنجليزية، واكتب الأرقام السفلية والعلوية بشكل Unicode مثل H₂O و CO₂ و k₀ و e⁻ᴱᵃ/ᴿᵀ.
4. اكتب كل معادلة كيميائية أو رياضية في سطر مستقل بالكامل، بدون أي كلمات عربية في نفس السطر.
5. انظر إلى صورة السلايد لتفهم ترتيبه. ركّب الجمل المقطوعة بشكل صحيح.
6. إذا كان السلايد يحتوي على رسم بياني أو مخطط أو رسم توضيحي:
   - لا تكتب كل تسمية من تسميات الرسم في سطر منفصل.
   - اكتب سطرًا واحدًا بعنوان "عناصر الرسم:" يجمع التسميات الأساسية مترجمة، مثل: ΔG = تغير الطاقة الحرة، Ea = طاقة التنشيط، uncatalyzed = بدون حفاز.
   - ثم اكتب شرحًا عربيًا قصيرًا (جملتان إلى أربع) يوضح ماذا يبيّن الرسم.
7. لا تستخدم أي رموز Markdown مثل ** أو ###. اكتب كل بند في سطر مستقل.
8. اكتب الترجمة فقط بدون أي مقدمة.
9. كل سلايد يبدأ بسطر على شكل ===PAGE رقم=== . اكتب نفس هذا السطر بالضبط قبل ترجمة كل سلايد.
"""

OCR_PROMPT = """
You are an expert OCR reader for chemistry and science study material.
You receive photos or scans of lecture slides or exam papers (they may be rotated, skewed or taken at an angle).
Rules:
1. Transcribe ALL printed and handwritten text exactly as written, keeping the original language, numbering and question order.
2. Write formulas with Unicode (H₂O, CO₂, k₀).
3. For every chemical structure, Newman/Fischer projection, graph, diagram or table, add a short description in square brackets at the place where it appears,
   for example [Compound A: Newman projection with OH, F, H on the front carbon].
4. Each image starts with a line like ===IMAGE n===. Keep these separator lines.
5. Output only the transcription, no comments.
"""

SOLVE_PROMPT = """
You are a university professor of organic chemistry and stereochemistry.
You receive photos or scans of an exam paper (they may be rotated, skewed, or include structures, Newman or Fischer projections, tables).
Instructions:
1. Read every question carefully. For structure questions, analyze the drawing step by step
   (for example CIP priorities for R/S and E/Z) before you answer.
2. For each question give: the question number, the answer, and a one or two sentence reason.
3. If a question or a drawing is unreadable or ambiguous, say so clearly instead of guessing.
4. Write the reasons in {expl_lang}. Keep technical terms and the answer letters/words as in the exam.
5. No Markdown. Use exactly this format for every question, one block per question:
Q1) Answer: ... | Reason: ...
6. Output only the answers, no introduction.
"""

ARABIC = re.compile(r'[\u0600-\u06FF]')
EQUATION = re.compile(r'[→⇌=+<>]')
BAD_WORDS = ["robotics", "tts", "image", "live", "embedding", "audio", "computer",
             "aqa", "imagen", "veo", "lyria", "deep-research", "learnlm", "gemma",
             "vision", "thinking-exp"]


# ---------------------------------------------------------------- أدوات عامة
def clean_markdown(text):
    text = re.sub(r'\*+(.*?)\*+', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    return text.strip()


def esc(text):
    return html.escape(str(text)).replace("\n", "<br>")


@st.cache_resource(show_spinner="بنزّل الخط العربي...")
def ensure_font():
    os.makedirs(FONT_DIR, exist_ok=True)
    path = os.path.join(FONT_DIR, FONT_FILE)
    if os.path.exists(path) and os.path.getsize(path) > 10000:
        return True
    for url in FONT_URLS:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 10000:
                with open(path, "wb") as f:
                    f.write(r.content)
                return True
        except Exception:
            continue
    return False


@st.cache_data(show_spinner=False)
def get_models(provider, key):
    if provider == "Claude":
        try:
            client = anthropic.Anthropic(api_key=key)
            names = [m.id for m in client.models.list(limit=50).data]
        except Exception:
            names = []
        names = [n for n in names if n.startswith("claude")] or CLAUDE_FALLBACK

        def rank(n):
            return 0 if "sonnet" in n else (1 if "haiku" in n else 2)
        return sorted(names, key=lambda n: (rank(n), n))

    genai.configure(api_key=key)
    names = []
    for m in genai.list_models():
        n = m.name.replace("models/", "")
        if "generateContent" not in m.supported_generation_methods:
            continue
        if not n.startswith("gemini") or any(b in n for b in BAD_WORDS):
            continue
        names.append(n)

    def rank(n):
        if "flash" in n and "lite" not in n:
            return 0
        return 1 if "flash" in n else 2
    return sorted(names, key=lambda n: (rank(n), n))


def ask_gemini(model_name, items, system, key):
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name)
    content = [system]
    for kind, data in items:
        content.append({"mime_type": "image/png", "data": data} if kind == "image" else data)
    r = model.generate_content(
        content,
        generation_config={"temperature": 0.2},
        request_options={"timeout": 150, "retry": None},
    )
    return r.text


def ask_claude(model_name, items, system, key):
    client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=150)
    content = []
    for kind, data in items:
        if kind == "image":
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.b64encode(data).decode()}})
        else:
            content.append({"type": "text", "text": data})
    r = client.messages.create(
        model=model_name,
        max_tokens=8000,
        temperature=0.2,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    return "".join(b.text for b in r.content if b.type == "text")


def parse_delay(msg):
    m = re.search(r'(?:retry|try again) in\s+((?:\d+h)?(?:\d+m)?(?:\d+(?:\.\d+)?s)?)', msg)
    if not m or not m.group(1):
        return None
    g = m.group(1)
    total = 0.0
    for pat, mult in ((r'(\d+)h', 3600), (r'(\d+)m', 60), (r'(\d+(?:\.\d+)?)s', 1)):
        x = re.search(pat, g)
        if x:
            total += float(x.group(1)) * mult
    return total


def call_model(provider, model_name, items, system=SYSTEM_PROMPT, key=None):
    key = key or api_key
    for attempt in range(5):
        try:
            if provider == "Claude":
                return ask_claude(model_name, items, system, key)
            return ask_gemini(model_name, items, system, key)
        except Exception as e:
            msg = str(e)
            low = msg.lower()
            if "limit: 0" in msg:
                raise RuntimeError("النموذج ده مش متاح في الباقة المجانية، اختار نموذج تاني من القايمة.")
            if "credit balance" in low:
                raise RuntimeError("رصيد Claude خلص، اشحن الحساب من console.anthropic.com.")
            if "401" in msg or "authentication" in low or "invalid api key" in low:
                raise RuntimeError("المفتاح غلط أو منتهي.")
            if "429" in msg or "rate_limit" in low or "rate limit" in low or "overloaded" in low:
                delay = parse_delay(msg)
                if delay is not None and delay > 90:
                    raise RuntimeError(
                        f"وصلت للحد اليومي لهذا النموذج. جرّب نموذج تاني أو ارجع بعد {delay / 3600:.1f} ساعة تقريبًا.")
                time.sleep(min((delay or 30) + 2, 90))
                continue
            if attempt == 4:
                raise RuntimeError(msg[:200])
            time.sleep(3)
    raise RuntimeError("وصلت لحد الاستخدام، استنى دقيقة وجرّب تاني.")


def detect_lang(text):
    """يحدد لغة المصدر: عربي أو إنجليزي (أو غيره)"""
    sample = text[:6000]
    ar = len(re.findall(r'[\u0600-\u06FF]', sample))
    lat = len(re.findall(r'[A-Za-z]', sample))
    if ar > lat:
        return "Arabic", True
    return "the same language as the text (for example English)", False


def parse_json_block(raw):
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?', '', raw).strip()
    raw = re.sub(r'```$', '', raw).strip()
    a, b = raw.find('{'), raw.rfind('}')
    if a == -1 or b == -1:
        raise ValueError("الرد مش JSON")
    return json.loads(raw[a:b + 1])


# ---------------------------------------------------------------- قراءة الصور
def page_png(pg, max_side=1600):
    r = pg.rect
    z = max_side / max(r.width, r.height)
    return pg.get_pixmap(matrix=fitz.Matrix(z, z)).tobytes("png")


def file_to_pngs(f, max_side=1600):
    """أي ملف (PDF أو صورة) -> قايمة صور PNG"""
    name = f.name.lower()
    if name.endswith(".pdf"):
        doc = fitz.open(stream=f.getvalue(), filetype="pdf")
        return [page_png(pg, max_side) for pg in doc]
    ext = name.rsplit(".", 1)[-1]
    ext = "jpeg" if ext == "jpg" else ext
    doc = fitz.open(stream=f.getvalue(), filetype=ext)
    return [page_png(doc[0], max_side)]


def ocr_images(pngs, provider, model_name, key):
    out = []
    for s in range(0, len(pngs), 3):
        batch = pngs[s:s + 3]
        items = []
        for j, p in enumerate(batch):
            items.append(("text", f"===IMAGE {s + j + 1}==="))
            items.append(("image", p))
        out.append(call_model(provider, model_name, items, system=OCR_PROMPT, key=key))
    return "\n\n".join(out)


def read_sources(files, provider, model_name, key):
    """نص من ملفات PDF (فيها نص) + صور + PDF ممسوح"""
    texts, scans = [], []
    for f in files:
        name = f.name.lower()
        if name.endswith(".pdf"):
            doc = fitz.open(stream=f.getvalue(), filetype="pdf")
            for i, pg in enumerate(doc):
                t = pg.get_text().strip()
                if len(t) >= 40:
                    texts.append(f"--- {f.name} / page {i + 1} ---\n{t}")
                else:
                    scans.append(page_png(pg))
        else:
            scans.append(file_to_pngs(f)[0])
    if scans:
        with st.spinner(f"بقرأ {len(scans)} صورة/صفحة..."):
            texts.append(ocr_images(scans, provider, model_name, key))
    return "\n\n".join(texts)


# ---------------------------------------------------------------- الترجمة
def translate_chunk(provider, model_name, chunk):
    start = time.time()
    items = []
    for i, text, png in chunk:
        items.append(("text", f"===PAGE {i + 1}==="))
        items.append(("image", png))
        items.append(("text", f"النص المستخرج:\n{text}"))
    raw = clean_markdown(call_model(provider, model_name, items))

    parts = re.split(r'===\s*PAGE\s*(\d+)\s*===', raw)
    out = {}
    for k in range(1, len(parts) - 1, 2):
        out[int(parts[k]) - 1] = parts[k + 1].strip()
    if not out:
        out[chunk[0][0]] = raw
    return out, time.time() - start


CSS = """
@font-face { font-family: amiri; src: url(Amiri-Regular.ttf); }
body { font-family: amiri; font-size: 13px; line-height: 1.5; }
p { margin: 0 0 4px 0; }
.eq { text-align: center; font-size: 15px; }
.ttl { font-weight: bold; color: #1f4e79; font-size: 14px; }
"""


def build_html(num, text):
    rows = [f'<p dir="rtl" class="ttl">سلايد رقم {num + 1}</p>']
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        safe = html.escape(line)
        if not ARABIC.search(line) and EQUATION.search(line):
            rows.append(f'<p dir="ltr" class="eq">{safe}</p>')
        else:
            rows.append(f'<p dir="rtl">{safe}</p>')
    return "".join(rows)


def build_pdf(src_doc, results):
    out = fitz.open()
    W, H, M = 595, 842, 30
    arch = fitz.Archive(FONT_DIR)

    for num in sorted(results):
        page = out.new_page(width=W, height=H)

        src = src_doc.load_page(num)
        pix = src.get_pixmap(dpi=110)
        max_w, max_h = W - 2 * M, 330
        ratio = min(max_w / pix.width, max_h / pix.height)
        iw, ih = pix.width * ratio, pix.height * ratio
        x0 = (W - iw) / 2
        page.insert_image(fitz.Rect(x0, M, x0 + iw, M + ih), pixmap=pix)
        page.draw_rect(fitz.Rect(x0, M, x0 + iw, M + ih), color=(0.8, 0.8, 0.8), width=0.5)

        body = build_html(num, results[num])
        top = M + ih + 12
        rect = fitz.Rect(M, top, W - M, H - M)
        spare, _scale = page.insert_htmlbox(
            rect, body, css=CSS, archive=arch, scale_low=0.55
        )
        if spare < 0:
            page2 = out.new_page(width=W, height=H)
            page2.insert_htmlbox(
                fitz.Rect(M, M, W - M, H - M), body, css=CSS, archive=arch, scale_low=0.3
            )
    return out.tobytes(garbage=3, deflate=True)


def translate_tab(provider, model_name):
    st.write("ارفع ملف الـ PDF، وهتاخد PDF جديد فيه السلايد الأصلي + الترجمة تحته.")
    uploaded_file = st.file_uploader("اختر ملف PDF", type=["pdf"], key="tr_pdf")
    if uploaded_file is None or not st.button("ابدأ الترجمة 🚀"):
        return

    src_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pages = []
    for i in range(len(src_doc)):
        pg = src_doc.load_page(i)
        t = pg.get_text()
        if t.strip() or pg.get_images():
            png = pg.get_pixmap(dpi=80).tobytes("png")
            pages.append((i, t, png))

    if not pages:
        st.error("الملف فاضي.")
        return

    chunks = [pages[k:k + PAGES_PER_REQUEST]
              for k in range(0, len(pages), PAGES_PER_REQUEST)]

    results, failed = {}, []
    progress = st.progress(0.0)
    status = st.empty()
    done = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(translate_chunk, provider, model_name, c): c
                   for c in chunks}
        for fut in as_completed(futures):
            c = futures[fut]
            nums = f"{c[0][0] + 1}-{c[-1][0] + 1}"
            try:
                out, secs = fut.result()
                results.update(out)
                status.write(f"✅ سلايدات {nums} خلصت في {secs:.0f} ثانية")
            except Exception as e:
                failed.append(nums)
                st.error(f"سلايدات {nums}: {str(e)[:200]}")
            done += 1
            progress.progress(done / len(chunks))

    if not results:
        st.error("مفيش ولا صفحة اترجمت. جرّب نموذج تاني من القايمة.")
        return

    status.write("بجهّز ملف الـ PDF...")
    try:
        pdf_bytes = build_pdf(src_doc, results)
    except Exception as e:
        st.error(f"مشكلة في عمل الـ PDF: {str(e)[:300]}")
        return

    status.empty()
    if failed:
        st.warning(f"اترجمت {len(results)} من {len(pages)} صفحة. اللي فشل: {', '.join(failed)}")
    else:
        st.success(f"تمت ترجمة {len(results)} من {len(pages)} صفحة 🎉")
    st.download_button("📥 تحميل الـ PDF المترجم", pdf_bytes,
                       "Translated.pdf", "application/pdf")


# ---------------------------------------------------------------- الكويز
QUIZ_JSON_EXAMPLE = (
    '{"questions":[{"q":"question text","options":["option 1","option 2","option 3","option 4"],'
    '"answer":0,"explanation":"short explanation of the correct answer"}]}'
)


def make_quiz_prompt(n, level, qtype, lang_name):
    if qtype == "اختيار من متعدد":
        kind = "All questions are multiple choice with exactly 4 options each."
    elif qtype == "صح وخطأ":
        kind = ('All questions are true/false. The two options must be exactly '
                '["True","False"] (translated to the lecture language).')
    else:
        kind = ("A mix: about 70% multiple choice (4 options) and 30% true/false "
                "(two options: True and False, in the lecture language).")
    return f"""
You are a university professor specialized in chemistry, biology and sports science.
You will receive the text of a lecture. Create a quiz from it.

LANGUAGE (most important rule):
- Write the questions, options and explanations in the SAME language as the lecture text ({lang_name}).
- Do NOT translate the lecture to any other language. If the lecture is English, the whole quiz is English.
- Keep chemical formulas, symbols and units as in the lecture (use Unicode subscripts like H₂O, CO₂, k₀).

Rules:
1. Exactly {n} questions.
2. {kind}
3. Difficulty: {level}.
4. Exactly one correct answer per question. Wrong options must be plausible, not silly.
5. Questions must come from the lecture content only and cover its different parts, not just the first page. Focus on understanding, not word-for-word memorization.
6. Explanation: one or two sentences saying why the answer is correct.
7. No Markdown.
8. "answer" is the index of the correct option, starting from 0.
9. Output valid JSON only, no text before or after, no ``` fences, in this shape:
{QUIZ_JSON_EXAMPLE}
"""


def parse_quiz(raw):
    data = parse_json_block(raw)
    quiz = []
    for item in data.get("questions", []):
        q = str(item.get("q", "")).strip()
        opts = [str(o).strip() for o in item.get("options", [])]
        ans = item.get("answer")
        if not q or len(opts) < 2 or not isinstance(ans, int) or not (0 <= ans < len(opts)):
            continue
        correct = opts[ans]
        if len(opts) > 2:          # اختيار من متعدد: نخلط الترتيب. صح/خطأ: نسيبه زي ما هو
            random.shuffle(opts)
        quiz.append({
            "q": q,
            "options": opts,
            "answer": opts.index(correct),
            "explanation": str(item.get("explanation", "")).strip(),
        })
    if not quiz:
        raise ValueError("معرفتش أطلّع أسئلة من الرد")
    return quiz


def quiz_css(rtl):
    d = "rtl" if rtl else "ltr"
    a = "right" if rtl else "left"
    return f"""
<style>
[class*="st-key-quizq"] {{ direction: {d}; text-align: {a}; }}
.qbox {{ direction: {d}; text-align: {a}; font-size: 1.05rem;
        font-weight: 600; margin: 18px 0 4px 0; }}
.rbox {{ direction: {d}; text-align: {a}; padding: 10px 14px;
        border-radius: 10px; margin: 4px 0 10px 0; line-height: 1.7; }}
.ok  {{ background: rgba(46,160,67,.18); }}
.bad {{ background: rgba(218,54,51,.18); }}
.skip{{ background: rgba(150,150,150,.18); }}
</style>
"""


def quiz_tab(provider, model_name):
    st.write("ارفع المحاضرة (PDF أو صور) وهتطلعلك أسئلة بنفس لغة السلايد، تجاوب عليها وتشوف درجتك.")

    uploaded = st.file_uploader("اختر ملفات PDF أو صور للكويز", type=SRC_TYPES,
                                accept_multiple_files=True, key="quiz_pdf")
    c1, c2 = st.columns(2)
    n = c1.slider("عدد الأسئلة", 5, 20, 10)
    level = c2.selectbox("المستوى", ["سهل", "متوسط", "صعب"], index=1)
    level_en = {"سهل": "easy", "متوسط": "medium", "صعب": "hard"}[level]
    qtype = st.radio("نوع الأسئلة", ["اختيار من متعدد", "صح وخطأ", "مختلط"],
                     horizontal=True, key="qtype")

    if uploaded and st.button("ولّد الكويز 🧠"):
        try:
            text = read_sources(uploaded, provider, model_name, api_key)
        except Exception as e:
            st.error(str(e)[:250])
            return
        if not text.strip():
            st.error("معرفتش أقرأ أي نص من الملفات.")
            return
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
            st.info("الملف طويل، فالكويز مبني على أول جزء منه.")

        lang_name, rtl = detect_lang(text)
        with st.spinner("بجهّز الأسئلة..."):
            try:
                raw = call_model(provider, model_name, [("text", text)],
                                 system=make_quiz_prompt(n, level_en, qtype, lang_name))
                st.session_state.quiz = parse_quiz(raw)
                st.session_state.quiz_rtl = rtl
                st.session_state.submitted = False
                st.session_state.attempt = st.session_state.get("attempt", 0) + 1
            except Exception as e:
                st.error(f"معرفتش أجهّز الكويز: {str(e)[:250]}. جرّب تاني أو غيّر النموذج.")
                return

    quiz = st.session_state.get("quiz")
    if not quiz:
        return

    rtl = st.session_state.get("quiz_rtl", False)
    st.markdown(quiz_css(rtl), unsafe_allow_html=True)
    attempt = st.session_state.get("attempt", 1)
    submitted = st.session_state.get("submitted", False)
    st.divider()

    answers = []
    for i, item in enumerate(quiz):
        st.markdown(f'<div class="qbox">{i + 1}. {html.escape(item["q"])}</div>',
                    unsafe_allow_html=True)
        choice = st.radio(
            f"Question {i + 1}", item["options"], index=None,
            key=f"quizq_{attempt}_{i}", disabled=submitted,
            label_visibility="collapsed",
        )
        answers.append(choice)

        if submitted:
            correct = item["options"][item["answer"]]
            if choice is None:
                css, icon, head = "skip", "⏭️", "Not answered" if not rtl else "لم تجب"
            elif choice == correct:
                css, icon, head = "ok", "✅", "Correct" if not rtl else "إجابة صحيحة"
            else:
                label = "Your answer" if not rtl else "إجابتك"
                css, icon, head = "bad", "❌", f"{label}: {html.escape(choice)}"
            body = f"{icon} {head}"
            if choice != correct:
                label = "Correct answer" if not rtl else "الإجابة الصحيحة"
                body += f"<br>{label}: <b>{html.escape(correct)}</b>"
            if item["explanation"]:
                body += f"<br>{html.escape(item['explanation'])}"
            st.markdown(f'<div class="rbox {css}">{body}</div>', unsafe_allow_html=True)

    if not submitted:
        if st.button("صحّح الإجابات ✅"):
            st.session_state.submitted = True
            st.rerun()
    else:
        score = sum(1 for it, a in zip(quiz, answers)
                    if a is not None and it["options"].index(a) == it["answer"])
        pct = round(score / len(quiz) * 100)
        emoji = "🏆" if pct >= 85 else ("👍" if pct >= 60 else "📚")
        st.success(f"{emoji} درجتك: {score} من {len(quiz)} ({pct}%)")
        b1, b2 = st.columns(2)
        if b1.button("🔁 أعد المحاولة"):
            st.session_state.submitted = False
            st.session_state.attempt = attempt + 1
            st.rerun()
        if b2.button("🗑️ كويز جديد"):
            st.session_state.pop("quiz", None)
            st.session_state.submitted = False
            st.rerun()


# ---------------------------------------------------------------- حل ورقة من صورة
def solve_tab(provider, model_name):
    st.write("صوّر ورقة امتحان (أو ارفع PDF ممسوح)، والأداة هتقرأ الأسئلة وتحلها بسبب مختصر. "
             "مناسبة للأسئلة المرسومة (R/S، E/Z، Newman، Fischer...).")
    files = st.file_uploader("ورقة الامتحان (صور أو PDF)", type=SRC_TYPES,
                             accept_multiple_files=True, key="solve_files")
    expl = st.radio("لغة الشرح", ["العربية", "English"], horizontal=True, key="solve_lang")

    if files and st.button("اقرأ وحُل 📸"):
        pngs = []
        for f in files:
            pngs.extend(file_to_pngs(f, max_side=1800))
        if len(pngs) > 8:
            pngs = pngs[:8]
            st.info("الصور كتير، فهحل أول 8 صفحات بس.")

        items = []
        for j, p in enumerate(pngs):
            items.append(("text", f"===IMAGE {j + 1}==="))
            items.append(("image", p))
        prompt = SOLVE_PROMPT.format(expl_lang="Arabic" if expl == "العربية" else "English")
        with st.spinner("بقرأ الورقة وبحل..."):
            try:
                st.session_state.solved = (
                    call_model(provider, model_name, items, system=prompt),
                    expl == "العربية")
            except Exception as e:
                st.error(f"معرفتش أكمل: {str(e)[:250]}")
                return

    solved = st.session_state.get("solved")
    if solved:
        text, rtl = solved
        d = "rtl" if rtl else "ltr"
        a = "right" if rtl else "left"
        st.divider()
        st.warning("راجع الإجابات بنفسك، خصوصًا أسئلة الرسم (R/S وE/Z)، لأن النموذج ممكن يغلط في قراءة الرسومات.")
        for block in re.split(r'\n\s*\n|\n(?=Q\d+\))', text.strip()):
            if block.strip():
                st.markdown(f'<div style="direction:{d};text-align:{a};line-height:1.8;'
                            f'margin:6px 0">{esc(block.strip())}</div>', unsafe_allow_html=True)
        st.download_button("📥 تحميل الإجابات (txt)", text, "answers.txt", "text/plain")


# ---------------------------------------------------------------- محاكي المعادلات
SAFE_FUNCS = {
    "exp": np.exp, "ln": np.log, "log": np.log, "log10": np.log10, "log2": np.log2,
    "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "abs": np.abs,
}

SIM_PROMPT = """
You are a professor of chemistry, biology, sports science and physics who builds interactive teaching simulations.
You receive lecture material (text) and/or a topic or equation typed by the student.
Create @N@ simulation(s) of the most important quantitative relationship(s) (equations or formulas) in the material.
@FOCUS@

Each simulation is a JSON object with exactly these fields:
{
 "title": "short title",
 "description": "2-3 sentences: what the relationship means and what to try with the sliders",
 "formula": "the formula as plain text using Unicode, for example k = A·e^(−Ea/RT)",
 "variables": [{"name":"T","label":"Temperature","unit":"K","min":250,"max":500,"default":300,"step":1}],
 "x": "T",
 "x_label": "Temperature (K)",
 "y_label": "Rate constant k",
 "log_y": false,
 "constants": {"R": 8.314, "A": 1000000000},
 "curves": [{"label":"Without catalyst","expr":"A*exp(-Ea*1000/(R*T))"}],
 "readouts": [{"label":"k at the chosen T","expr":"A*exp(-Ea*1000/(R*T))"}],
 "observe": ["what the student should notice", "another observation"]
}

Rules for expressions ("expr"):
1. One Python-style math expression. Operators: + - * / ** (or ^) and parentheses.
   Functions allowed: exp, ln, log (natural log), log10, log2, sqrt, sin, cos, tan, sinh, cosh, tanh, abs.
   The constants pi and e are available.
2. Use ONLY names declared in "variables" or "constants". Names: letters, digits, underscore, starting with a letter, no spaces,
   and not equal to a function name or to "e" or "pi".
3. "x" must be the name of one variable. The curve is plotted against it while the other variables are controlled by sliders.
4. Use realistic ranges and units, and convert units inside the expression (for example kJ to J with *1000).
5. To compare cases (for example with and without a catalyst, trained vs untrained) use 2 or 3 curves with different labels
   and add extra variables for the differences.
6. At most 5 variables, 3 curves, 4 readouts.
7. Do not invent formulas. Use the one in the material or the standard textbook formula.
   If the material has no quantitative relationship, return {"simulations": []}.

LANGUAGE: write title, description, labels, "observe" and the words in "formula" in @LANG@. Keep symbols and units standard.
No Markdown. Output valid JSON only (no text before or after, no ``` fences) in this shape: {"simulations":[ ... ]}
"""


def safe_eval(expr, env):
    """تقييم آمن لمعادلة رياضية (بدون eval)"""
    expr = str(expr).strip().replace("^", "**").replace("×", "*").replace("·", "*").replace("−", "-")
    if not expr or len(expr) > 300:
        raise ValueError("bad expression")
    tree = ast.parse(expr, mode="eval")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return np.float64(node.value)
            raise ValueError("constant not allowed")
        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]
            raise ValueError(f"unknown name {node.id}")
        if isinstance(node, ast.UnaryOp):
            v = ev(node.operand)
            if isinstance(node.op, ast.USub):
                return -v
            if isinstance(node.op, ast.UAdd):
                return v
            raise ValueError("operator not allowed")
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.Pow):
                return np.power(a, b)
            raise ValueError("operator not allowed")
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCS
                    and not node.keywords and 1 <= len(node.args) <= 2):
                return SAFE_FUNCS[node.func.id](*[ev(x) for x in node.args])
            raise ValueError("function not allowed")
        raise ValueError("expression not allowed")

    with np.errstate(all="ignore"):
        return ev(tree)


def _f(v, default=None):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def clean_sim(spec):
    """يتحقق من المحاكي اللي النموذج طلّعه ويشيل أي حاجة بايظة"""
    if not isinstance(spec, dict):
        return None
    name_ok = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')

    variables, seen = [], set()
    for v in (spec.get("variables") or [])[:5]:
        if not isinstance(v, dict):
            continue
        name = str(v.get("name", "")).strip()
        if (not name_ok.fullmatch(name) or name in SAFE_FUNCS or name in ("e", "pi")
                or name in seen):
            continue
        lo, hi = _f(v.get("min")), _f(v.get("max"))
        if lo is None or hi is None or not lo < hi:
            continue
        de = _f(v.get("default"), (lo + hi) / 2)
        de = min(max(de, lo), hi)
        step = _f(v.get("step"), (hi - lo) / 100)
        if not step or step <= 0 or step > (hi - lo):
            step = (hi - lo) / 100
        seen.add(name)
        variables.append({
            "name": name, "label": str(v.get("label") or name),
            "unit": str(v.get("unit") or ""), "min": lo, "max": hi,
            "default": de, "step": step,
        })
    if not variables:
        return None

    constants = {}
    for k, v in (spec.get("constants") or {}).items():
        val = _f(v)
        if name_ok.fullmatch(str(k)) and val is not None and str(k) not in seen \
                and str(k) not in SAFE_FUNCS and str(k) not in ("e", "pi"):
            constants[str(k)] = np.float64(val)

    xname = str(spec.get("x", ""))
    if xname not in seen:
        xname = variables[0]["name"]

    test_env = {**constants, "pi": np.pi, "e": np.e}
    for v in variables:
        test_env[v["name"]] = np.float64(v["default"])
    xv = next(v for v in variables if v["name"] == xname)
    arr_env = {**test_env, xname: np.linspace(xv["min"], xv["max"], 20)}

    curves = []
    for c in (spec.get("curves") or [])[:3]:
        try:
            y = np.asarray(safe_eval(c.get("expr", ""), arr_env), dtype=float)
            if np.isfinite(y).any():
                curves.append({"label": str(c.get("label") or "y"), "expr": str(c["expr"])})
        except Exception:
            continue
    if not curves:
        return None

    readouts = []
    for r in (spec.get("readouts") or [])[:4]:
        try:
            safe_eval(r.get("expr", ""), test_env)
            readouts.append({"label": str(r.get("label") or ""), "expr": str(r["expr"])})
        except Exception:
            continue

    return {
        "title": str(spec.get("title") or "Simulation"),
        "description": str(spec.get("description") or ""),
        "formula": str(spec.get("formula") or ""),
        "variables": variables, "x": xname,
        "x_label": str(spec.get("x_label") or xv["label"]),
        "y_label": str(spec.get("y_label") or ""),
        "log_y": bool(spec.get("log_y")),
        "constants": constants, "curves": curves, "readouts": readouts,
        "observe": [str(o) for o in (spec.get("observe") or [])][:5],
    }


def show_chart(chart):
    for kwargs in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            st.altair_chart(chart, **kwargs)
            return
        except TypeError:
            continue


def render_sim(sim, idx, gen, rtl):
    d = "rtl" if rtl else "ltr"
    a = "right" if rtl else "left"
    if sim["description"]:
        st.markdown(f'<div style="direction:{d};text-align:{a};line-height:1.8">'
                    f'{esc(sim["description"])}</div>', unsafe_allow_html=True)
    if sim["formula"]:
        st.code(sim["formula"], language=None)

    vals = {}
    for v in sim["variables"]:
        label = f'{v["label"]} ({v["unit"]})' if v["unit"] else v["label"]
        vals[v["name"]] = st.slider(
            label, min_value=v["min"], max_value=v["max"], value=v["default"],
            step=v["step"], key=f"sim{gen}_{idx}_{v['name']}")

    xv = next(v for v in sim["variables"] if v["name"] == sim["x"])
    base = {**sim["constants"], "pi": np.pi, "e": np.e}
    base.update({k: np.float64(val) for k, val in vals.items()})
    xs = np.linspace(xv["min"], xv["max"], 300)

    rows, pts = [], []
    for c in sim["curves"]:
        try:
            ys = np.broadcast_to(np.asarray(safe_eval(c["expr"], {**base, sim["x"]: xs}),
                                            dtype=float), xs.shape)
            for x_, y_ in zip(xs, ys):
                if np.isfinite(y_) and (not sim["log_y"] or y_ > 0):
                    rows.append({"x": float(x_), "y": float(y_), "curve": c["label"]})
            yp = float(np.asarray(safe_eval(c["expr"], base)).reshape(-1)[0])
            if np.isfinite(yp) and (not sim["log_y"] or yp > 0):
                pts.append({"x": float(vals[sim["x"]]), "y": yp, "curve": c["label"]})
        except Exception:
            st.caption(f"⚠️ معرفتش أرسم: {c['label']}")

    if rows:
        df = pd.DataFrame(rows)
        if alt is not None:
            yscale = alt.Scale(type="log") if sim["log_y"] else alt.Scale(zero=False)
            line = alt.Chart(df).mark_line().encode(
                x=alt.X("x:Q", title=sim["x_label"]),
                y=alt.Y("y:Q", title=sim["y_label"], scale=yscale),
                color=alt.Color("curve:N", title=None,
                                legend=alt.Legend(orient="bottom")))
            layers = line
            if pts:
                layers = line + alt.Chart(pd.DataFrame(pts)).mark_point(
                    size=130, filled=True).encode(x="x:Q", y="y:Q", color="curve:N")
            show_chart(layers.properties(height=320))
        else:
            st.line_chart(df.pivot_table(index="x", columns="curve", values="y"))
    else:
        st.warning("المعادلة ما طلّعتش قيم صالحة بالقيم دي. جرّب تغيّر السلايدرات.")

    if sim["readouts"]:
        cols = st.columns(len(sim["readouts"]))
        for col, r in zip(cols, sim["readouts"]):
            try:
                val = float(np.asarray(safe_eval(r["expr"], base)).reshape(-1)[0])
                col.metric(r["label"], f"{val:.4g}")
            except Exception:
                col.metric(r["label"], "—")

    if sim["observe"]:
        items = "".join(f"<li>{esc(o)}</li>" for o in sim["observe"])
        st.markdown(f'<div style="direction:{d};text-align:{a};line-height:1.8">'
                    f'<b>{"🔎 لاحظ إن:" if rtl else "🔎 Notice:"}</b><ul>{items}</ul></div>',
                    unsafe_allow_html=True)


def sim_tab(provider, model_name):
    st.write("حوّل أي معادلة في المحاضرة لتجربة تلعب فيها بالسلايدرات وتشوف الرسم بيتغير قدامك. "
             "ارفع محاضرة (PDF أو صور) أو اكتب اسم المعادلة بنفسك.")
    topic = st.text_input("المعادلة أو الموضوع (اختياري لو رفعت ملف)",
                          placeholder="مثال: معادلة أرينيوس، أو Henderson-Hasselbalch، أو VO2max",
                          key="sim_topic")
    files = st.file_uploader("المحاضرة (PDF أو صور)", type=SRC_TYPES,
                             accept_multiple_files=True, key="sim_files")
    c1, c2 = st.columns(2)
    n = c1.slider("عدد المحاكيات", 1, 3, 1, key="sim_n")
    lang_opt = c2.selectbox("لغة الشرح", ["نفس لغة المصدر", "العربية", "English"], key="sim_lang")

    if (files or topic.strip()) and st.button("ابنِ المحاكي 🎛️"):
        src = ""
        if files:
            try:
                src = read_sources(files, provider, model_name, api_key)
            except Exception as e:
                st.error(str(e)[:250])
                return
        if len(src) > MAX_CHARS:
            src = src[:MAX_CHARS]
            st.info("الملف طويل، فالمحاكي مبني على أول جزء منه.")
        if topic.strip():
            src += f"\n\nSTUDENT REQUEST: {topic.strip()}"
        if not src.strip():
            st.error("معرفتش أقرأ أي محتوى.")
            return

        if lang_opt == "العربية":
            lang_name, rtl = "Arabic", True
        elif lang_opt == "English":
            lang_name, rtl = "English", False
        else:
            lang_name, rtl = detect_lang(src)
        focus = (f"Focus on this request from the student: {topic.strip()}" if topic.strip()
                 else "Pick the most important relationships that can be simulated.")
        prompt = (SIM_PROMPT.replace("@N@", str(n)).replace("@FOCUS@", focus)
                  .replace("@LANG@", lang_name))

        with st.spinner("بجهّز المحاكي..."):
            try:
                raw = call_model(provider, model_name, [("text", src)], system=prompt)
                data = parse_json_block(raw)
                sims = [s for s in (clean_sim(x) for x in data.get("simulations", [])) if s]
            except Exception as e:
                st.error(f"معرفتش أجهّز المحاكي: {str(e)[:250]}. جرّب تاني أو غيّر النموذج.")
                return
        if not sims:
            st.error("معرفتش أطلّع محاكي من المحتوى ده. جرّب تكتب اسم المعادلة بنفسك.")
            return
        st.session_state.sims = (sims, rtl, st.session_state.get("sim_gen", 0) + 1)
        st.session_state.sim_gen = st.session_state.sims[2]

    saved = st.session_state.get("sims")
    if saved:
        sims, rtl, gen = saved
        st.divider()
        st.warning("قارن المعادلة المكتوبة تحت العنوان بالمحاضرة، لأن النموذج ممكن يغلط في صياغتها أو في الأرقام.")
        for i, sim in enumerate(sims):
            with st.expander(f"🎛️ {sim['title']}", expanded=(i == 0)):
                render_sim(sim, i, gen, rtl)
        if st.button("🗑️ امسح المحاكيات"):
            st.session_state.pop("sims", None)
            st.rerun()


# ---------------------------------------------------------------- التشغيل
if api_key:
    if provider == "Gemini":
        genai.configure(api_key=api_key)

    if not ensure_font():
        st.error("معرفتش أنزّل الخط العربي. اتأكد إن التطبيق عنده إنترنت وجرّب تعمل Reboot.")
        st.stop()

    try:
        models = get_models(provider, api_key)
    except Exception as e:
        st.error(f"مشكلة في المفتاح أو الاتصال: {str(e)[:200]}")
        st.stop()

    if not models:
        st.error("مفيش نماذج متاحة لمفتاحك.")
        st.stop()

    model_name = st.selectbox("اختر النموذج:", models, index=0)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🌐 الترجمة", "🧠 كويز تفاعلي", "📸 حل ورقة من صورة", "🎛️ محاكي المعادلات"])
    with tab1:
        translate_tab(provider, model_name)
    with tab2:
        quiz_tab(provider, model_name)
    with tab3:
        solve_tab(provider, model_name)
    with tab4:
        sim_tab(provider, model_name)
