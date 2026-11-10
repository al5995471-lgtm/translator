import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
import anthropic
import base64
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import os
import re
import time
import requests

st.set_page_config(page_title="المترجم العلمي للمحاضرات", page_icon="🔬")
st.title("🔬 أدوات المحاضرات العلمية")

provider = st.radio("اختر المزوّد:", ["Claude", "Gemini"], horizontal=True)
KEY_NAME = "ANTHROPIC_API_KEY" if provider == "Claude" else "GEMINI_API_KEY"

try:
    api_key = st.secrets.get(KEY_NAME, "")
except Exception:
    api_key = ""
if not api_key:
    api_key = st.text_input(f"أدخل مفتاح {provider} API بتاعك:", type="password")

PAGES_PER_REQUEST = 2
WORKERS = 2
MAX_QUIZ_CHARS = 40000
FONT_DIR = "fonts"
FONT_FILE = "Amiri-Regular.ttf"
FONT_URLS = [
    "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/amiri/Amiri-Regular.ttf",
]
CLAUDE_FALLBACK = ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"]

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


def ask_gemini(model_name, items, system):
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


def ask_claude(model_name, items, system):
    client = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=150)
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


def call_model(provider, model_name, items, system=SYSTEM_PROMPT):
    for attempt in range(5):
        try:
            if provider == "Claude":
                return ask_claude(model_name, items, system)
            return ask_gemini(model_name, items, system)
        except Exception as e:
            msg = str(e)
            if "limit: 0" in msg:
                raise RuntimeError("النموذج ده مش متاح في الباقة المجانية، اختار نموذج تاني من القايمة.")
            if "credit balance" in msg.lower():
                raise RuntimeError("رصيد Claude خلص، اشحن الحساب من console.anthropic.com.")
            if "401" in msg or "authentication" in msg.lower():
                raise RuntimeError("المفتاح غلط أو منتهي.")
            if "429" in msg or "rate_limit" in msg.lower() or "overloaded" in msg.lower():
                m = re.search(r'retry in (\d+(?:\.\d+)?)s', msg)
                delay = float(m.group(1)) if m else 30
                time.sleep(min(delay + 2, 75))
                continue
            if attempt == 4:
                raise RuntimeError(msg[:200])
            time.sleep(3)
    raise RuntimeError("وصلت لحد الاستخدام، استنى دقيقة وجرّب تاني.")


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


def detect_lang(text):
    """يحدد لغة المحاضرة: عربي أو إنجليزي (أو غيره)"""
    sample = text[:6000]
    ar = len(re.findall(r'[\u0600-\u06FF]', sample))
    lat = len(re.findall(r'[A-Za-z]', sample))
    if ar > lat:
        return "Arabic", True
    return "the same language as the text (for example English)", False


def parse_quiz(raw):
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?', '', raw).strip()
    raw = re.sub(r'```$', '', raw).strip()
    a, b = raw.find('{'), raw.rfind('}')
    if a == -1 or b == -1:
        raise ValueError("الرد مش JSON")
    data = json.loads(raw[a:b + 1])
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
    st.write("ارفع المحاضرة وهتطلعلك أسئلة بنفس لغة السلايد، تجاوب عليها وتشوف درجتك.")

    uploaded = st.file_uploader("اختر ملف PDF للكويز", type=["pdf"], key="quiz_pdf")
    c1, c2 = st.columns(2)
    n = c1.slider("عدد الأسئلة", 5, 20, 10)
    level = c2.selectbox("المستوى", ["سهل", "متوسط", "صعب"], index=1)
    level_en = {"سهل": "easy", "متوسط": "medium", "صعب": "hard"}[level]
    qtype = st.radio("نوع الأسئلة", ["اختيار من متعدد", "صح وخطأ", "مختلط"],
                     horizontal=True, key="qtype")

    if uploaded is not None and st.button("ولّد الكويز 🧠"):
        doc = fitz.open(stream=uploaded.read(), filetype="pdf")
        parts = []
        for i in range(len(doc)):
            t = doc.load_page(i).get_text().strip()
            if t:
                parts.append(f"--- Slide {i + 1} ---\n{t}")
        text = "\n\n".join(parts)
        if not text.strip():
            st.error("الملف مفيهوش نص (ممكن يكون صور ممسوحة).")
            return
        if len(text) > MAX_QUIZ_CHARS:
            text = text[:MAX_QUIZ_CHARS]
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
                st.error(f"معرفتش أجهّز الكويز: {str(e)[:200]}. جرّب تاني أو غيّر النموذج.")
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

    tab1, tab2 = st.tabs(["🌐 الترجمة", "🧠 كويز تفاعلي"])
    with tab1:
        translate_tab(provider, model_name)
    with tab2:
        quiz_tab(provider, model_name)
