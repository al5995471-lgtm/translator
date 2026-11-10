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

provider = st.radio("اختر المزوّد:", ["Gemini", "Claude", "Groq"], horizontal=True)
KEY_NAMES = {"Gemini": "GEMINI_API_KEY", "Claude": "ANTHROPIC_API_KEY", "Groq": "GROQ_API_KEY"}
KEY_NAME = KEY_NAMES[provider]

try:
    api_key = st.secrets.get(KEY_NAME, "")
except Exception:
    api_key = ""
if not api_key:
    api_key = st.text_input(f"أدخل مفتاح {provider} API بتاعك:", type="password")

PAGES_PER_REQUEST = 2
WORKERS = 2
FONT_DIR = "fonts"
FONT_FILE = "Amiri-Regular.ttf"
FONT_URLS = [
    "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/amiri/Amiri-Regular.ttf",
]
CLAUDE_FALLBACK = ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"]
GROQ_URL = "https://api.groq.com/openai/v1"
SRC_TYPES = ["pdf", "png", "jpg", "jpeg"]


def text_budget(provider, normal=40000):
    """أقصى عدد حروف نبعته في الطلب (Groq المجاني حده ضيق)"""
    return 9000 if provider == "Groq" else normal


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
GROQ_BAD = ["whisper", "guard", "tts", "orpheus", "playai", "safeguard", "embed"]


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

    if provider == "Groq":
        r = requests.get(f"{GROQ_URL}/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"Groq: {r.status_code} {r.text[:150]}")
        names = [m["id"] for m in r.json().get("data", [])
                 if not any(b in m["id"].lower() for b in GROQ_BAD)]

        def rank(n):
            n = n.lower()
            if "gpt-oss-120b" in n:
                return 0
            if "llama-3.3-70b" in n:
                return 1
            if "qwen" in n:
                return 2
            return 3
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


def ask_groq(model_name, items, system, key):
    # Groq (المجاني) نصي فقط: الصور بتتجاهل
    text = "\n\n".join(d for k, d in items if k == "text")
    payload = {
        "model": model_name,
        "temperature": 0.2,
        "max_tokens": 3500,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": text}],
    }
    if "gpt-oss" in model_name.lower():
        payload["reasoning_effort"] = "low"
    r = requests.post(f"{GROQ_URL}/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json=payload, timeout=150)
    if r.status_code != 200:
        raise RuntimeError(f"{r.status_code} {r.text[:400]}")
    return r.json()["choices"][0]["message"].get("content") or ""


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
            if provider == "Groq":
                return ask_groq(model_name, items, system, key)
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
            if "413" in msg or "request too large" in low:
                raise RuntimeError("الملف كبير على حد المزوّد المجاني. قسّم الملف لأجزاء أصغر أو استخدم Gemini.")
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
    """يحدد لغة المحاضرة: عربي أو إنجليزي (أو غيره)"""
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


def get_reader(provider, model_name, key):
    """مزوّد بيشوف الصور: المختار لو Gemini/Claude، وإلا من Secrets"""
    if provider in ("Gemini", "Claude"):
        return provider, model_name, key
    for prov, kn in (("Gemini", "GEMINI_API_KEY"), ("Claude", "ANTHROPIC_API_KEY")):
        try:
            k = st.secrets.get(kn, "")
        except Exception:
            k = ""
        if k:
            try:
                ms = get_models(prov, k)
            except Exception:
                continue
            if ms:
                return prov, ms[0], k
    return None


NO_READER_MSG = ("قراءة الصور بتحتاج Gemini أو Claude. اختار Gemini من فوق، "
                 "أو ضيف GEMINI_API_KEY في Secrets.")


def ocr_images(pngs, reader):
    prov, model, key = reader
    out = []
    for s in range(0, len(pngs), 3):
        batch = pngs[s:s + 3]
        items = []
        for j, p in enumerate(batch):
            items.append(("text", f"===IMAGE {s + j + 1}==="))
            items.append(("image", p))
        out.append(call_model(prov, model, items, system=OCR_PROMPT, key=key))
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
        reader = get_reader(provider, model_name, key)
        if not reader:
            raise RuntimeError(NO_READER_MSG)
        with st.spinner(f"بقرأ {len(scans)} صورة/صفحة..."):
            texts.append(ocr_images(scans, reader))
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
    if provider == "Groq":
        st.info("Groq بيقرأ النص بس ومبيشوفش صور السلايدات، فشرح الرسومات هيبقى أضعف. "
                "الأفضل للترجمة Gemini.")
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
    workers = 1 if provider == "Groq" else WORKERS

    with ThreadPoolExecutor(max_workers=workers) as pool:
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
        limit = text_budget(provider)
        if len(text) > limit:
            text = text[:limit]
            st.info("الملف طويل على حد المزوّد، فالكويز مبني على أول جزء منه. "
                    "لو عايز الملف كله استخدم Gemini.")

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


# ---------------------------------------------------------------- متنبئ الامتحان
PRED_SCHEMA = (
    '{"style":"2-4 sentences describing the instructor exam style",'
    '"topics":[{"name":"topic","likelihood":"high","reason":"one short sentence"}],'
    '"tips":["short study tip"],'
    '"exam":[{"type":"mcq","q":"question","options":["a","b","c","d"],'
    '"answer":"the correct option text","explanation":"one short sentence"}]}'
)


def make_predict_prompt(n, lang_name):
    return f"""
You are an experienced university professor in chemistry, biology and sports science.
You will receive two parts of text:
A) PAST EXAMS of the same instructor (may be empty).
B) LECTURE material for the coming exam.

Your tasks:
1. Analyze the instructor's style from the past exams: question types, difficulty, wording style,
   and which topics appear most. If no past exams are given, say that in "style" and use a general university exam style.
2. List 5 to 8 topics from the lectures that are most likely to appear, with likelihood "high", "medium" or "low"
   and a one-sentence reason (for example: asked before, heavily emphasized, or not covered by past exams yet).
   This is only an educated estimate, never a guarantee.
3. Give 3 to 5 short study tips based on the analysis.
4. Write a mock exam of exactly {n} questions imitating the instructor's style and mixing the same question types.
   Question "type" is one of: "mcq" (exactly 4 options, "answer" = the exact text of the correct option),
   "tf" (options exactly ["True","False"], "answer" = "True" or "False"),
   "short" or "problem" (options = [], "answer" = a model answer; for problems show the steps).

LANGUAGE (important): write everything in the SAME language as the lecture text ({lang_name}).
Do NOT translate to another language. Keep formulas, symbols and units as in the lecture (Unicode subscripts like H₂O).
No Markdown. Output valid JSON only, no text before or after, no ``` fences, in this shape:
{PRED_SCHEMA}
"""


def parse_prediction(raw):
    data = parse_json_block(raw)
    exam = []
    for q in data.get("exam", []):
        if not str(q.get("q", "")).strip():
            continue
        exam.append({
            "type": str(q.get("type", "short")).lower(),
            "q": str(q.get("q", "")).strip(),
            "options": [str(o) for o in q.get("options", [])],
            "answer": str(q.get("answer", "")).strip(),
            "explanation": str(q.get("explanation", "")).strip(),
        })
    if not exam and not data.get("topics"):
        raise ValueError("الرد فاضي")
    data["exam"] = exam
    return data


def render_prediction(data, rtl):
    d = "rtl" if rtl else "ltr"
    a = "right" if rtl else "left"
    st.markdown(f"""<style>
.pbox {{ direction:{d}; text-align:{a}; line-height:1.8; margin:4px 0 8px 0; }}
.pq {{ direction:{d}; text-align:{a}; font-weight:600; margin-top:16px; }}
</style>""", unsafe_allow_html=True)
    show = "إظهار الإجابة" if rtl else "Show answer"
    icons = {"high": "🔴", "medium": "🟠", "low": "🟢"}

    if data.get("style"):
        st.subheader("🧭 أسلوب الدكتور")
        st.markdown(f'<div class="pbox">{esc(data["style"])}</div>', unsafe_allow_html=True)

    topics = data.get("topics", [])
    if topics:
        st.subheader("🎯 المواضيع المتوقعة")
        for t in topics:
            icon = icons.get(str(t.get("likelihood", "")).lower(), "⚪")
            st.markdown(
                f'<div class="pbox">{icon} <b>{esc(t.get("name", ""))}</b>'
                f' — {esc(t.get("reason", ""))}</div>', unsafe_allow_html=True)
        st.caption("🔴 احتمال عالي   🟠 متوسط   🟢 أقل. ده تقدير مش ضمان.")

    tips = data.get("tips", [])
    if tips:
        st.subheader("💡 نصايح للمذاكرة")
        for tip in tips:
            st.markdown(f'<div class="pbox">• {esc(tip)}</div>', unsafe_allow_html=True)

    exam = data.get("exam", [])
    if exam:
        st.subheader("📝 امتحان تجريبي بأسلوب الدكتور")
        for i, q in enumerate(exam):
            st.markdown(f'<div class="pq">{i + 1}. {esc(q["q"])}</div>', unsafe_allow_html=True)
            if q["options"]:
                opts = "<br>".join(f"{chr(65 + k)}) {esc(o)}" for k, o in enumerate(q["options"]))
                st.markdown(f'<div class="pbox">{opts}</div>', unsafe_allow_html=True)
            with st.expander(show):
                body = esc(q["answer"])
                if q["explanation"]:
                    body += "<br><br>" + esc(q["explanation"])
                st.markdown(f'<div class="pbox">{body}</div>', unsafe_allow_html=True)


def predict_tab(provider, model_name):
    st.write("ارفع امتحانات سنين فاتت + المحاضرات (PDF أو صور)، وهتحلل أسلوب الدكتور وتطلعلك "
             "المواضيع المتوقعة وامتحان تجريبي بنفس أسلوبه.")
    exams = st.file_uploader("امتحانات سابقة (PDF أو صور) — اختياري لكن بيحسّن التنبؤ",
                             type=SRC_TYPES, accept_multiple_files=True, key="pred_exams")
    lects = st.file_uploader("المحاضرات (PDF أو صور)", type=SRC_TYPES,
                             accept_multiple_files=True, key="pred_lects")
    n = st.slider("عدد أسئلة الامتحان التجريبي", 5, 20, 10, key="pred_n")

    if lects and st.button("حلّل وتنبّأ 🔮"):
        try:
            lect_text = read_sources(lects, provider, model_name, api_key)
            exam_text = read_sources(exams, provider, model_name, api_key) if exams else ""
        except Exception as e:
            st.error(str(e)[:250])
            return
        if not lect_text.strip():
            st.error("معرفتش أقرأ أي نص من المحاضرات.")
            return

        budget = text_budget(provider, 60000)
        exam_budget = int(budget * 0.4) if exam_text else 0
        lect_budget = budget - exam_budget
        if len(exam_text) > exam_budget and exam_text:
            exam_text = exam_text[:exam_budget]
            st.info("الامتحانات طويلة، فاتحلل أول جزء منها.")
        if len(lect_text) > lect_budget:
            lect_text = lect_text[:lect_budget]
            st.info("المحاضرات طويلة على حد المزوّد، فاتحلل أول جزء منها. "
                    "لو عايزها كلها استخدم Gemini.")

        lang_name, rtl = detect_lang(lect_text)
        payload = (f"=== PAST EXAMS ===\n{exam_text or '(none provided)'}\n\n"
                   f"=== LECTURES ===\n{lect_text}")
        with st.spinner("بحلل الامتحانات والمحاضرات..."):
            try:
                raw = call_model(provider, model_name, [("text", payload)],
                                 system=make_predict_prompt(n, lang_name))
                st.session_state.pred = (parse_prediction(raw), rtl)
            except Exception as e:
                st.error(f"معرفتش أكمل: {str(e)[:250]}. جرّب تاني أو غيّر النموذج.")
                return

    pred = st.session_state.get("pred")
    if pred:
        st.divider()
        render_prediction(pred[0], pred[1])
        if st.button("🗑️ امسح النتيجة"):
            st.session_state.pop("pred", None)
            st.rerun()


# ---------------------------------------------------------------- حل ورقة من صورة
def solve_tab(provider, model_name):
    st.write("صوّر ورقة امتحان (أو ارفع PDF ممسوح)، والأداة هتقرأ الأسئلة وتحلها بسبب مختصر. "
             "مناسبة للأسئلة المرسومة (R/S، E/Z، Newman، Fischer...).")
    files = st.file_uploader("ورقة الامتحان (صور أو PDF)", type=SRC_TYPES,
                             accept_multiple_files=True, key="solve_files")
    expl = st.radio("لغة الشرح", ["العربية", "English"], horizontal=True, key="solve_lang")

    if files and st.button("اقرأ وحُل 📸"):
        reader = get_reader(provider, model_name, api_key)
        if not reader:
            st.error(NO_READER_MSG)
            return
        pngs = []
        for f in files:
            pngs.extend(file_to_pngs(f, max_side=1800))
        if len(pngs) > 8:
            pngs = pngs[:8]
            st.info("الصور كتير، فهحل أول 8 صفحات بس.")

        prov, model, key = reader
        items = []
        for j, p in enumerate(pngs):
            items.append(("text", f"===IMAGE {j + 1}==="))
            items.append(("image", p))
        prompt = SOLVE_PROMPT.format(expl_lang="Arabic" if expl == "العربية" else "English")
        with st.spinner("بقرأ الورقة وبحل..."):
            try:
                st.session_state.solved = (call_model(prov, model, items, system=prompt, key=key),
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
        ["🌐 الترجمة", "🧠 كويز تفاعلي", "🔮 متنبئ الامتحان", "📸 حل ورقة من صورة"])
    with tab1:
        translate_tab(provider, model_name)
    with tab2:
        quiz_tab(provider, model_name)
    with tab3:
        predict_tab(provider, model_name)
    with tab4:
        solve_tab(provider, model_name)
