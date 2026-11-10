import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import html
import os
import re
import time
import requests

st.set_page_config(page_title="المترجم العلمي للمحاضرات", page_icon="🔬")
st.title("🔬 مترجم المحاضرات العلمي (كيمياء، بايو، رياضة)")
st.write("ارفع ملف الـ PDF، وهتاخد PDF جديد فيه السلايد الأصلي + الترجمة تحته.")

# المفتاح من Secrets، ولو مش موجود يطلبه يدويًا
try:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    api_key = ""
if not api_key:
    api_key = st.text_input("أدخل مفتاح Gemini API بتاعك:", type="password")

PAGES_PER_REQUEST = 2
WORKERS = 2
FONT_DIR = "fonts"
FONT_FILE = "Amiri-Regular.ttf"
FONT_URLS = [
    "https://github.com/google/fonts/raw/main/ofl/amiri/Amiri-Regular.ttf",
    "https://raw.githubusercontent.com/google/fonts/main/ofl/amiri/Amiri-Regular.ttf",
]

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
def get_models(key):
    genai.configure(api_key=key)
    names = []
    for m in genai.list_models():
        n = m.name.replace("models/", "")
        if "generateContent" not in m.supported_generation_methods:
            continue
        if not n.startswith("gemini"):
            continue
        if any(b in n for b in BAD_WORDS):
            continue
        names.append(n)

    def rank(n):
        if "flash" in n and "lite" not in n:
            return 0
        if "flash" in n:
            return 1
        return 2

    return sorted(names, key=lambda n: (rank(n), n))


def call_model(model_name, content):
    model = genai.GenerativeModel(model_name)
    for attempt in range(5):
        try:
            r = model.generate_content(
                content,
                generation_config={"temperature": 0.2},
                request_options={"timeout": 150, "retry": None},
            )
            return r.text
        except Exception as e:
            msg = str(e)
            if "limit: 0" in msg:
                raise RuntimeError("النموذج ده مش متاح في الباقة المجانية، اختار نموذج تاني من القايمة.")
            if "429" in msg:
                m = re.search(r'retry in (\d+(?:\.\d+)?)s', msg)
                delay = float(m.group(1)) if m else 30
                time.sleep(min(delay + 2, 75))
                continue
            if attempt == 4:
                raise RuntimeError(msg[:200])
            time.sleep(3)
    raise RuntimeError("وصلت لحد الاستخدام، استنى دقيقة وجرّب تاني.")


def translate_chunk(model_name, chunk):
    start = time.time()
    content = [SYSTEM_PROMPT]
    for i, text, png in chunk:
        content.append(f"===PAGE {i + 1}===")
        content.append({"mime_type": "image/png", "data": png})
        content.append(f"النص المستخرج:\n{text}")
    raw = clean_markdown(call_model(model_name, content))

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


if api_key:
    genai.configure(api_key=api_key)

    if not ensure_font():
        st.error("معرفتش أنزّل الخط العربي. اتأكد إن التطبيق عنده إنترنت وجرّب تعمل Reboot.")
        st.stop()

    try:
        models = get_models(api_key)
    except Exception as e:
        st.error(f"مشكلة في المفتاح أو الاتصال: {str(e)[:200]}")
        st.stop()

    if not models:
        st.error("مفيش نماذج Gemini متاحة لمفتاحك.")
        st.stop()

    model_name = st.selectbox("اختر النموذج:", models, index=0)
    uploaded_file = st.file_uploader("اختر ملف PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("ابدأ الترجمة 🚀"):
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
                st.stop()

            chunks = [pages[k:k + PAGES_PER_REQUEST]
                      for k in range(0, len(pages), PAGES_PER_REQUEST)]

            results, failed = {}, []
            progress = st.progress(0.0)
            status = st.empty()
            done = 0

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = {pool.submit(translate_chunk, model_name, c): c for c in chunks}
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
                st.stop()

            status.write("بجهّز ملف الـ PDF...")
            try:
                pdf_bytes = build_pdf(src_doc, results)
            except Exception as e:
                st.error(f"مشكلة في عمل الـ PDF: {str(e)[:300]}")
                st.stop()

            status.empty()
            if failed:
                st.warning(f"اترجمت {len(results)} من {len(pages)} صفحة. اللي فشل: {', '.join(failed)}")
            else:
                st.success(f"تمت ترجمة {len(results)} من {len(pages)} صفحة 🎉")
            st.download_button(
                "📥 تحميل الـ PDF المترجم",
                pdf_bytes,
                "Translated.pdf",
                "application/pdf",
)
