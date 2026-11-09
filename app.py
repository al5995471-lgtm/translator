import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import io
import time

st.set_page_config(page_title="المترجم العلمي للمحاضرات", page_icon="🔬")
st.title("🔬 مترجم المحاضرات العلمي (كيمياء، بايو، رياضة)")
st.write("ارفع ملف الـ PDF، والأداة هتقرأ كل سلايد وتترجمها ترجمة علمية سياقية.")

api_key = st.text_input("أدخل مفتاح Gemini API بتاعك:", type="password")

PAGES_PER_REQUEST = 3   # عدد الصفحات في كل طلب (يقلل الطلبات)
WORKERS = 2             # طلبات متوازية

SYSTEM_PROMPT = """
أنت مترجم أكاديمي متخصص في الكيمياء والأحياء وعلوم الرياضة.
قواعد الترجمة:
1. ترجم بالمعنى إلى عربية علمية سليمة وسلسة، وليس ترجمة حرفية.
2. اكتب المصطلح العلمي بالعربي وبعده بين قوسين المصطلح الإنجليزي، مرة واحدة فقط عند أول ظهور في الصفحة.
3. اترك المعادلات والرموز والوحدات بالإنجليزية، واكتب الأرقام السفلية بشكل Unicode مثل H₂O و CO₂ و O₂.
4. اكتب كل معادلة كيميائية أو رياضية في سطر مستقل بالكامل، بدون أي كلمات عربية في نفس السطر.
5. النص مستخرج من PDF، فالجمل ممكن تكون مقطوعة؛ ركّب الجملة بشكل صحيح.
6. لا تستخدم أي رموز Markdown مثل ** أو ###. اكتب كل بند في سطر مستقل.
7. اكتب الترجمة فقط بدون أي شرح أو مقدمة.
8. النص فيه عدة صفحات، كل صفحة تبدأ بسطر على شكل ===PAGE رقم=== . اكتب نفس هذا السطر بالضبط قبل ترجمة كل صفحة.
"""

ARABIC = re.compile(r'[\u0600-\u06FF]')
EQUATION = re.compile(r'[→⇌=+]')
BAD_WORDS = ["robotics", "tts", "image", "live", "embedding", "audio", "computer",
             "aqa", "imagen", "veo", "lyria", "deep-research", "learnlm", "gemma",
             "vision", "thinking-exp"]


def clean_markdown(text):
    text = re.sub(r'\*+(.*?)\*+', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    return text.strip()


def make_rtl(p, size=13):
    pPr = p._p.get_or_add_pPr()
    pPr.append(OxmlElement('w:bidi'))
    for run in p.runs:
        run.font.name = 'Arial'
        run.font.size = Pt(size)
        run.font.rtl = True
        rPr = run._r.get_or_add_rPr()
        rPr.rFonts.set(qn('w:cs'), 'Arial')
        szCs = OxmlElement('w:szCs')
        szCs.set(qn('w:val'), str(int(size * 2)))
        sz = rPr.find(qn('w:sz'))
        if sz is not None:
            sz.addnext(szCs)


def add_line(out_doc, line):
    line = line.strip()
    if not line:
        return
    p = out_doc.add_paragraph(line)
    if not ARABIC.search(line) and EQUATION.search(line):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = 'Arial'
            run.font.size = Pt(14)
    else:
        make_rtl(p, 13)


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


def call_model(model_name, prompt):
    model = genai.GenerativeModel(model_name)
    for attempt in range(5):
        try:
            r = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2},
                request_options={"timeout": 120, "retry": None},
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
    body = "\n".join(f"===PAGE {i + 1}===\n{t}" for i, t in chunk)
    raw = call_model(model_name, f"{SYSTEM_PROMPT}\n\nالنص:\n{body}")
    raw = clean_markdown(raw)

    parts = re.split(r'===\s*PAGE\s*(\d+)\s*===', raw)
    out = {}
    # parts = [قبل أول فاصل, رقم, نص, رقم, نص, ...]
    for k in range(1, len(parts) - 1, 2):
        out[int(parts[k]) - 1] = parts[k + 1].strip()
    if not out:  # لو الفواصل ضاعت
        out[chunk[0][0]] = raw
    return out, time.time() - start


if api_key:
    genai.configure(api_key=api_key)

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
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            pages = []
            for i in range(len(doc)):
                t = doc.load_page(i).get_text()
                if t.strip():
                    pages.append((i, t))

            if not pages:
                st.error("الملف مفيهوش نص (ممكن يكون صور ممسوحة).")
                st.stop()

            chunks = [pages[k:k + PAGES_PER_REQUEST]
                      for k in range(0, len(pages), PAGES_PER_REQUEST)]

            results = {}
            failed = []
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

            out_doc = Document()
            h = out_doc.add_heading('ترجمة المحاضرة', 0)
            make_rtl(h, 20)

            for num in sorted(results):
                head = out_doc.add_heading(f'سلايد رقم {num + 1}', level=1)
                make_rtl(head, 16)
                for line in results[num].split('\n'):
                    add_line(out_doc, line)
                out_doc.add_page_break()

            bio = io.BytesIO()
            out_doc.save(bio)
            if failed:
                st.warning(f"اترجمت {len(results)} من {len(pages)} صفحة. اللي فشل: {', '.join(failed)}")
            else:
                st.success(f"تمت ترجمة {len(results)} من {len(pages)} صفحة 🎉")
            st.download_button(
                "📥 تحميل ملف الترجمة",
                bio.getvalue(),
                "Translated_Clean.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
