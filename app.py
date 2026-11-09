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
"""

ARABIC = re.compile(r'[\u0600-\u06FF]')
EQUATION = re.compile(r'[→⇌=+]')


def clean_markdown(text):
    text = re.sub(r'\*+(.*?)\*+', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    return text.strip()


def make_rtl(p, size=13):
    """فقرة عربي (يمين لليسار) حقيقية في Word"""
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
    # سطر معادلة (من غير عربي): شمال لليمين وفي النص
    if not ARABIC.search(line) and EQUATION.search(line):
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = 'Arial'
            run.font.size = Pt(14)
    else:
        make_rtl(p, 13)


def translate_page(model, page_num, text):
    start = time.time()
    response = model.generate_content(
        f"{SYSTEM_PROMPT}\n\nالنص:\n{text}",
        generation_config={"temperature": 0.2},
        request_options={"timeout": 90, "retry": None},
    )
    return page_num, clean_markdown(response.text), time.time() - start


if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.8-flash')

    uploaded_file = st.file_uploader("اختر ملف PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("ابدأ الترجمة 🚀"):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            pages = []
            for i in range(len(doc)):
                t = doc.load_page(i).get_text()
                if t.strip():
                    pages.append((i, t))

            results = {}
            progress = st.progress(0.0)
            status = st.empty()
            done = 0

            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(translate_page, model, i, t): i for i, t in pages}
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        num, text, secs = fut.result()
                        results[num] = text
                        status.write(f"✅ سلايد {num + 1} خلص في {secs:.0f} ثانية")
                    except Exception as e:
                        results[i] = None
                        st.error(f"مشكلة في سلايد {i + 1}: {e}")
                    done += 1
                    progress.progress(done / len(pages))

            out_doc = Document()
            h = out_doc.add_heading('ترجمة المحاضرة', 0)
            make_rtl(h, 20)

            for num in sorted(results):
                if results[num] is None:
                    continue
                head = out_doc.add_heading(f'سلايد رقم {num + 1}', level=1)
                make_rtl(head, 16)
                for line in results[num].split('\n'):
                    add_line(out_doc, line)
                out_doc.add_page_break()

            bio = io.BytesIO()
            out_doc.save(bio)
            st.success("تمت الترجمة بنجاح! 🎉")
            st.download_button(
                "📥 تحميل ملف الترجمة",
                bio.getvalue(),
                "Translated_Clean.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
