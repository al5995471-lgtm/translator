import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from docx import Document
from docx.shared import Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import re
import io

st.set_page_config(page_title="المترجم العلمي للمحاضرات", page_icon="🔬")
st.title("🔬 مترجم المحاضرات العلمي (كيمياء، بايو، رياضة)")
st.write("ارفع ملف الـ PDF، والأداة هتقرأ كل سلايد وتترجمها ترجمة علمية سياقية.")

api_key = st.text_input("أدخل مفتاح Gemini API بتاعك:", type="password")


def clean_markdown(text):
    text = re.sub(r'\*+(.*?)\*+', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    return text.strip()


def make_rtl(p, size=13):
    """يخلي الفقرة عربي (يمين لليسار) بشكل حقيقي في Word"""
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


if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.8-flash')

    uploaded_file = st.file_uploader("اختر ملف PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("ابدأ الترجمة 🚀"):
            with st.spinner("جاري الترجمة وتنظيم الملف..."):
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                out_doc = Document()

                h = out_doc.add_heading('ترجمة المحاضرة', 0)
                make_rtl(h, 20)

                system_prompt = """
أنت مترجم أكاديمي متخصص في الكيمياء والأحياء وعلوم الرياضة.
قواعد الترجمة:
1. ترجم بالمعنى إلى عربية علمية سليمة وسلسة، وليس ترجمة حرفية.
2. اكتب المصطلح العلمي بالعربي وبعده بين قوسين المصطلح الإنجليزي، مرة واحدة فقط عند أول ظهور في الصفحة.
3. اترك المعادلات والرموز والوحدات بالإنجليزية، واكتب الأرقام السفلية بشكل Unicode مثل H₂O و CO₂ و O₂.
4. النص مستخرج من PDF، فالجمل ممكن تكون مقطوعة؛ ركّب الجملة بشكل صحيح.
5. لا تستخدم أي رموز Markdown مثل ** أو ###. اكتب كل بند في سطر مستقل.
6. اكتب الترجمة فقط بدون أي شرح أو مقدمة.
"""

                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        try:
                            response = model.generate_content(
                                f"{system_prompt}\n\nالنص:\n{text}",
                                generation_config={"temperature": 0.2},
                            )
                            cleaned_text = clean_markdown(response.text)

                            head = out_doc.add_heading(f'سلايد رقم {page_num + 1}', level=1)
                            make_rtl(head, 16)

                            for line in cleaned_text.split('\n'):
                                if line.strip():
                                    p = out_doc.add_paragraph(line.strip())
                                    make_rtl(p, 13)

                            out_doc.add_page_break()
                        except Exception as e:
                            st.error(f"مشكلة في سلايد {page_num + 1}: {e}")

                bio = io.BytesIO()
                out_doc.save(bio)
                st.success("تمت الترجمة بنجاح! 🎉")
                st.download_button(
                    "📥 تحميل ملف الترجمة",
                    bio.getvalue(),
                    "Translated_Clean.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
