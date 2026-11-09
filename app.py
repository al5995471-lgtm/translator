import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
import re
import io

st.set_page_config(page_title="المترجم العلمي للمحاضرات", page_icon="🔬")
st.title("🔬 مترجم المحاضرات العلمي (كيمياء، بايو، رياضة)")
st.write("ارفع ملف الـ PDF، والأداة هتقرأ كل سلايد وتترجمها ترجمة علمية سياقية.")

api_key = st.text_input("أدخل مفتاح Gemini API بتاعك:", type="password")

def clean_markdown(text):
    # مسح رموز التنسيق markdown كي تظهر نصوص ناعمة في Word
    text = re.sub(r'\*+(.*?)\*+', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    return text.strip()

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
                h.alignment = WD_ALIGN_PARAGRAPH.RIGHT

                system_prompt = """
                أنت مترجم علمي خبير متناول لمجالات: الكيمياء، الأحياء، والرياضيات.
                قواعد الترجمة:
                1. ترجم النص لغة عربية سياقية مفهومة، مع الحفاظ على المصطلحات بالإنجليزية بين قوسين بدون رموز نجمية.
                2. اترك المعادلات الرياضية والكيميائية كما هي بالإنجليزية.
                3. لا تستخدم رموز Markdown إطلاقاً مثل ** أو ### في إجابتك.
                """

                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        try:
                            response = model.generate_content(f"{system_prompt}\n\nالنص:\n{text}")
                            cleaned_text = clean_markdown(response.text)
                            
                            head = out_doc.add_heading(f'سلايد رقم {page_num + 1}', level=1)
                            head.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                            
                            for line in cleaned_text.split('\n'):
                                if line.strip():
                                    p = out_doc.add_paragraph(line.strip())
                                    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                            
                            out_doc.add_page_break()
                        except Exception as e:
                            st.error(f"مشكلة في سلايد {page_num + 1}: {e}")

                bio = io.BytesIO()
                out_doc.save(bio)
                st.success("تمت الترجمة بنجاح! 🎉")
                st.download_button("📥 تحميل ملف الترجمة", bio.getvalue(), "Translated_Clean.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
