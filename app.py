import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from docx import Document
import io

st.set_page_config(page_title="المترجم العلمي للمحاضرات", page_icon="🔬")
st.title("🔬 مترجم المحاضرات العلمي (كيمياء، بايو، رياضة)")
st.write("ارفع ملف الـ PDF، والأداة هتقرأ كل سلايد وتترجمها ترجمة علمية سياقية.")

api_key = st.text_input("أدخل مفتاح Gemini API بتاعك:", type="password")

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    uploaded_file = st.file_uploader("اختر ملف PDF", type=["pdf"])

    if uploaded_file is not None:
        if st.button("ابدأ الترجمة 🚀"):
            with st.spinner("جاري الترجمة..."):
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                out_doc = Document()
                out_doc.add_heading('ترجمة المحاضرة', 0)

                system_prompt = """
                أنت مترجم علمي خبير. قم بترجمة النص التالي إلى اللغة العربية.
                قواعد الترجمة:
                1. الترجمة سياقية تفهم المعنى العلمي وليس ترجمة حرفية.
                2. حافظ على المصطلحات العلمية بالإنجليزية بين قوسين.
                3. لا تترجم المعادلات الكيميائية أو الرياضية.
                """

                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        try:
                            response = model.generate_content(f"{system_prompt}\n\nالنص:\n{text}")
                            out_doc.add_heading(f'سلايد رقم {page_num + 1}', level=1)
                            out_doc.add_paragraph(response.text)
                            out_doc.add_page_break()
                        except Exception as e:
                            st.error(f"مشكلة في سلايد {page_num + 1}: {e}")

                bio = io.BytesIO()
                out_doc.save(bio)
                st.success("تمت الترجمة بنجاح! 🎉")
                st.download_button("📥 تحميل ملف الترجمة", bio.getvalue(), "Translated.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
