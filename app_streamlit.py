import streamlit as st
import os
import tempfile
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Cheque OCR", layout="wide")

st.title("🧾 Cheque OCR System")
st.markdown("---")

# Try to import OCR
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=False, show_log=False, lang='en')
    ocr_available = True
    st.success("✅ OCR Engine Ready")
except:
    ocr_available = False
    st.error("❌ OCR Engine Not Available")

# File uploader
uploaded_files = st.file_uploader("Select Cheque Images", type=['png', 'jpg', 'jpeg', 'tiff'], accept_multiple_files=True)

if uploaded_files and ocr_available:
    results = []
    progress_bar = st.progress(0)
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # OCR
        result = ocr.ocr(tmp_path, cls=False)
        
        if result and result[0]:
            text = " ".join([line[1][0] for line in result[0]])
        else:
            text = "NO TEXT"
        
        results.append({"Image": uploaded_file.name, "Extracted Text": text})
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Cleanup
        os.unlink(tmp_path)
    
    # Show results
    st.subheader("Results")
    df = pd.DataFrame(results)
    st.dataframe(df)
    
    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", csv, "ocr_results.csv", "text/csv")

st.markdown("---")
st.caption("© MJ IT Solution")