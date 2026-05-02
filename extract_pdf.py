import fitz # PyMuPDF

pdf_path = "/Users/saithejas/Desktop/FINAL_PROJECT_MCA/PassWordPaper_12.pdf"
txt_path = "/Users/saithejas/Desktop/FINAL_PROJECT_MCA/paper_text.txt"

doc = fitz.open(pdf_path)
text = ""
for page in doc:
    text += page.get_text()

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Extracted {len(text)} characters.")
