import os
import pymupdf4llm
import config

def convert_all_pdfs():
    # 1. Create data directory if it doesn't exist
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)

    # 2. Iterate through all PDFs in the data folder
    for filename in os.listdir(config.DATA_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(config.DATA_DIR, filename)
            md_path = os.path.join(config.DATA_DIR, filename.replace(".pdf", ".md"))
            
            print(f"Converting {filename} to Markdown...")
            
            # Convert PDF to MD string
            md_text = pymupdf4llm.to_markdown(pdf_path)
            
            # Save the file
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_text)
                
    print("✅ Conversion complete. You can now run ingest.py")

if __name__ == "__main__":
    convert_all_pdfs()