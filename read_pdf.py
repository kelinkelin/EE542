#!/usr/bin/env python3
import sys
import os

# Install and use pdfplumber
try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    os.system('pip3 install pdfplumber --break-system-packages -q')
    import pdfplumber

pdf_path = sys.argv[1]
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            print(text)
            print("\n" + "="*80 + "\n")
