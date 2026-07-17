from layer1_ingestion import load_resume

text = load_resume("C:/Users/tahsk/Downloads/23csu031_resume.pdf")
print(f"Total chars: {len(text)}")
print("---")
print(repr(text[:500]))  # repr shows hidden characters, spaces, newlines