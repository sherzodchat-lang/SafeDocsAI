import re
text = "2023\n2024\n2025"
text = text.replace("\r\n", "\n").replace("\r", "\n")
text = re.sub(r"-\s*\n\s*", "", text)
text = re.sub(r"[ \t]+", " ", text)
text = re.sub(r"\n{3,}", "\n\n", text)
text = text.strip()
print(repr(text))
