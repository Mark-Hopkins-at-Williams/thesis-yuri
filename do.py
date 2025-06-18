# -*- coding: utf-8 -*-
from pathlib import Path
from collections import defaultdict


DATA_DIR   = Path(".")        
OUT_DIR    = Path("./data")  
OUT_DIR.mkdir(exist_ok=True)


LANGS = [
    "bg", "cs", "da", "de", "el",
    "es", "et", "fi", "fr", "hu",
    "it", "lt", "lv", "nl", "pl",
    "pt", "ro", "sk", "sl", "sv",
]


# 1.  BUILD LIST OF EXISTING PAIRS

pairs      = []       # (english_path, lang_path, code)
found_lang = []

for code in LANGS:
    en_file  = DATA_DIR / f"europarl-v7.{code}-en.en"
    xx_file  = DATA_DIR / f"europarl-v7.{code}-en.{code}"
    if en_file.exists() and xx_file.exists():
        pairs.append((en_file, xx_file, code))
        found_lang.append(code)
    else:
        print(f"Missing files for {code}: skipped")

if not pairs:
    raise RuntimeError("No language pairs found – check file names / paths.")

print(f" using {len(found_lang)} languages: {', '.join(found_lang)}")


# 2.  BUILD:  english_sentence  →  {code: translation}

table = defaultdict(dict)

for en_path, xx_path, code in pairs:
    with en_path.open(encoding="utf-8") as f_en, xx_path.open(encoding="utf-8") as f_xx:
        for en_line, xx_line in zip(f_en, f_xx):
            en = en_line.rstrip()
            xx = xx_line.rstrip()
            table[en][code] = xx


# 3.  KEEP SENTENCES PRESENT IN *EVERY* LANGUAGE
aligned_en = [s for s, d in table.items() if len(d) == len(found_lang)]
print(f"{len(aligned_en):,} sentences aligned across all languages")

# 4.  WRITE FINAL OUTPUT FILES

full_name = {
    "bg": "Bulgarian",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "hu": "Hungarian",
    "it": "Italian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
}
writers = {
    "en": (OUT_DIR / "english.txt").open("w", encoding="utf-8", newline="\n")
}
for code in found_lang:
    writers[code] = (OUT_DIR / f"{full_name[code]}.txt").open("w", encoding="utf-8", newline="\n")

for en in aligned_en:
    writers["en"].write(en + "\n")
    for code in found_lang:
        writers[code].write(table[en][code] + "\n")

for f in writers.values():
    f.close()

print(" Done ")




