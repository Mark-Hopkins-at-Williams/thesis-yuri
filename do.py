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


# set up the ist of pairs and languages

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


print(f"Found 20 {len(found_lang)} languages: {', '.join(found_lang)}. The data is being procesed, please wait.")


# Build a dictionary. 
# Key: en sentence 
# Value: dictionary of tranlations into other languages 
  # Key: language code xx
  # Vaue: translation into xx language

table = defaultdict(dict) # using dict allows to eleminate duplicates

for en_path, xx_path, code in pairs:
    with en_path.open(encoding="utf-8") as f_en, xx_path.open(encoding="utf-8") as f_xx:
        for en_line, xx_line in zip(f_en, f_xx):
            en = en_line.rstrip() # the english of xx language
            xx = xx_line.rstrip() # the translation into xx language
            table[en][code] = xx #


# Only keep the sentences that appear in every single language 
keep_en = [s for s, d in table.items() if len(d) == len(found_lang)] 


# Outputiing files 

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
}  # used for proper file names 


output = {
    "en": (OUT_DIR / "english.txt").open("w", encoding="utf-8", newline="\n")
}

for code in found_lang:
    output[code] = (OUT_DIR / f"{full_name[code]}.txt").open("w", encoding="utf-8", newline="\n")

for en in keep_en:
    output["en"].write(en + "\n") #write english file
    for code in found_lang:
        output[code].write(table[en][code] + "\n") # write each xx language file

for file in output.values():
    file.close()

print(" Done ")


