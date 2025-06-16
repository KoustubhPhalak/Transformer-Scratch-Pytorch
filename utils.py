

# Create combined training corpus for SentencePiece
def build_corpus(input_files, output_path, lang='en'):
    with open(output_path, 'w', encoding='utf-8') as f_out:
        if lang == 'en':
            lang_file = input_files['en']
        else:
            lang_file = input_files['de']
        with open(lang_file, 'r', encoding='utf-8', errors='ignore') as f_in:
            f_out.writelines(f_in.readlines())

# Split data into train and test
def get_split(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        split = 1000000
        return lines[:split], lines[split:]