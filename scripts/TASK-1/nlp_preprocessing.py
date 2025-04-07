import nltk
from transformers import AutoTokenizer, AutoModelForTokenClassification
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer # didn't used
import re
import emoji
from tqdm import tqdm
import torch
import pandas as pd

# this is file that I saved, provided in repo also
df = pd.read_csv('mental_health_postsV1.csv') # use the file_name saved via reddit extraction code

data = df.copy()

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) - {'no', 'not', 'nor', 'never'} # don't remove negation words

def preprocess_text_with_counts(text):
    if pd.isnull(text):
        return text, 0, 0, 0

    emoji_count = len([char for char in text if char in emoji.EMOJI_DATA])
    special_char_count = len(re.findall(r'[^A-Za-z\s]', text))
    stopword_count = len([word for word in text.split() if word.lower() in stop_words])

    text = emoji.replace_emoji(text, replace="")
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()

    # Remove stopwords
    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text, emoji_count, special_char_count, stopword_count

original_content = data['content'].copy()
emoji_changes = []
special_char_changes = []
stopword_changes = []

def process_row(row):
    processed_text, emoji_count, special_char_count, stopword_count = preprocess_text_with_counts(row)
    emoji_changes.append(emoji_count)
    special_char_changes.append(special_char_count)
    stopword_changes.append(stopword_count)
    return processed_text

for index, row in tqdm(data['content'].items(), total=len(data['content']), desc="Processing rows"):
    data.at[index, 'content'] = process_row(row)

changed_rows = (original_content != data['content']).sum()
data.to_csv('mental_health_postsV1_preprocessed.csv', index=False)

# Calculate total counts and rows with changes
total_emoji_removed = sum(emoji_changes)
total_special_chars_removed = sum(special_char_changes)
total_stopwords_removed = sum(stopword_changes)

rows_with_emoji_changes = sum(1 for count in emoji_changes if count > 0)
rows_with_special_char_changes = sum(1 for count in special_char_changes if count > 0)
rows_with_stopword_changes = sum(1 for count in stopword_changes if count > 0)

# Print the results
print(f"Number of rows changed: {changed_rows}")
print(f"Total emoji removed: {total_emoji_removed}")
print(f"Rows with emoji changes: {rows_with_emoji_changes}")
print(f"Total special characters removed: {total_special_chars_removed}")
print(f"Rows with special character changes: {rows_with_special_char_changes}")
print(f"Total stopwords removed: {total_stopwords_removed}")
print(f"Rows with stopword changes: {rows_with_stopword_changes}")
