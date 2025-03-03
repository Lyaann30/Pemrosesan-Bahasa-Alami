import nltk

# Coba download ulang 'punkt'
nltk.download('punkt')
nltk.download('punkt_tab')

# Coba tokenisasi sederhana
from nltk.tokenize import word_tokenize

text = "Hello, this is a test to check NLTK tokenization."
tokens = word_tokenize(text)

print("Tokenized words:", tokens)
