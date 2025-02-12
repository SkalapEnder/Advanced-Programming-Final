def truncate_text(text, word_limit=10000):
    return " ".join(text.split()[:word_limit])