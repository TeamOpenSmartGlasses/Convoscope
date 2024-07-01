from string import punctuation


explicit_wake_words = [
    "hey mira",
    "he mira",

    "hey mara",
    "he mara",

    "hey mirror",
    "he mirror",

    "hey miara",
    "he miara",

    "hey mia",
    "he mia",

    "hey mural",
    "he mural"
]


def does_text_contain_wake_word(transcript):
    transcript_low = transcript.lower()
    for term in explicit_wake_words:
        if term in transcript_low:
            return True

    return False


def get_explicit_query_from_transcript(transcript):
    transcript_low = transcript.lower().replace("，", " ").replace(",", "")
    print(transcript_low)
    for term in explicit_wake_words:
        if term in transcript_low:
            index = transcript_low.rfind(term) + len(term)
            base_query = transcript_low[index:]
            return base_query.strip(punctuation).strip()

    return None
