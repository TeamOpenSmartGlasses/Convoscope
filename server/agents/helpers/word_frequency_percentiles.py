import pandas as pd
import os
import jieba

# Mapping of incoming language codes to the FrequencyWords repository's folder names
language_code_map = {
    "English": "en",
    "Spanish": "es",
    "Russian": "ru",
    # "Japanese": "ja",
    "French": "fr",
    "Chinese": "zh_cn",  # Assuming simplified Chinese; adjust if necessary for traditional
    "Chinese (Hanzi)": "zh_cn", # Assuming simplified Chinese; adjust if necessary for traditional
    "Chinese (Pinyin)": "zh_cn", # Assuming simplified Chinese; adjust if necessary for traditional
    "German": "de",
    "Arabic": "ar",
    "Korean": "ko",
    "Italian": "it",
    "Turkish": "tr",
    "Portuguese": "pt"
}


def load_frequency_list(language_code: str, type: str) -> pd.DataFrame:
    """
    Load the frequency list for the specified language.
    """
    if language_code not in language_code_map:
        raise ValueError(f"Unsupported language code: {language_code}")

    # Construct the path to the frequency list
    folder_name = language_code_map[language_code]
    file_path = os.path.join("./word_frequency_lists/content/2018", folder_name, f"{folder_name}_{type}.txt")

    # Load the frequency list into a DataFrame
    df = pd.read_csv(file_path, sep=" ", names=['word', 'frequency'], header=None)
    df['rank'] = df['frequency'].rank(ascending=False)
    return df

# Load frequency lists at the start
freq_dicts = {}

# Attempt to preload both '50k' and 'full' frequency lists, with fallback
for lang_code in language_code_map.keys():
    try:
        # Attempt to load the '50k' frequency list
        df_50k = load_frequency_list(lang_code, '50k')
        freq_dicts[(lang_code, '50k')] = pd.Series(df_50k['rank'].values, index=df_50k['word']).to_dict()
    except FileNotFoundError:
        # If '50k' file does not exist
        pass

    # Ensure the 'full' list is always loaded
    df_full = load_frequency_list(lang_code, 'full')
    freq_dicts[(lang_code, 'full')] = pd.Series(df_full['rank'].values, index=df_full['word']).to_dict()

def get_word_frequency_percentiles(transcript: str, language_code: str) -> dict:
    words_dict_key = (language_code, '50k') if (language_code, '50k') in freq_dicts else (language_code, 'full')
    words_dict = freq_dicts[words_dict_key]
    total_words = len(words_dict)  # Get total words from the selected dictionary

    # Tokenize the transcript based on the language
    if language_code == "Chinese" or language_code == "Chinese (Pinyin)":
        words = list(jieba.cut(transcript))
    else:
        words = transcript.split()

    words_rank = {}

    for word in words:
        # For languages other than Chinese, consider lowercasing the word to improve match rate
        word_lower = word.lower() if language_code in ["English", "Spanish", "French"] else word
        word_lower_token = word_lower.split("'")[0].translate(str.maketrans("", "", "?。!.,;？\""))
        if word_lower_token in words_dict:
            # Convert rank to percentile
            percentile = round((words_dict[word_lower_token] / total_words) * 100, 1)
            words_rank[word.translate(str.maketrans("", "", "?。!.,;？\""))] = percentile
            #words_rank[word] = percentile
        else: #large percentile for words not in small dataset
            percentile = 98.2
            words_rank[word] = percentile

    # print("WORDS RANK:")
    # print(words_rank)

    return words_rank


if __name__ == "__main__":
    to_test_str = "你好，你今天过得怎么样？ 这是一个协同效应。"
    word_freqs = get_word_frequency_percentiles(to_test_str, "Chinese")
    print(word_freqs)
