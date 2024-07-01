# Deployment Guide
1. Clone repo
2. Configure `server_config.py` (see details below)
3. Create a venv
4. Install deps `pip install -r requirements.txt`
5. Run server `python server.py`

# Get Data

Data sources used for `english_word_freq_list`:
- Google: https://github.com/garyongguanjie/entrie/blob/main/unigram_freq.csv or https://www.kaggle.com/datasets/rtatman/english-word-frequency (there are other mirrors as well)
- Norvig: https://github.com/arstgit/high-frequency-vocabulary (30k.txt, change to .csv and add a single line at the top that just says "word" (CSV header))

# Parse Bookmarks
If you want to parse your bookmarks, use the `parse_bookmarks.py` file (see comment on top of that file for usage).
