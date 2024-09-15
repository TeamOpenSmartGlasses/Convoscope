from tinydb import TinyDB, Query
from datetime import datetime, timedelta

class DataStorage:
    def __init__(self, db_file='.db.json'):
        self.db = TinyDB(db_file)

    def store_data(self, transcript):
        """
        Store a new transcript in the database.
        """
        self.db.insert(transcript)

    def get_transcripts_from_last_n_seconds(self, n_seconds):
        """
        Get transcripts from the last n seconds.
        """
        time_threshold = datetime.utcnow() - timedelta(seconds=n_seconds)
        Transcript = Query()
        return self.db.search(Transcript.timestamp >= time_threshold.isoformat())

    def get_all_transcripts(self):
        """
        Get all transcripts.
        """
        return self.db.all()

# Example usage in TPA client
# storage = DataStorage()

# # Store a new transcript
# new_transcript = {
#     'text': 'This is a new transcript.',
#     'timestamp': datetime.utcnow().isoformat(),
# }
# storage.store_transcript(new_transcript)

# # Retrieve transcripts from the last 30 seconds
# recent_transcripts = storage.get_transcripts_from_last_n_seconds(30)
# print(recent_transcripts)
