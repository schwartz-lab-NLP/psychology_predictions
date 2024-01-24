import time
import re
# from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LAB_DATA_PATH = '/dsi/atzil-lab/'
TRANCRIPTION_TRANSLATIONS_FILE = '~/project/data/translatedDepText.csv'
SBS_FILE = '~/project/data/sbsDepDone.csv'
ENCODING = 'cp1255'  # maybe 'latin8'? This is what Anmol used.

TAG_REPLACEMENTS = {
    'cry': '(crying)',
    'laugh': '(laughing)',
    'character': 'them',
    'figure': 'them',
    'location': 'this place',
    'place': 'this place',
    'fill': 'hmm',
    'commotion': 'mm-hmm',
}

CONDITION_TO_TAG = {
    (lambda sbs: sbs['diff'] < 0): -1,
    (lambda sbs: sbs['diff'] > 1): 1,
    # Anmol used -1.3 and 1.3 as boundaries.
    # 0 and 1 give better balanced groups, as per Amir's suggestion.
}



class Preprocess:
    def __init__(self):
        self.df = None
        self.sbs = None
        self.original_preprocessed_df = None
    
    def run(self):
        self._load_csvs()
        self._replace_tags()
        self._remove_first_and_last_sessions()
        self._flatten_timestamp_rows()
        self.original_preprocessed_df = self.df.copy()
        self._add_tag_using_sbs()
        return self.df

    def _load_csvs(self):
        #self.df = pd.read_csv(TRANCRIPTION_TRANSLATIONS_FILE, encoding=ENCODING)
        self.df = pd.read_csv(TRANCRIPTION_TRANSLATIONS_FILE)
        self.sbs = pd.read_csv(SBS_FILE, encoding=ENCODING)

    def _replace_tags(self):
        """
        Replacing the annotated tags in the english text based on Amir's suggestions
        """
        compiled_regexes = {re.compile(fr'<([^<>]*{tag_part}[^<>]*)>', re.IGNORECASE): replacement
                            for tag_part, replacement in TAG_REPLACEMENTS.items()}
        
        def replace(text):
            if text is np.NaN:
                return np.NaN
            for regex, replacement in compiled_regexes.items():
                text = re.sub(regex, replacement, text)
                return text
        
        self.df['eng_event_plaintext'] = pd.Series([
            replace(text) for text in self.df.eng_event_plaintext
#            replace(text) for text in tqdm(self.df.eng_event_plaintext, total=self.df.shape[0])
        ])

    def _remove_first_and_last_sessions(self):
        """
        Removing the first and last session of each treatment (c_code) from the dataset
        """
        groups = self.df.groupby('c_code')['session_n']
        first_sessions = [f"{row.c_code}_{row.session_n}" 
                          for row in groups.min().reset_index().itertuples()]
        last_sessions = [f"{row.c_code}_{row.session_n}" 
                         for row in groups.max().reset_index().itertuples()]
        filterindexes = first_sessions + last_sessions

        self.df = self.df[~self.df.filterindex.isin(filterindexes)]

    def _flatten_timestamp_rows(self):
        """
        Flattening the dataframe - each timestamp is an individual row. melting this into a new column
        """
        timestamps = []
        for row in self.df.itertuples():
            event = row.event_speaker
            if event == 'Timestamp':
                timestamps.append(row.event_plaintext_full)
            else:
                timestamps.append(timestamps[-1])

        self.df['timestamp'] = pd.Series(timestamps).apply(lambda x: pd.to_datetime("".join(x.split()), format='%H:%M:%S'))
        self.df = self.df[self.df.event_speaker != 'Timestamp']

    def _add_tag_using_sbs(self):
        """
        Adds 'tag' attribute to self.df, 
        based on CONDITION_TO_TAG with conditions activated on SBS dataframe
        """
        self.sbs['idx'] = self.sbs.c_code + '_' + self.sbs.session_n.astype(str)
        self.sbs['diff'] = self.sbs.c_a_ors_Post_Session_1 - self.sbs.c_b_ors_Pre_Session_1
        
        conditions, tags = zip(*CONDITION_TO_TAG.items())
        bool_arrays = [condition(self.sbs) for condition in conditions]
        default = 0 if 0 not in tags else max(tags) + 1

        self.sbs['tag'] = np.select(bool_arrays, tags, default=default)
        
        self.df = pd.merge(self.df, self.sbs[['idx', 'tag']], left_on='filterindex', right_on='idx', how='left')
        self.df = self.df[self.df.tag.notna()]
        self.df = self.df[self.df.tag.isin(tags)]


def segment_to_10_mins(df):
    """
    Extract 10 minute segments from the whole session
    """
    training_ids = df.filterindex.unique()
    training_data = pd.DataFrame()

    for doc_id in training_ids:
        subset = df[df.filterindex == doc_id]
        startTime = subset.timestamp.min()
        endTime = subset.timestamp.max()
        ten_min = pd.Timedelta(minutes=10)
        filtered_subset = subset[(subset.timestamp >= startTime+ten_min) & (subset.timestamp <= endTime-ten_min)]
        training_data = pd.concat([training_data, filtered_subset], ignore_index=True)

    training_data.timestamp = training_data.timestamp.apply(lambda x: x.time())

    for idx, group in training_data.groupby('filterindex'):
        if group.shape[0] < 10:
            continue
        speakers = group.dialog_turn_main_speaker
        group['turn_n'] = speakers.ne(speakers.shift()).cumsum()
        for i in range(group.turn_n.min(), group.turn_n.max(), 10):
            turns = group[(group.turn_n >= i) & (group.turn_n < i + 10)][['dialog_turn_main_speaker', 'event_speaker', 'eng_event_plaintext', 'turn_n']]

            if group.tag.unique()[0] == 1:
                turns.to_csv(f'./training_docs/good_sessions/{idx}_{i}:{i+10}.csv', index=None)
            elif group.tag.unique()[0] == -1:
                turns.to_csv(f'./training_docs/poor_sessions/{idx}_{i}:{i+10}.csv', index=None)

    og = og[~og.filterindex.isin(training_ids)]
    anno_doc_ids = np.random.choice(og.filterindex.unique(), size=350, replace=False)
    anno_data = pd.DataFrame()

    for doc_id in anno_doc_ids:
        subset = og[og.filterindex==doc_id]
        startTime = subset.timestamp.min()
        endTime = subset.timestamp.max()
        ten_min = pd.Timedelta(minutes=10)
        filtered_subset = subset[(subset.timestamp >= startTime+ten_min) & (subset.timestamp <= endTime-ten_min)]
        anno_data = pd.concat([anno_data, filtered_subset], ignore_index=True)

    anno_data.timestamp = anno_data.timestamp.apply(lambda x: x.time())

    dialogue_turns = {}
    for idx, group in anno_data.groupby('filterindex'):
        if group.shape[0]<10: continue
        speakers = group.dialog_turn_main_speaker
        group['turn_n'] = speakers.ne(speakers.shift()).cumsum()
        max_turn = group.turn_n.max()
        random_turn = max_turn+1
        while (random_turn+12>max_turn) and (random_turn!=1):
            random_turn = np.random.choice(group.turn_n, 1)[0]

        turns = group[(group.turn_n>=random_turn) & (group.turn_n<random_turn+11)][['dialog_turn_main_speaker', 'event_speaker', 'eng_event_plaintext', 'turn_n']]
        dialogue_turns[idx] = list(turns.itertuples(name=None, index=False))
        turns.to_csv(f'./anno_docs/{idx}.csv', index=None)


if __name__ == '__main__':
    start = time.time()
    try:
        df = Preprocess().run()
        print("Dataframe shape:", df.shape)
        print("Group sizes by tag:")
        print(df[['filterindex', 'tag']].reset_index().groupby('filterindex').max().groupby('tag').count().iloc[:, 0])
        # segment_to_10_mins(df)
    finally:
        end = time.time()
        print(f"Run took {end - start} seconds")
