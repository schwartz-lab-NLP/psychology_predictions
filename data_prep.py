import time
import re
# from tqdm import tqdm
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt


LAB_DATA_PATH = '/dsi/atzil-lab/'
TRANCRIPTION_TRANSLATIONS_FILE = '/workspace/project/data/translatedDepText.csv'
SBS_FILE = '/workspace/project/data/sbsDepDone.csv'
ENCODING = 'cp1255'  # maybe 'latin8'? This is what Anmol used.
SAVE_FILE = '/workspace/project/data/preprocessed_20240207.csv'

# Mainly so that the train/val/test sets will have the same clients each time we run this
RAND_SEED = 7337
GENERATOR = np.random.default_rng(seed=RAND_SEED)

SETS_PERCENTAGE = (0.75, 0.125, 0.125)

TAG_REPLACEMENTS = {
    'cry': '(crying)',
    'laugh': '(laughing)',
    'character': 'them',
    'figure': 'them',
    'location': 'this place',
    'place': 'this place',
    'fill': 'hmm',
    'commotion': 'mm-hmm',
    'hum': 'mm-hmm',
}

CONDITION_TO_TAG = {
    (lambda sbs: sbs['diff'] < 0): -1,
    (lambda sbs: sbs['diff'] > 1): 1,
    # Anmol used -1.3 and 1.3 as boundaries.
    # 0 and 1 give better balanced groups, as per Amir's suggestion.
}

# FINAL_COLUMNS = ['idx', 'c_code', 'session_n', 'event_n', 
#                  'dialog_turn_main_speaker', 'dialog_turn_number', 
#                  'eng_event_plaintext', 'timestamp', 'diff', 'tag']
SESSION_GROUP_COLUMNS = ['idx', 'c_code', 'session_n', 'diff', 'tag']

THERAPIST_SPEECH_TURN_TEMPLATE = 'T: "{text}"'
CLIENT_SPEECH_TURN_TEMPLATE = 'C: "{text}"'
ANNOTATOR_SPEECH_TURN_TEMPLATE = 'Annotation: {text}'
SPEECH_TURNS_DELIMITER = "\n"
DIALOG_TURN_TO_TEMPLATE = {
    "Therapist": THERAPIST_SPEECH_TURN_TEMPLATE,
    "Client": CLIENT_SPEECH_TURN_TEMPLATE,
    "Annotator": ANNOTATOR_SPEECH_TURN_TEMPLATE,
}




class Preprocess:
    def __init__(self, divide_into_sets=True, verbose=True):
        self.divide_into_sets = divide_into_sets
        self.verbose = verbose

        self.df = None
        self.sbs = None
        self.original_preprocessed_df = None
        self.final_speech_turns_df = None
        self.df_train, self.df_val, self.df_test = None, None, None
        self._start, self._end, self._runtime = None, None, None
    
    def verbose_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    
    def run(self):
        self._start = time.time()
        
        self._load_csvs()
        self._replace_tags()
        self._remove_first_and_last_sessions()
        self._flatten_timestamp_rows()
        
        self.original_preprocessed_df = self.df.copy()
        
        self._add_tag_using_sbs()
        self._unify_sessions()
                        
        if self.divide_into_sets:
            self._divide_into_sets()
            return self._return(self.df_train, self.df_val, self.df_test)
        
        return self._return(self.df)

    def _return(self, *return_value):
        self._end = time.time()
        self._runtime = self._end - self._start
        self.verbose_print(f"Finished Preprocessing in: {round(self._runtime, 2)} seconds")
        return return_value

    def _load_csvs(self):
        self.verbose_print("Loading csvs...")
        #self.df = pd.read_csv(TRANCRIPTION_TRANSLATIONS_FILE, encoding=ENCODING)
        self.df = pd.read_csv(TRANCRIPTION_TRANSLATIONS_FILE)
        self.sbs = pd.read_csv(SBS_FILE, encoding=ENCODING)

    def _replace_tags(self):
        """
        Replacing the annotated tags in the english text based on Amir's suggestions
        """
        self.verbose_print("Replacing <> tags...")
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
        self.verbose_print("Removing first/last sessions for each client...")
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
        self.verbose_print("Extracting 'timestamp' rows as new column...")
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
        self.verbose_print("Adding ORS diff tag from SBS...")
        self.sbs['idx'] = self.sbs.c_code + '_' + self.sbs.session_n.astype(str)
        self.sbs['diff'] = self.sbs.c_a_ors_Post_Session_1 - self.sbs.c_b_ors_Pre_Session_1
        
        conditions, tags = zip(*CONDITION_TO_TAG.items())
        bool_arrays = [condition(self.sbs) for condition in conditions]
        default = 0 if 0 not in tags else max(tags) + 1

        self.sbs['tag'] = np.select(bool_arrays, tags, default=default)
        
        self.df = pd.merge(self.df, self.sbs[['idx', 'diff', 'tag']], left_on='filterindex', right_on='idx', how='left')
        self.df = self.df[self.df.tag.notna()]
        self.df = self.df[self.df.tag.isin(tags)]

        self.verbose_print("Group sizes by tag:")
        self.verbose_print(self.df[['filterindex', 'tag']].reset_index().groupby('filterindex').max().groupby('tag').count().iloc[:, 0])
    
    def _unify_sessions(self):
        """
        Unifies speech turns into a single dataframe row for each session, with the full treatment text.
        """
        self.verbose_print("Unifying session speech turns...")

        self.df = self.df[self.df.dialog_turn_main_speaker.isin(set(DIALOG_TURN_TO_TEMPLATE))]
        self.df = self.df[~self.df.eng_event_plaintext.isna()]
        self.df = self.df.copy()
        
        self.df['eng_event_plaintext_fmt'] = self.df.apply(
            lambda row: DIALOG_TURN_TO_TEMPLATE[row["dialog_turn_main_speaker"]].format(text=row["eng_event_plaintext"]),
            axis=1
        )

        sessions_text = self.df.groupby(SESSION_GROUP_COLUMNS)['eng_event_plaintext_fmt'].agg(SPEECH_TURNS_DELIMITER.join)
        sessions = sessions_text.reset_index().rename(columns={'eng_event_plaintext_fmt': 'eng_session_plaintext'})

        self.final_speech_turns_df = self.df
        self.df = sessions

        self.verbose_print("Sessions df columns:", self.df.columns)
        self.verbose_print("Sessions df shape:", self.df.shape)
    
    def _divide_into_sets(self):
        """
        Divides self.df into train, val, and test sets
        based on SETS_PERCENTAGE constant.
        """
        self.verbose_print("Dividing into train/val/test sets...")
        clients = sorted(set(self.df.c_code))
    
        n_clients = len(clients)
        n_train = round(SETS_PERCENTAGE[0] * n_clients)
        n_val = round(SETS_PERCENTAGE[1] * n_clients)
        n_test = n_clients - n_train - n_val
        self.verbose_print(f"Clients in each set: Train [{n_train}], Val [{n_val}], Test [{n_test}]")

        GENERATOR.shuffle(clients)

        c_train = clients[:n_train]
        c_val = clients[n_train:n_train + n_val]
        c_test = clients[-n_test:]

        self.verbose_print("First and last 5 clients in train set:", c_train[:5], c_train[-5:])

        self.df_train = self.df[self.df.c_code.isin(c_train)]
        self.df_val = self.df[self.df.c_code.isin(c_val)]
        self.df_test = self.df[self.df.c_code.isin(c_test)]

        self.verbose_print(f"Speech turns in each set: "
              f"Train [{self.df_train.shape[0]}, {round((self.df_train.shape[0] / self.df.shape[0]) * 100, 1)}%], "
              f"Val [{self.df_val.shape[0]}, {round((self.df_val.shape[0] / self.df.shape[0]) * 100, 1)}%], "
              f"Test [{self.df_test.shape[0]}, {round((self.df_test.shape[0] / self.df.shape[0]) * 100, 1)}%]")

        self.verbose_print(f"Tag averages in each set: "
              f"Train [{round(self.df_train.tag.mean(), 5)}], "
              f"Val [{round(self.df_val.tag.mean(), 5)}], "
              f"Test [{round(self.df_test.tag.mean(), 5)}]")

        # for s, name in ((self.df_train, 'train'),
        #                 (self.df_val, 'val'),
        #                 (self.df_test, 'test')):
        #     s.to_csv(SAVE_FILE_PREFIX + name + '.csv')


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
    anno_doc_ids = GENERATOR.choice(og.filterindex.unique(), size=350, replace=False)
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
            random_turn = GENERATOR.choice(group.turn_n, 1)[0]

        turns = group[(group.turn_n>=random_turn) & (group.turn_n<random_turn+11)][['dialog_turn_main_speaker', 'event_speaker', 'eng_event_plaintext', 'turn_n']]
        dialogue_turns[idx] = list(turns.itertuples(name=None, index=False))
        turns.to_csv(f'./anno_docs/{idx}.csv', index=None)


if __name__ == '__main__':
    start = time.time()
    try:
        df = Preprocess().run()
        print("Dataframe shape:", df.shape)
        # df.to_csv(SAVE_FILE)
        # segment_to_10_mins(df)
    finally:
        end = time.time()
        print(f"Run took {end - start} seconds")
