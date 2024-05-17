# import ptvsd
# ptvsd.enable_attach(address=('0.0.0.0', 5678))

import time
import re
from collections import Counter
# from tqdm import tqdm
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt


LAB_DATA_PATH = '/dsi/atzil-lab/'
TRANCRIPTION_TRANSLATIONS_FILE = '/workspace/project/data/translatedDepText.csv'
SBS_FILE = '/workspace/project/data/sbsDepDone.csv'
ENCODING = 'cp1255'  # maybe 'latin8'? This is what Anmol used.
SAVE_FILE = '/workspace/project/data/preprocessed_20240207.csv'

THROW_AWAY_TOO_SHORT_SESSIONS = True
MIN_TIMESTAMP = pd.Timestamp('1900-01-01 00:00:00')
WORKING_TIME_LENGTH = pd.Timedelta(15, unit='minute')
WORKING_TIME_FROM_END_OF_SESSION = True
# make negative if WORKING_TIME_FROM_END_OF_SESSION=True, 
#  and positive if WORKING_TIME_FROM_END_OF_SESSION=False
WORKING_TIME_BEGIN_DELTA = pd.Timedelta(-20, unit='minute')

IGNORE_SESSIONS = []

# Mainly so that the train/val/test sets will have the same clients each time we run this
RAND_SEED = 7577
# This seed should give very balanced-out sets:
#  Sessions in each set: Train [252, 74.3%], Val [44, 13.0%], Test [43, 12.7%]
#  Tag averages in each set: Train [0.10317], Val [0.09091], Test [0.16279]
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
    'sigh': '(sigh)',
    'cluck': '(cluck)',
    'shudder': '(shudder)',
    'silence': '(silence)',
}
EXACT_TAG_REPLACEMENTS = {
    'q': '...'
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

# SESSION_PREFIX = "Below is part of a psychology session transcript, with the therapist (T:) and the client (C:):\n"
# SESSION_POSTFIX = "\nAnswer in one word: Did the session improve the client's wellbeing? (Yes/No): "
# SESSION_PREFIX = 'Below (delimited by triple quotes) is part of a psychology session transcript, with the therapist (T:) and the client (C:):\n"""'
# SESSION_POSTFIX = '"""\nBased on this psychology session, did the session improve the client\'s wellbeing? Answer in one word (Yes/No): '

# SESSION_PREFIX = 'Below (delimited by triple quotes) is part of a psychology session transcript, with the therapist (T:) and the client (C:) speech turns on alternating lines:\n"""'
# SESSION_POSTFIX = '"""\nAnswer the following three questions. ' \
#                   'Did you understand all that was written? (Yes/No): Yes. ' \
#                   'Did this transcript cover the whole session? (Yes/No): No. ' \
#                   'Based on this psychology session, did the session improve the client\'s wellbeing? (Yes/No): '

SESSION_PREFIX = '**Task: Binary Classification**\n' \
                 '**Classes: Yes/No**\n' \
                 '**Input Text Description: Part of a psychology session transcript, ' \
                 'with the therapist (T:) and the client (C:) speech turns on alternating lines**\n' \
                 '**Input Text: '
SESSION_POSTFIX = '**\n' \
                  '**Question: Did the session improve the client\'s wellbeing?**\n' \
                  '**Answer (Yes/No): '


class Preprocess:
    def __init__(self, divide_into_sets=True, verbose=True):
        self.divide_into_sets = divide_into_sets
        self.verbose = verbose

        self.df = None
        self.sbs = None
        self.final_speech_turns_df = None
        self.df_train, self.df_val, self.df_test = None, None, None
        self._start, self._end, self._runtime = None, None, None
    
    def verbose_print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
    
    def run(self):
        self._start = time.time()
        
        self._load_csvs()
        self._manually_handle_special_sessions()
        self._remove_first_and_last_sessions()
        self._flatten_timestamp_rows()
        self._consolidate_timestamp_values()

        self.df = self.df[~self.df.eng_event_plaintext.isna()].copy()
        
        self._fix_timestamp_anomalies()
        self._extract_working_time()
        self._replace_tags()
        
        self._add_tag_using_sbs()
        self._unify_sessions()
                        
        if self.divide_into_sets:
            self._divide_into_sets()
            return self._return(self.df_train, self.df_val, self.df_test)
        
        return self._return(self.df)

    def _return(self, *return_value):
        self._end = time.time()
        self._runtime = self._end - self._start
        self.verbose_print(f"Finished Preprocessing in: {round(self._runtime, 2)} seconds\n")
        return return_value

    def _load_csvs(self):
        self.verbose_print("Loading csvs...")
        #self.df = pd.read_csv(TRANCRIPTION_TRANSLATIONS_FILE, encoding=ENCODING)
        self.df = pd.read_csv(TRANCRIPTION_TRANSLATIONS_FILE)
        self.sbs = pd.read_csv(SBS_FILE, encoding=ENCODING)
    
    def _manually_handle_special_sessions(self):
        # 'OE9233_2'  # This session has 46 lines that were accidentally duplicated, and then translated twice, differently.
        # events at ilocs 137:183 == 183:229 (before preprocessing)
        # events at ilocs 76:108 == 108:140 (after preprocessing)
        ses = self.df[self.df.filterindex == 'OE9233_2']
        a = ses.iloc[137:183]
        b = ses.iloc[183:229]
        assert (a.event_plaintext_full.to_numpy() == b.event_plaintext_full.to_numpy()).all(), "Check again the indexes of 'OE9233_2's duplicated lines"
        self.df = self.df[(self.df.filterindex != 'OE9233_2') | (~self.df.index.isin(b.index))].copy()

        if IGNORE_SESSIONS:
            self.verbose_print(f"Ignoring {len(IGNORE_SESSIONS)} faulty sessions...")
            self.df = self.df[~self.df.filterindex.isin(IGNORE_SESSIONS)].copy()

    def _replace_tags(self):
        """
        Replacing the annotated tags in the english text based on Amir's suggestions
        """
        self.verbose_print("Replacing <> tags...")
        compiled_regexes = {re.compile(fr'<([^<>]*{tag_part}[^<>]*)>', re.IGNORECASE): replacement
                            for tag_part, replacement in TAG_REPLACEMENTS.items()}
        exact_compiled_regexes = {re.compile(fr'<({tag_part})>', re.IGNORECASE): replacement
                                  for tag_part, replacement in EXACT_TAG_REPLACEMENTS.items()}
        all_tags_compiled_regex = re.compile(r'<([^<>]*)>', re.IGNORECASE)
        
        def replace(text):
            if text is np.NaN:
                return np.NaN
            for regex, replacement in compiled_regexes.items():
                text = re.sub(regex, replacement, text)
            for regex, replacement in exact_compiled_regexes.items():
                text = re.sub(regex, replacement, text)
            text = re.sub(all_tags_compiled_regex, '', text).replace('  ', ' ')
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

        self.df = self.df[~self.df.filterindex.isin(filterindexes)].copy()

    def _flatten_timestamp_rows(self):
        """
        Flattening the dataframe - each timestamp is an individual row. melting this into a new column
        """
        self.verbose_print("Extracting 'timestamp' rows as new column...")

        # Get timestamps from timestamp texts
        timestamps = self.df.event_plaintext_full.where(self.df.event_speaker == 'Timestamp')
        timestamps = timestamps.str.replace(' ', '')
        timestamps = pd.to_datetime(timestamps, format='%H:%M:%S')

        self.df['timestamp'] = timestamps

        # Push timestamps forward to cover windows until next timestamp
        self.df['timestamp'] = self.df.groupby('filterindex')['timestamp'].ffill()

        # Pull timestamps back to cover times when the session does not begin with a timestamp
        self.df['timestamp'] = self.df.groupby('filterindex')['timestamp'].bfill()

        self.df = self.df[self.df.event_speaker != 'Timestamp'].copy()

    def _consolidate_timestamp_values(self):
        """
        Subtracts the beginning time from each session's timestamp so each session begins at time 00:00.
        """
        self.verbose_print("Consolidating timestamp values to begin at 00:00:00...")

        timestamp_diffs = self.df.groupby(self.df.filterindex).timestamp.transform('min') - MIN_TIMESTAMP
        self.df['timestamp'] = self.df['timestamp'] - timestamp_diffs

        self.df = self.df.copy()

    def _fix_timestamp_anomalies(self):
        """
        Fixes various timestamp tagging errors
        """
        self.verbose_print("Fixing timestamp tagging anomalies...")
        # fix typos resulting in super high times
        typos = self.df[self.df.timestamp > pd.Timestamp('1900-01-01 05:00:00')].timestamp
        hours_to_subtract = pd.to_timedelta(typos.dt.hour, unit='hour')
        self.df.loc[hours_to_subtract.index, 'timestamp'] = self.df.loc[hours_to_subtract.index, 'timestamp'] - hours_to_subtract

        # fix non-monotonously increasing timestamps in a session
        self.verbose_print("Cleaning timestamps making session times non-monotonously increasing...")
        before_timestamps = self.df.timestamp.copy()

        search = True
        counter = 0
        while search:
            counter += 1
            self.verbose_print(f'\tIteration {counter}: ', end='')
            problematic_idxs = []
            for session_label, full_session in self.df.groupby(self.df.filterindex):
                if not full_session.timestamp.is_monotonic_increasing:
                    problems_idxs, answers_idxs = self._get_timestamp_error_fixes(session_label, full_session)
                    problematic_idxs.extend(zip(problems_idxs, answers_idxs))
            
            if problematic_idxs:
                self.verbose_print(f"{len(problematic_idxs)} anomalies found...")
                problems_idxs_full, answers_idxs_full = zip(*problematic_idxs)
                self.df.loc[problems_idxs_full, ['timestamp']] = self.df.loc[answers_idxs_full, :].timestamp.to_numpy()
            else:
                self.verbose_print("0 anomalies found!")
                search = False
        
        after_timestamps = self.df.timestamp.copy()
        changes = before_timestamps != after_timestamps
        self.verbose_print(f"\t{changes.sum()} changes made in timestamps "
                           f"(out of {self.df.shape[0]} - "
                           f"{round((changes.sum() / self.df.shape[0]) * 100, 2)}%)")
        
        c = Counter(list(self.df[changes].filterindex))
        self.verbose_print("\t3 most changed sessions:")
        for session_label, count in c.most_common(3):
            self.verbose_print('\t', session_label, '-', count, 'changes')
        
        self.df = self.df.copy()
    
    def _get_timestamp_error_fixes(self, session_label, full_session):
        """
        This function finds all of the timestamp tagging mistakes in the given session,
         (based on the non-monotonity of the timestamp series)
         and decides if the mistake jumps up or down.
        It returns two lists, mapping each error index in the first list 
         to an index of a row with a timestamp value it should copy, to fix the mistake.
        """
        problems_idxs, answers_idxs = [], []

        for i, (idx, ts) in enumerate(full_session.timestamp.items()):
            if i == 0:
                continue
            prev_ts = full_session.timestamp.iloc[i - 1]
            if prev_ts > ts:
                # Count previous timestamps that are later (bigger) than this ts
                potential_high_problems = []
                j = i - 1
                while j >= 0 and full_session.timestamp.iloc[j] > ts:
                    potential_high_problems.append(full_session.iloc[j].name)
                    j -= 1
                
                # Count next timestamps that are earlier (smaller) than prev_ts
                potential_low_problems = []
                j = i
                while j < full_session.shape[0] and full_session.timestamp.iloc[j] < prev_ts:
                    potential_low_problems.append(full_session.iloc[j].name)
                    j += 1
                
                # pick the change that will change the least amount of values
                if len(potential_high_problems) <= len(potential_low_problems):
                    problems_idxs.extend(potential_high_problems)
                    answers_idxs.extend([idx] * len(potential_high_problems))
                else:
                    problems_idxs.extend(potential_low_problems)
                    answers_idxs.extend([full_session.iloc[i - 1].name] * len(potential_low_problems))

        return problems_idxs, answers_idxs
    
    def _extract_working_time(self):
        """
        Extracts working time (first 15 minutes of last 20 minutes of session) from each session.
        This shortens the context length so we can use it in our models.
        """
        self.verbose_print("Extracting working time from each session")

        df_groupby_session = self.df.groupby(self.df.filterindex)
        timestamp_session_mins = df_groupby_session.timestamp.min()
        timestamp_session_maxs = df_groupby_session.timestamp.max()

        comparison_timestamp = timestamp_session_maxs if WORKING_TIME_FROM_END_OF_SESSION \
                                                      else timestamp_session_mins
        
        beg_times = np.maximum(comparison_timestamp + WORKING_TIME_BEGIN_DELTA, timestamp_session_mins)
        
        all_extracted_working_times = []

        for session_label, full_session in df_groupby_session:
            if (THROW_AWAY_TOO_SHORT_SESSIONS and 
                (full_session.timestamp.max() - full_session.timestamp.min()) < WORKING_TIME_LENGTH):
                continue
            session_beg_time = beg_times.loc[session_label]
            working_time = extract_time_segment_from_session(full_session, session_beg_time, WORKING_TIME_LENGTH)
            all_extracted_working_times.append(working_time)

        self.df = pd.concat(all_extracted_working_times, ignore_index=True).copy()

        self.verbose_print("Speech turns df shape:", self.df.shape)

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
        self.df = self.df.copy()

        self.verbose_print("Group sizes by tag:")
        self.verbose_print(self.df[['filterindex', 'tag']].reset_index().groupby('filterindex').max().groupby('tag').count().iloc[:, 0])
        self.verbose_print("Speech turns df shape:", self.df.shape)
    
    def _unify_sessions(self):
        """
        Unifies speech turns into a single dataframe row for each session, with the full treatment text.
        """
        self.verbose_print("Unifying session speech turns...")

        self.df = self.df[self.df.dialog_turn_main_speaker.isin(set(DIALOG_TURN_TO_TEMPLATE))]
        self.df = self.df.copy()

        self.df['eng_event_plaintext_fmt'] = self.df.apply(
            lambda row: DIALOG_TURN_TO_TEMPLATE[row["dialog_turn_main_speaker"]].format(text=row["eng_event_plaintext"]),
            axis=1
        )

        sessions_text = self.df.groupby(SESSION_GROUP_COLUMNS)['eng_event_plaintext_fmt'].agg(SPEECH_TURNS_DELIMITER.join)
        sessions_text = SESSION_PREFIX + sessions_text + SESSION_POSTFIX
        sessions = sessions_text.reset_index().rename(columns={'eng_event_plaintext_fmt': 'eng_session_plaintext'})

        self.final_speech_turns_df = self.df.copy()
        self.df = sessions

        self.verbose_print("Sessions df columns:", self.df.columns)
        self.verbose_print("Sessions df shape:", self.df.shape)
        word_amounts = self.df.eng_session_plaintext.apply(lambda x: len(x.split())).to_numpy()
        self.verbose_print("Words in each session: "
                           f"Min: {round(word_amounts.min(), 2)} | "
                           f"Avg: {round(word_amounts.mean(), 2)} | "
                           f"Std: {round(word_amounts.std(), 2)} | "
                           f"Max: {round(word_amounts.max(), 2)}")
    
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

        self.verbose_print(f"Sessions in each set: "
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


def extract_time_segment_from_session(full_session, beg_ts, segment_delta):
    """
    Determine if to keep the speech turns happening during beg_ts and beg_ts + segment_delta
    This is determined by the closest combination having the length 'segment_delta'.
    """
    # Validations
    session_label = full_session.iloc[0].filterindex
    assert not full_session.empty, f"Session {session_label} has no speech turns"

    if not full_session.timestamp.is_monotonic_increasing:
        raise ValueError(f"Timestamps in session {session_label} are not monotonically increasing")

    if beg_ts < full_session.iloc[0].timestamp:
        raise ValueError(f"Given segment begin time '{beg_ts}' "
                         f"is before the beginning of the session ({full_session.iloc[0].timestamp}) "
                         f"for session {session_label}")
    
    if beg_ts > full_session.iloc[-1].timestamp:
        raise ValueError(f"Given segment begin time '{beg_ts}' "
                         f"is after the last speech turn of the session ({full_session.iloc[-1].timestamp}) "
                         f"for session {session_label}")
    
    # Get the options for beginning the extracted segment
    before = full_session[full_session.timestamp <= beg_ts]
    turn_during_begin_idx = max(before.shape[0] - 1, 0)
    turn_after_begin_idx = min(turn_during_begin_idx + 1, full_session.shape[0] - 1)

    begin_idx_options = set()
    for idx_option in (turn_during_begin_idx, turn_after_begin_idx):
        timestamp = full_session.iloc[idx_option].timestamp
        all_idxs_with_this_timestamp = np.where(full_session.timestamp == timestamp)[0]
        begin_idx_options.add(min(all_idxs_with_this_timestamp))
    assert begin_idx_options, f"No begin turn options for session {session_label}"
    
    # Get the options for ending the extracted segment
    up_to_end = full_session[full_session.timestamp <= (beg_ts + segment_delta)]
    turn_during_end_idx = max(up_to_end.shape[0] - 1, 0)
    turn_before_end_idx = max(turn_during_end_idx - 1, 0)

    end_idx_options = set()
    for idx_option in (turn_during_end_idx, turn_before_end_idx):
        timestamp = full_session.iloc[idx_option].timestamp
        all_idxs_with_this_timestamp = np.where(full_session.timestamp == timestamp)[0]
        end_idx_options.add(max(all_idxs_with_this_timestamp))
    assert end_idx_options, f"No end turn options for session {session_label}"
    
    # Get best combination
    best_delta_diff = pd.Timedelta(days=1)
    best_begin_idx, best_end_idx = None, None
    for begin_idx in begin_idx_options:
        for end_idx in end_idx_options:
            begin_timestamp = full_session.iloc[begin_idx].timestamp
            if end_idx + 1 < full_session.shape[0]:
                end_timestamp = full_session.iloc[end_idx + 1].timestamp
            else:  # approximate the end timestamp
                avg_turn_length = (full_session.iloc[1:].timestamp - full_session.iloc[:-1].timestamp).mean()
                end_timestamp = full_session.iloc[end_idx].timestamp + avg_turn_length
            delta = end_timestamp - begin_timestamp
            delta_diff = abs(delta - segment_delta)
            if delta_diff < best_delta_diff:
                best_delta_diff = delta_diff
                best_begin_idx = begin_idx
                best_end_idx = end_idx
    
    return full_session.iloc[best_begin_idx:best_end_idx + 1]


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
        train, test, val = Preprocess().run()
        print("Dataframe shape:", df.shape)
        # df.to_csv(SAVE_FILE)
        # segment_to_10_mins(df)
    finally:
        end = time.time()
        print(f"Run took {end - start} seconds")
