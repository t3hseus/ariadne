import multiprocessing
import os
import sys
import datetime

sys.path.append(os.path.abspath('./'))
from ariadne_v2 import jit_cacher

def confirm():
    """
    Ask user to enter Y or N (case-insensitive).
    :return: True if the answer is Y.
    :rtype: bool
    """
    answer = ""
    while answer not in ["y", "n"]:
        answer = input("OK with that [Y/N]? ").lower()
    return answer == "y"


def clean_jit_cache(str_to_clean):
    global cacher, date
    global_lock = multiprocessing.Lock()
    jit_cacher.init_locks(global_lock)
    with jit_cacher.instance() as cacher:
        cacher.init()
    df = cacher.cache

    time_to_clean = str_to_clean[:-1]
    time = int(time_to_clean)
    clear_from = None
    if str_to_clean[-1] == 'w':
        clear_from = datetime.datetime.now() - datetime.timedelta(weeks=time)
    if str_to_clean[-1] == 'm':
        clear_from = datetime.datetime.now() - datetime.timedelta(minutes=time)
    if str_to_clean[-1] == 'd':
        clear_from = datetime.datetime.now() - datetime.timedelta(days=time)
    if str_to_clean[-1] == 'h':
        clear_from = datetime.datetime.now() - datetime.timedelta(hours=time)
    assert clear_from is not None, 'provide time to clean cache in format like "1w"(one week) or "4m" (4 minutes) or ' \
                                   '"3h" (three hours) '
    print(f"warning, you are going to clean all cache for the {str_to_clean} starting from the {clear_from} till now")
    all_times = [datetime.datetime.fromisoformat(date).timestamp() for date in df.date.values]
    removed = []
    for idx, val in enumerate(all_times):
        if val > clear_from.timestamp():
            removed.append(idx)
    if confirm():
        with cacher.handle(cacher.cache_db_path, mode='a') as f:
            for ind in removed:
                row = df.iloc[ind]
                path = row.key
                if path in f:
                    print(f"deleting path {path}")
                    del f[path]


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'provide time to clean cache in format like "1w"(one week) or "4m" (4 minutes) or "3h" ' \
                               '(three hours) '
    clean_jit_cache(sys.argv[1])
