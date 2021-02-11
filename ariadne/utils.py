import torch
import numpy as np
import itertools
import pandas as pd

def fix_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.deterministic = True #torch.backends.cudnn.determenistic
        torch.benchmark = False #torch.backends.cudnn.benchmark

def cartesian(df1, df2):
    rows = itertools.product(df1.iterrows(), df2.iterrows())
    df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
    df_fakes = df[(df['track_left'] == -1) & (df['track_right'] == -1)]
    df = df[(df['track_left'] != df['track_right'])]
    df = pd.concat([df, df_fakes], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)
    return df.reset_index(drop=True)