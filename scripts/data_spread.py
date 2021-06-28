
import re

import numpy as np

from cuticle_analysis.dataset import Dataset

data = Dataset((16, 16), dataset_type='rough_smooth', rebuild=True)

# convert human expert labels using majority voting
label_cols = data.ant_data[['Jp', 'Becca', 'Katy']].mode(axis=1)[0]
label_cols = label_cols.fillna('_')

label_cols = label_cols[~label_cols.isin(['_', 'NA ', 'Na'])]
vals, counts = np.unique(label_cols, return_counts=True)

res = {
    'rd': 0,
    'rn': 0,
    'rr': 0,
    'rt': 0,
    'sg': 0,
    'ss': 0,
}

for i in range(len(vals)):
    val = vals[i]
    count = counts[i]

    if re.match(r'^[r][d].*', val):
        res['rd'] += count
    elif re.match(r'^[r][n].*', val):
        res['rn'] += count
    elif re.match(r'^[r][r].*', val):
        res['rr'] += count
    elif re.match(r'^[r][t].*', val):
        res['rt'] += count
    elif re.match(r'^[s][g].*', val):
        res['sg'] += count
    elif re.match(r'^[s][s].*', val):
        res['ss'] += count

for key in res.keys():
    print(f'Code {key}: {res[key]}')
