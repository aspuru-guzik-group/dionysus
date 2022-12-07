import glob, os
import pandas as pd
from rdkit.Chem import PandasTools

fnames = glob.glob('*.sdf')
name_tags = pd.Series(fnames).str.split('_').str.get(1)
for i, fname in enumerate(fnames):
    outname = f'opera_{name_tags[i]}.csv'
    df = PandasTools.LoadSDF(fname, molColName=None)
    if os.path.exists(outname):
        old_df = pd.read_csv(outname)
        df = pd.concat([old_df, df])
        print(f'{outname}: number of molecules {len(df)}')
    df.to_csv(outname, index=False)

import pdb; pdb.set_trace()