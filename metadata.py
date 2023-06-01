import os
import pandas as pd


df1 = pd.read_csv(os.path.join('data/ISIC', 'metadata_fx_false.csv'))
df2 = pd.read_csv(os.path.join('data/ISIC', 'metadata_fx_false.csv'))

# print(df['diagnosis'].tolist())
print(df1.benign_malignant.unique())
print(len(df1.benign_malignant.unique()))
print(df1['benign_malignant'].value_counts())

out = pd.concat([df1, df2])
with open('data/ISIC/metadata_combined.csv', 'w', encoding='utf-8') as f:
    out.to_csv(f)

print(pd.Series(out['isic_id'].tolist()).dropna().is_unique)