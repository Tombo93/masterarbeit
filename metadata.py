import os
import pandas as pd


df1 = pd.read_csv(os.path.join('data/ISIC', 'metadata_fx_false.csv'))
df2 = pd.read_csv(os.path.join('data/ISIC', 'metadata_fx_true.csv'))

# print(df['diagnosis'].tolist())

df2.drop(columns=['mel_mitotic_index'])
out = pd.concat([df1, df2])
# with open('data/ISIC/metadata_combined.csv', 'w', encoding='utf-8') as f:
#     out.to_csv(f)

print(pd.Series(out['isic_id'].tolist()).dropna().is_unique)
print(out.benign_malignant.unique())
print(len(out.benign_malignant.unique()))
print(out['benign_malignant'].value_counts())
print('############')
print(out['diagnosis'].value_counts())
print('############')
print(out['mel_class'].value_counts())
print('############')
print(out['nevus_type'].value_counts())
print('############')
print(out['anatom_site_general'].value_counts())
print('############')