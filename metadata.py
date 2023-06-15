import os
import pandas as pd


def create_fam_hx_metadata(filename, metadata_dir, outname='family_history.csv'):
    metadata = pd.read_csv(os.path.join(metadata_dir, filename))
    fx_metadata = metadata[['isic_id', 'family_hx_mm']]
    fx_metadata['isic_id'] = fx_metadata['isic_id'].astype(str) + '.JPG'
    fx_metadata['family_hx_mm'] = fx_metadata['family_hx_mm'].astype(int)
    with open(os.path.join(metadata_dir, outname), 'w', encoding='utf-8') as f:
        fx_metadata.to_csv(f)


if __name__ == '__main__':
    create_fam_hx_metadata('metadata_combined.csv', '/home/bay1989/masterarbeit/data/ISIC')

    create_metadata = False
    if create_metadata:
        download_cmd_fx_true = "isic metadata download --search 'family_hx_mm:true' > metadata_fx_true.csv"
        download_cmd_fx_false = "isic metadata download --search 'family_hx_mm:false' > metadata_fx_false.csv"

        df1 = pd.read_csv(os.path.join('data/ISIC', 'metadata_fx_false.csv'))
        df2 = pd.read_csv(os.path.join('data/ISIC', 'metadata_fx_true.csv'))

        # print(df['diagnosis'].tolist())

        # drop this column as it interferes with merging the two files
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