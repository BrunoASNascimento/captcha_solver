import pandas as pd
import os

df = pd.read_csv('data/CAPTCHA_images/train_labels.csv')
df.sort_values(by=['filename', 'xmin'], inplace=True)

df['text'] = df.groupby(['filename'])[
    'class'].transform(lambda x: ''.join(x))
df_edit = df[['filename', 'text']].drop_duplicates()

for row in df_edit.itertuples(index=False):
    print(f'{row[0]} -> {row[1]}')
    try:
        os.rename(
            f'data/CAPTCHA_images/train/{row[0]}',
            f'data/CAPTCHA_images/img/{row[1]}.jpg'
        )
    except Exception as error:
        print(error)
