import pandas as pd
import os

PATH = 'D:/developer/data/captcha/4_letters'
list_files = os.listdir(PATH)
df = pd.DataFrame(list_files, columns=['name_file'])
df[['name_clean', '_']] = df['name_file'].str.split(
    '.', 0, expand=True)
df['name_len'] = df['name_clean'].str.len()
print(df.sample(100))

for name_file in df.sample(137)['name_file'].values:
    print(name_file)
    os.remove(f'{PATH}/{name_file}')
