import urllib.request
from tqdm import tqdm
import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

control = 0
for _ in tqdm(range(200)):
    urllib.request.urlretrieve(
        os.environ.get('LINK_CAPTCHA'),
        f"D:/developer/data/captcha/4_letters/{str(control).zfill(8)}.png"
    )
    control += 1
