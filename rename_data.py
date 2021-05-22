import shutil
import os

list_files = os.listdir(
    '/media/bruno-nascimento/SSD/developer/data/captcha/4_letters')
for name_file in list_files:
    if '_' in name_file:
        print(name_file)
        try:
            shutil.move(
                f"/media/bruno-nascimento/SSD/developer/data/captcha/4_letters/{name_file}",
                f"/media/bruno-nascimento/SSD/developer/data/captcha/4_letters/{name_file.split('_')[1]}"
            )
        except:
            pass
