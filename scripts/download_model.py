import os
from googledrivedownloader import download_file_from_google_drive

# https://drive.google.com/file/d/1eXyI9LD0V0SM1p_4obv--ykIkQd9sjr_/view?usp=sharing
file_id = '1eXyI9LD0V0SM1p_4obv--ykIkQd9sjr_'
file_name = 'checkpoints.zip'
chpt_path = './'
if not os.path.isdir(chpt_path):
	os.makedirs(chpt_path)

destination = os.path.join(chpt_path, file_name)
download_file_from_google_drive(file_id=file_id,
                                    dest_path=destination,
                                    unzip=True)
