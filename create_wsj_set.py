import pandas as pd
import numpy as np
import glob
from pathlib import Path
from joblib import Parallel, delayed
import joblib 
from scipy.io import wavfile
from functools import partial

def id_spk_from_path(path):
    separated_path = path.split('/')
    spk_id = separated_path[-2]
    return spk_id.lower()

def check_duration(path):
    """
    return : the path with duration greater than sec_require seconds
    """
    #print(path)
    sample_rate, data = wavfile.read(path)
  
    len_data = len(data)  # holds length of the numpy array
    sec_len = len_data / sample_rate
  
    return sec_len

s_path_wsj_trainAndtest = glob.glob("/dsi/gannot-lab/datasets/sharon_db/wsj0/**/**.wav", recursive=True)
a = len(list(glob.glob("/dsi/gannot-lab/datasets/sharon_db/wsj0/Train/*"))) +len(list(glob.glob("/dsi/gannot-lab/datasets/sharon_db/wsj0/Test/*")))
print(a)

df_train = pd.DataFrame()
df_train["path_file"] = list(s_path_wsj_trainAndtest)


#add speaker id column
df_train["spk_id"] = list(map(id_spk_from_path, list(s_path_wsj_trainAndtest)))


#add duration of audio in sec
df_train["duration_sec"] = list(map(check_duration, list(s_path_wsj_trainAndtest)))


##add gender information
with open("Gender_wsj.txt", "r") as file_gender:
    list_gender_speaker = file_gender.read().splitlines()
    gender_speaker_dict = dict(list(map(lambda sp_ge: tuple([sp_ge.split(',')[0].lower(), sp_ge.split(',')[1]]), list_gender_speaker)))


##add speaker gender

df_train["spk_gender"] = list(map(lambda spk_id: gender_speaker_dict[spk_id], df_train["spk_id"]))#zero for female and one for male
df_train["spk_gender"] = df_train["spk_gender"] .replace(to_replace=['F', 'M'], value=[0, 1])


require_sec = 5
#remove audio that their duraion audio us less tham require_sec seconds
df_train = df_train.drop(df_train[df_train['duration_sec'] < require_sec].index, inplace = False)



print(sum(df_train["spk_gender"]==0))
print(sum(df_train["spk_gender"]==1))
df_train.to_csv("data_wsj.csv")
