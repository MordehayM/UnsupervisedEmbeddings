import pandas as pd
import numpy as np
import glob
from pathlib import Path
from joblib import Parallel, delayed
import joblib 
from scipy.io import wavfile
from functools import partial

def id_spk_from_path(path):
    stem_path = Path(path).stem
    spk_id = stem_path.split('-')[0]
    return int(spk_id)

def check_duration(path):
    """
    return : the path with duration greater than sec_require seconds
    """
    #print(path)
    sample_rate, data = wavfile.read(path)
  
    len_data = len(data)  # holds length of the numpy array
    sec_len = len_data / sample_rate
  
    return sec_len

s_path_libri_train = glob.glob("/dsi/gannot-lab/datasets/LibriSpeech/LibriSpeech/Train/**/**.wav", recursive=True)
s_path_libri_test = glob.glob("/dsi/gannot-lab/datasets/LibriSpeech/LibriSpeech/Test/**/**.wav", recursive=True)

df_train = pd.DataFrame()
df_test = pd.DataFrame()
df_train["path_file"] = list(s_path_libri_train)
df_test["path_file"] = list(s_path_libri_test)

#add speaker id column
df_train["spk_id"] = list(map(id_spk_from_path, list(s_path_libri_train)))
df_test["spk_id"] = list(map(id_spk_from_path, list(s_path_libri_test)))

#add duration of audio in sec
df_train["duration_sec"] = list(map(check_duration, list(s_path_libri_train)))
df_test["duration_sec"] = list(map(check_duration, list(s_path_libri_test)))

##add gender information
with open("Gender_libri.txt", "r") as file_gender:
    list_gender_speaker = file_gender.read().splitlines()
    gender_speaker_dict = dict(list(map(lambda sp_ge: tuple([int(sp_ge.split(',')[0]), int(sp_ge.split(',')[1])]), list_gender_speaker)))


##add speaker gender

df_train["spk_gender"] = list(map(lambda spk_id: gender_speaker_dict[spk_id], df_train["spk_id"]))#zero for female and one for male
df_test["spk_gender"] = list(map(lambda spk_id: gender_speaker_dict[spk_id], df_test["spk_id"])) 


require_sec = 5
#remove audio that their duraion audio us less tham require_sec seconds
df_train = df_train.drop(df_train[df_train['duration_sec'] < require_sec].index, inplace = False)
df_test = df_test.drop(df_test[df_test['duration_sec'] < require_sec].index, inplace = False)


print(sum(df_train["spk_gender"]==0))
print(sum(df_test["spk_gender"]==0))
print(sum(df_train["spk_gender"]==1))
print(sum(df_test["spk_gender"]==1))
df_train.to_csv("Train.csv")
df_test.to_csv("Test.csv")