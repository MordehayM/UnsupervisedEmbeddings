# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import numpy
import torch.utils.data
from scipy.io import wavfile
import librosa
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for a numpy dataset.

    @param dataset Numpy array representing the dataset.
    """
    def __init__(self, path_csv, nb_random_sample=10):
        self.dataset_df = pd.read_csv(path_csv)
        self.fs = 16000
        self.duraion = 5
        self.nb_random_sample = nb_random_sample

    def __len__(self):
        return self.dataset_df.shape[0]

    def __getitem__(self, index):
        path = self.dataset_df.loc[index, "path_file"]
        sample_rate, audio = wavfile.read(path)
        audio = numpy.float64(audio)
        audio = 2*(audio - audio.min()) / (audio.max() - audio.min()) - 1
        audio = audio[:self.duraion * self.fs]
        #print(audio.dtype)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=512, hop_length=256, win_length=512, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec)
        
        spk_id = self.dataset_df.loc[index, "spk_id"]
        
        indexes_diff = self.dataset_df.index[self.dataset_df["spk_id"] != spk_id].to_list()
        neg_samples_index = numpy.random.choice(indexes_diff, size=self.nb_random_sample)
        ##another positive sample
        indexes_same = self.dataset_df.index[self.dataset_df["spk_id"] == spk_id].to_list()
        another_positive_index = numpy.random.choice(indexes_same, size=None)
        """path = self.dataset_df.loc[another_positive_index, "path_file"]
        sample_rate, audio = wavfile.read(path)
        audio = numpy.float64(audio)
        audio = 2*(audio - audio.min()) / (audio.max() - audio.min()) - 1
        audio = audio[:self.duraion * self.fs]
        #print(audio.dtype)
        mel_spec_another_positive = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=512, hop_length=256, win_length=512, n_mels=128)
        mel_spec_db_another_positive = librosa.power_to_db(mel_spec_another_positive)"""
        sample = {"spk_id": spk_id, "mel_spec_db": torch.tensor(mel_spec_db), "neg_samples_index": neg_samples_index,
                  "another_positive_index":another_positive_index}
        
        return sample
        
        
        
        


class LabelledDataset(torch.utils.data.Dataset):
    """
    PyTorch wrapper for a numpy dataset and its associated labels.

    @param dataset Numpy array representing the dataset.
    @param labels One-dimensional array of the same length as dataset with
           non-negative int values.
    """
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return numpy.shape(self.dataset)[0]

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]
