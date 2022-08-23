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
# under the License


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    
import json
import math
import torch
import numpy
import pandas
import argparse

import scikit_wrappers
import utils
import pandas as pd

print(torch.cuda.device_count())

def fit_hyperparameters(file, path_csv_train, cuda, gpu, save_path,
                        save_memory=False):
    """
    Creates a classifier from the given set of hyperparameters in the input
    file, fits it and return it.

    @param file Path of a file containing a set of hyperparemeters.
    @param train Training set.
    @param train_labels Labels for the training set.
    @param cuda If True, enables computations on the GPU.
    @param gpu GPU to use if CUDA is enabled.
    @param save_memory If True, save GPU memory by propagating gradients after
           each loss term, instead of doing it after computing the whole loss.
    """
    classifier = scikit_wrappers.CausalCNNEncoderClassifier()

    # Loads a given set of hyperparameters and fits a model with those
    hf = open(os.path.join(file), 'r')
    params = json.load(hf)
    hf.close()
    # Check the number of input channels
    train_torch_dataset = utils.Dataset(path_csv_train)
    params['in_channels'] = train_torch_dataset[0]["mel_spec_db"].shape[0] #num of mel bands
    params['cuda'] = cuda
    params['gpu'] = gpu
    load = False
    params['save_path'] = save_path
    
    classifier.set_params(**params, load=load)
    with open(save_path + 'hyperparameters.json', 'w') as fp:
        json.dump(params, fp)
    return classifier.fit(
        path_csv_train, save_memory=save_memory, verbose=True
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Classification tests for UCR repository datasets'
    )


    parser.add_argument('--save_path', type=str, metavar='PATH', default="/home/dsi/moradim/UnsupervisedScalableRepresentationLearningTimeSeries/another_run_20/",
                        help='path where the estimator is/should be saved')
    parser.add_argument('--cuda', type=bool, default=1,
                        help='activate to use CUDA')
    parser.add_argument('--gpu', type=int, default=0, metavar='GPU',
                        help='index of GPU used for computations (default: 0)')
    parser.add_argument('--hyper', type=str, metavar='FILE', default="/home/dsi/moradim/UnsupervisedScalableRepresentationLearningTimeSeries/default_hyperparameters.json",
                        help='path of the file of hyperparameters to use; ' +
                             'for training; must be a JSON file')
    parser.add_argument('--load', action='store_true', default=True,
                        help='activate to load the estimator instead of ' +
                             'training it')
    parser.add_argument('--fit_classifier', action='store_true', default=False,
                        help='if not supervised, activate to load the ' +
                             'model and retrain the classifier')

    return parser.parse_args()


if __name__ == '__main__':
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print(device)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    args = parse_arguments()
    #print(args.load)
    if args.cuda and not torch.cuda.is_available():
        print("CUDA is not available, proceeding without it...")
        args.cuda = False

    path_csv_train = "/home/dsi/moradim/UnsupervisedScalableRepresentationLearningTimeSeries/Train.csv"
    path_csv_test = "/home/dsi/moradim/UnsupervisedScalableRepresentationLearningTimeSeries/data_wsj.csv" #/home/dsi/moradim/UnsupervisedScalableRepresentationLearningTimeSeries/Test.csv"
    
    if not args.load and not args.fit_classifier:
        classifier = fit_hyperparameters(
            args.hyper, path_csv_train, args.cuda, args.gpu, args.save_path
        )
    else:
        classifier = scikit_wrappers.CausalCNNEncoderClassifier()
        hf = open(
            os.path.join(
                args.save_path, 'hyperparameters.json'
            ), 'r'
        )
        hp_dict = json.load(hf)
        hf.close()
        hp_dict['cuda'] = args.cuda
        hp_dict['gpu'] = args.gpu
        load = True
        hp_dict['save_path'] = args.save_path
        classifier.set_params(**hp_dict, load=load)
        classifier.load(os.path.join(args.save_path))

    if not args.load:
        if args.fit_classifier:
            y_train = pd.read_csv(path_csv_train)["spk_gender"]
            classifier.fit_classifier(classifier.encode(path_csv_train), y_train)
        classifier.save(
            os.path.join(args.save_path)
        )
        with open(
            os.path.join(
                args.save_path, 'libri_hyperparameters.json'
            ), 'w'
        ) as fp:
            json.dump(classifier.get_params(), fp)
    print("Start test")
    y_test = pd.read_csv(path_csv_test)["spk_gender"]
    print("Test accuracy: " + str(classifier.score(path_csv_test, y_test)))
