import os
import sys
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import json

def index_words(list_of_words):
    x = enumerate(list_of_words)
    d = {k:v for (v,k) in x}
    return d

def get_files(folder_path):
    return [f for f in os.listdir(folder_path) if isfile(join(folder_path, f))]

def get_label_names(list_of_files):

    def process_label_name(label):
        out = label.replace(".txt","")
        out = out.split("_")
        return out[0]

    y = [process_label_name(name) for name in list_of_files]
    print(y)
    return np.array(y)

def get_features(list_of_files, folder_path=""):
    x = []

    for file in list_of_files:
        with open (os.path.join(folder_path,file), "r") as myfile:
            data=myfile.readlines()
        data=eval(data[0])
        data=data[:125]
        x.append(data)
        data = np.array(data)
        # print(data.shape)

    x = np.array(x)
    # print(len(list_of_files))
    # print(x.shape)
    return x

def make_dataset(folder_path=""):
    list_of_files = get_files(folder_path)
    labels = get_label_names(list_of_files)
    features = get_features(list_of_files,folder_path)
    dataset = [features]
    return dataset
    # np.save("eeg_data_100_words",dataset)

def plot_data(data):
    plt.plot(data)
    plt.ylabel('some numbers')
    plt.show()

# a = get_files("./raw_data")
# print(a)
# print(get_label_names(a))
# print(get_features(a, folder_path="./raw_data"))
# print(index_words(get_label_names(a)))

dataset = make_dataset(folder_path="./raw_data")
print(dataset)
np.save("eeg_data_100_words.npy",dataset)
d = np.load("eeg_data_100_words.npy",allow_pickle=True)
print(d)
# print(index_words(get_label_names(get_label_names(get_files("./raw_data")))))
labels = index_words(get_label_names(get_label_names(get_files("./raw_data"))))
with open('labels.json','w') as fp:
    json.dump(labels,fp)