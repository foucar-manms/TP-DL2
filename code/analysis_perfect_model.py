"""
Ce fichier permet de préentraîner, entraîner et tester un modèle DNN
construit comme un empilement de RBM.
"""

import sys
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from tqdm import tqdm
from mlxtend.data import loadlocal_mnist

from utils import cross_entropy, vec2img, monitor_experience
from principal_RBM_alpha import entree_sortie_RBM, lire_alpha_digit
from principal_DBN_alpha import DNNStruct, generer_image_DBN, pretrain_DNN, PATH_TO_STRUCTURES
from principal_DNN_MNIST import test_DNN, retropropagation, PATH_TO_DATA, PATH_TO_DNN, PATH_TO_CONFIGS, DATA_FILES


if __name__ == "__main__":

    # Reference parameters for the model
    REF_N_HIDDEN_LAYERS = 2
    REF_N_HIDDEN_UNITS = 700
    REF_DATA_SIZE = 60000

    # Pretraining paramters
    n_epochs_pretrain = 10
    batch_size_pretrain = 64
    lr_pretrain = 0.01

    PRETRAIN_DIGITS = (0,1,2,3,4,5,6,7,8,9)
    assert all([0 <= x <= 9 for x in PRETRAIN_DIGITS]), "only one-character digits, between 0 and 9"

    # Train parametersz

    n_epochs = 10
    batch_size = 64
    lr = 0.01

    # load data
    if not os.path.exists(PATH_TO_DATA):
        os.mkdir(PATH_TO_DATA)
    for f in DATA_FILES:
        assert os.path.isfile(os.path.join(PATH_TO_DATA, f)), (
            "the file {} does not exist in the {}\\ folder, make sure you have "
            "a {}\\ folder with {} in it (you may download it at "
            "http://yann.lecun.com/exdb/mnist/). You may also want "
            "to check that the file name match yours".format(f, PATH_TO_DATA, PATH_TO_DATA, f)
            )
    ###########

    X_train, y_train = loadlocal_mnist(
        images_path=os.path.join('data', 'train-images.idx3-ubyte'), 
        labels_path=os.path.join('data' ,'train-labels.idx1-ubyte')
    )
    X_test, y_test = loadlocal_mnist(
        images_path=os.path.join('data', 't10k-images.idx3-ubyte'), 
        labels_path=os.path.join('data', 't10k-labels.idx1-ubyte')
    )
    n_classes = len(np.unique(y_test))

    train_labels = np.zeros(shape=(len(y_train), n_classes))
    X_pretrain = np.concatenate((X_train, X_test), axis=0)

    # one-hot encoding
    index = []
    for i in range(len(y_train)):
        train_labels[i][y_train[i]] = 1.

    # make the images binary
    for i in range(len(X_pretrain)):
        X_pretrain[i] = (X_pretrain[i] > np.max(X_pretrain[i] / 2)) * 1.

    input_dim = X_train.shape[1]
    dbn_size = [input_dim] + [REF_N_HIDDEN_UNITS] * REF_N_HIDDEN_LAYERS

    model = DNNStruct(dbn_size)

    print("Pretraining the model...\n")
    model = pretrain_DNN(
        X_pretrain,
        model,
        n_epochs=n_epochs_pretrain,
        lr=lr_pretrain,
        batch_size=batch_size_pretrain)

    model.stack_layer(n_classes)
    train_error = []
    test_error = []
    n_epochs_list = np.arange(1, n_epochs+1)

    for _ in n_epochs_list:
        model = retropropagation(
            X_train,
            train_labels,
            model,
            lr=lr,
            n_epochs=1,
            batch_size=batch_size
        )
        train_error.append(1 - test_DNN(model, X_train, y_train))
        test_error.append(1 - test_DNN(model, X_test, y_test))

    # Plotting
    IMAGES_FOLDER = 'images'
    FIG_SAVE_FOLDER = os.path.join(IMAGES_FOLDER, 'plots')
    if not os.path.exists(IMAGES_FOLDER):
        os.mkdir(IMAGES_FOLDER)
    if not os.path.exists(FIG_SAVE_FOLDER):
        os.mkdir(FIG_SAVE_FOLDER)

    fig = plt.figure(figsize=(20,10))
    plt.plot(n_epochs_list, train_error, '.-', label="Training error", color="blue")
    plt.plot(n_epochs_list, test_error, '.-', label="Testing error", color="red")
    plt.legend(loc="best")
    plt.xlabel("Number of epochs")
    plt.ylabel("Error")
    plt.title("Performance of the model per epoch")
    fname = os.path.join(FIG_SAVE_FOLDER, "optimal_model_restults.jpeg")
    plt.savefig(fname)
    plt.show()

