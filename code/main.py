"""
Lorsque vos algorithmes sont plus ou moins capables de re-générer les caractères appris de la base Binary
AlphaDigit, on se focalisera sur l’analyse finale qui s’effectuera sur la base MNIST. Voici un plan pour vous
guider dans la construction de votre programme principal:
    - Spécifier les paramètres liés au réseau et à l’apprentissage : taille du réseau (vecteur contenant le
nombre de neurones), nombre d’itérations pour les descentes de gradient (100 pour les RBM, 200 pour
    - l’algorithme de rétro-propagation du gradient), learning rate (ex : 0.1), taille des mini-batch, le 
    nombre de données d’apprentissage, ...
    - Charger les données;
    - Initialisation aléatoire du DNN;
    - Si pré-apprentissage, pré-entraîner de manière non supervisée le DNN;
    - Entraîner de manière supervisé le DNN préalablement pré-entrainé via l’algorithme de rétro-propagation
du gradient.
    - Avec le réseau appris, observer les probabilités de sortie de quelques images de la base d’apprentissage.
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


TEST_SUBJECTS = ["Number of layers", "Number of hidden units", "Training data size"]

def init_dnns(dbn_size):
    """
    This function initializes two DNN_struc (see documentation in 
    principal_DNN_MNIST) instances that are identical. Used to see effect of 
    pre-training.
    ## Arguments:
        - dbn_size: list of int32 of length the depth of the DNN to be created and
        containing number of units for each layer
    ## Returns:
        - tuple of DNN_Struct instances, of length 2
    """
    dnn_struct_1 = DNNStruct(dbn_size)
    dnn_struct_2 = DNNStruct(dbn_size)
    # To ensure we have identical networks
    for i, param in enumerate(dnn_struct_1.parameters):
        dnn_struct_2.update_layer(*param, i)

    return dnn_struct_1, dnn_struct_2

def add_same_layer(dnn_1, dnn_2, size):
    """
    Adds the same randomly initialized layer to two DNNStruct instances.
    ## Arguments:
        - dnn_1: DNNStruct instance
        - dnn_2: DNNStruct instance
        - size: int representing the number of units in the new layer
    ## Returns:
        - None
    """
    dnn_1.stack_layer(n_classes)
    dnn_2.stack_layer(n_classes)
    dnn_2.update_layer(*dnn_1.parameters[-1], len(dnn_2.parameters) - 1)

def get_errors(
        dbn_size,
        pretrain_set,
        train_sets,
        test_sets,
        train_error,
        test_error,
        train_params=(10, 64, 0.01),
        pretrain_params=(10, 64, 0.01)
        ):
    """
    Intializes two identical DNNStruct instances, with one to pretrain and not the other.
    Performs backward-propagation for training and gets accuracy on test sets.
    ## Arguments:
        - dbn_size : list of int, length representing the depth, and elements representing
          the number of hidden units for each layer
        - pretrain_params : the parameters of the pretrain step. Elements are as following
          * int : number of epochs (default=10)
          * int : the batch size (default=64)
          * float : learning rate (default=0.01)
        - pretrain_set: numpy.ndarray, the set used for pretraining
        - train_params: the parameters of the training the networks. Elements are as following
          * int : number of epochs (default=10)
          * int : the batch size (default=64)
          * float : learning rate (default=0.01)
        - train_error : list of list, where to store train_error of the models
        - test_error : list of list, where to store test_error of the models
        - train_sets : tuple of 3 numpy.ndarray, representing features and targets and one-hot
        encoded targets for training
        - test_sets: tuple of 2 numpy.ndarray, representing features and targets for testing
    ## Returns:
        - None
    """
    # Retrieving the elements passed as arguments
    X_train, y_train, train_labels = train_sets
    X_test, y_test = test_sets
    X_pretrain = pretrain_set

    n_classes = len(np.unique(y_test))

    n_epochs_pretrain, batch_size_pretrain, lr_pretrain = pretrain_params
    n_epochs, batch_size, lr = train_params

    dnn_pretrain, dnn_no_pretrain = init_dnns(dbn_size)
    dnn_pretrain = pretrain_DNN(
        X_pretrain,
        dnn_pretrain,
        n_epochs=n_epochs_pretrain,
        lr=lr_pretrain,
        batch_size=batch_size_pretrain)

    # Add identical output layers
    add_same_layer(dnn_pretrain, dnn_no_pretrain, n_classes)


    dnn_pretrain = retropropagation(
        X_train,
        train_labels,
        dnn_pretrain,
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size
    )
    dnn_no_pretrain = retropropagation(
        X_train,
        train_labels,
        dnn_no_pretrain,
        lr=lr,
        n_epochs=n_epochs,
        batch_size=batch_size
    )


    # Get error of each model

    ## On train set
    train_error[0].append(1 - test_DNN(dnn_pretrain, X_train, y_train))
    train_error[1].append(1 - test_DNN(dnn_no_pretrain, X_train, y_train))

    ## On test set
    test_error[0].append(1 - test_DNN(dnn_pretrain, X_test, y_test))
    test_error[1].append(1 - test_DNN(dnn_no_pretrain, X_test, y_test))


if __name__ == "__main__":

    idx = -1
    attempts = 0
    while not (isinstance(idx, int) and idx in [0,1,2]):
        if attempts > 0:
            print("Invalid value given. Try again")
        attempts += 1
        displayed = "Which parameters to explore? \n1: " + TEST_SUBJECTS[0] 
        displayed += "\n2: " + TEST_SUBJECTS[1]
        displayed += "\n3: " + TEST_SUBJECTS[2] + "\n"
        idx = input(displayed)
        try:
            idx = int(idx) - 1
        except:
            continue

    # Reference values to be fixed when other ones will change in the test phases
    REF_N_HIDDEN_LAYERS = 2
    REF_N_HIDDEN_UNITS = 200
    REF_DATA_SIZE = 60000

    # Values to be explored
    N_HIDDEN_LAYERS = [2, 3, 4, 5, 6, 8, 10]
    N_HIDDEN_UNITS = [100, 200, 300, 500, 700]
    DATA_SIZE = [1000, 2000, 3000, 5000, 7000, 10000, 20000, 30000, 60000]

    # Mapping idx to list of parameters' values
    MAPPING = {
        0 : N_HIDDEN_LAYERS,
        1 : N_HIDDEN_UNITS,
        2 : DATA_SIZE
    }

    # Pretraining paramters
    n_epochs_pretrain = 10
    batch_size_pretrain = 64
    lr_pretrain = 0.01

    PRETRAIN_DIGITS = (0,1,2,3,4,5,6,7,8,9)
    assert all([0 <= x <= 9 for x in PRETRAIN_DIGITS]), "only one-character digits, between 0 and 9"
    # file name if you want to use a pretrained model ; should be set to None otherwise
    PRETRAIN_FILE = None #"pretrain_2.pkl"

    # Train parametersz

    n_epochs = 10
    batch_size = 64
    lr = 0.01

    # whether to store the data from experiment
    MONITOR = True
    # where to store data from experiment ; will only be used if MONITOR is True
    RESULTS_OUT = "experiment.txt"

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

    print("Assessing ", TEST_SUBJECTS[idx].lower(), " influence on accuracy")

    train_error = [[], []]
    test_error = [[], []]
    if idx == 0:

        for depth in N_HIDDEN_LAYERS:
            dbn_size = [input_dim]
            for _ in range(depth):
                dbn_size.append(200)

            get_errors(
                dbn_size,
                pretrain_set=X_pretrain,
                train_sets=(X_train, y_train, train_labels),
                test_sets=(X_test, y_test),
                train_error=train_error,
                test_error=test_error,
                train_params=(n_epochs, batch_size, lr),
                pretrain_params=(n_epochs_pretrain, batch_size_pretrain, lr_pretrain)
            )
            

    elif idx == 1:

        for width in N_HIDDEN_UNITS:
            dbn_size = [input_dim] + [width] * REF_N_HIDDEN_LAYERS

            get_errors(
                dbn_size,
                pretrain_set=X_pretrain,
                train_sets=(X_train, y_train, train_labels),
                test_sets=(X_test, y_test),
                train_error=train_error,
                test_error=test_error,
                train_params=(n_epochs, batch_size, lr),
                pretrain_params=(n_epochs_pretrain, batch_size_pretrain, lr_pretrain)
            )

    else:
        dbn_size = [input_dim] + [REF_N_HIDDEN_UNITS] * REF_N_HIDDEN_LAYERS
        for size in DATA_SIZE:
            size = min(size, X_train.shape[0])
            indices = np.random.choice(np.arange(X_train.shape[0]), size, replace=False)

            get_errors(
                dbn_size,
                pretrain_set=X_pretrain,
                train_sets=(X_train[indices], y_train[indices], train_labels[indices]),
                test_sets=(X_test, y_test),
                train_error=train_error,
                test_error=test_error,
                train_params=(n_epochs, batch_size, lr),
                pretrain_params=(n_epochs_pretrain, batch_size_pretrain, lr_pretrain)
            )


    # Plots
    IMAGES_FOLDER = 'images'
    FIG_SAVE_FOLDER = os.path.join(IMAGES_FOLDER, 'plots')
    if not os.path.exists(IMAGES_FOLDER):
        os.mkdir(IMAGES_FOLDER)
    if not os.path.exists(FIG_SAVE_FOLDER):
        os.mkdir(FIG_SAVE_FOLDER)

    fig = plt.figure(figsize=(20,10))
    plt.plot(MAPPING[idx], train_error[0], '.-', label="Pretrained model", color="green")
    plt.plot(MAPPING[idx], train_error[1], '.-', label="No pretraining", color="red")
    title = "Influence of " + TEST_SUBJECTS[idx].lower() + " on training"
    plt.xlabel(TEST_SUBJECTS[idx])
    plt.ylabel("Error")
    plt.title(title)
    plt.legend(loc="best")
    figname = os.path.join(FIG_SAVE_FOLDER, title.replace(" ", "_")+".jpeg")
    plt.savefig(figname)
    plt.show()


    fig = plt.figure(figsize=(20,10))
    plt.plot(MAPPING[idx], test_error[0], '.-', label="Pretrained model", color="green")
    plt.plot(MAPPING[idx], test_error[1], '.-', label="No pretraining", color="red")
    title = "Influence of " + TEST_SUBJECTS[idx].lower() + " on testing"
    plt.xlabel(TEST_SUBJECTS[idx])
    plt.ylabel("Error")
    plt.title(title)
    plt.legend(loc="best")
    figname = os.path.join(FIG_SAVE_FOLDER, title.replace(" ", "_")+".jpeg")
    plt.savefig(figname)
    plt.show()