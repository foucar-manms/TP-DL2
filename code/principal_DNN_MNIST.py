""" On complètera au fur et à mesure un programme principal principal_DNN_MNIST permettant d’apprendre
    un réseau de neurones profonds pré-entrainé par via un DBN
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



PATH_TO_DATA = 'data'
PATH_TO_DNN = "DNN_structures"
PATH_TO_CONFIGS = "configs"

DATA_FILES = [
    "t10k-images.idx3-ubyte",
    "t10k-labels.idx1-ubyte",
    "train-images.idx3-ubyte",
    "train-labels.idx1-ubyte"
]



def calcul_softmax(inputs, rbm_struct):
    """ prend en argument un RBM, des données d’entrée et retourne des probabilités sur les
        unités de sortie à partir de la fonction softmax
    """
    
    _, b, w = rbm_struct()
    if len(inputs.shape) == 1: inputs = np.expand_dims(inputs, axis=0)
    linear = np.dot(inputs, w) + b
    softmax = np.exp(linear) / np.sum(np.exp(linear), axis=1).reshape(-1,1)
    return softmax

def entree_sortie_reseau(inputs, dnn_struct):
    """ prend en argument un DNN, des données en entrée du réseau et retourne dans un tableau
        les sorties sur chaque couche cachées du réseau ainsi que les probabilités sur les
        unités de sortie.
        
        Cette fonction pourra utiliser les fonctions entree_sortie_RBM et calcul_softmax
    """
    
    if len(inputs.shape) == 0: inputs = np.expand_dims(inputs, axis=0)
    out = inputs.copy()
    rbm_struct = dnn_struct.make_rbm(0)
    layers = []
    for i in range(len(dnn_struct.rbm_stack) - 1):
        rbm_struct = dnn_struct.make_rbm(i)
        out = entree_sortie_RBM(out, rbm_struct)
        layers.append(out)
    rbm_struct = dnn_struct.make_rbm(-1)
    out = calcul_softmax(out, rbm_struct)
    layers.append(out)
    return layers

def retropropagation(inputs, labels, dnn_struct, lr, n_epochs, batch_size):
    """ estime les poids/biais du réseau à partir de données labellisées, retourne un DNN et
        prend en argument un DNN, le nombre d’itérations de la descente de gradient, le learning
        rate, la taille du mini-batch, des données d’entrée, leur label,...
        
        On pensera à calculer à la fin de chaque epoch, après la mise à jour des paramètres, la
        valeur de l’entropie croisée que l’on cherche à minimiser afin de s’assurer que l’on
        minimise bien cette entropie
    """
    
    n_examples = len(inputs)
    n_batch = n_examples // batch_size + 1 * (n_examples % batch_size != 0)
    tqdm_dict = {"cross-entropy loss": 0.0}
    for epoch in range(n_epochs):
        
        with tqdm(total=n_batch, unit_scale=True, desc="Epoch : %i/%i" % (epoch+1, n_epochs),
                    ncols=100) as pbar:
            
            indexes = np.random.permutation(n_examples)
            total_loss = 0.0
            
            for i in range(0, n_examples, batch_size):
                batch_indexes = indexes[i:(i+batch_size)]
                batch_in = inputs[batch_indexes]
                targets = labels[batch_indexes]
                # usually equal to batch_size, except for the last batch that may be shorter
                actual_batch_size = len(batch_in)
                
                # forward pass
                layers = [batch_in] + entree_sortie_reseau(batch_in, dnn_struct)
                batch_out = layers[-1]

                # compute the loss
                loss = np.sum(cross_entropy(batch_out, targets)) / actual_batch_size
                total_loss += loss

                # update parameters
                # cross entropy gradient wrt the final layer units before softmax
                delta_1 = batch_out - targets
                w, b = dnn_struct.get_parameters(-1)
                h = layers[-2]
                # cross entropy gradient wrt the previous layer
                delta_2 = np.dot(delta_1, w.T) * h * (1 - h)
                w = w - lr * np.dot(h.T, delta_1) / actual_batch_size
                b = b - lr * np.sum(delta_1, axis=0) / actual_batch_size
                dnn_struct.update_layer(w, b, -1)
                # iterate over other layers
                for j in reversed(range(len(dnn_struct) - 1)):
                    w, b = dnn_struct.get_parameters(j)
                    h = layers[j]
                    delta_2_prev = delta_2.copy()
                    delta_2 = np.dot(delta_2, w.T) * h * (1 - h)
                    w = w - lr * np.dot(h.T, delta_2_prev) / actual_batch_size
                    b = b - lr * np.sum(delta_2_prev, axis=0) / actual_batch_size
                    dnn_struct.update_layer(w, b, j)
                ###########################

                tqdm_dict['cross-entropy loss'] = total_loss / (i+1)
                pbar.set_postfix(tqdm_dict)
                pbar.update(1)
    return dnn_struct

def test_DNN(dnn_struct, X, y):
    """ teste les performances du réseau appris, prend en argument un DNN appris, un jeu de données
        test, et les vrais labels associés
        
        Elle commencera par estimer le label associé à chaque donnée test (on pourra utiliser
        entree_sortie_reseau) puis comparera ces labels estimés aux vrais labels. Elle retournera
        enfin le taux d’erreur
    """
    
    if len(X.shape) == 1: X = np.expand_dims(X, axis=1)
    y_pred = entree_sortie_reseau(X, dnn_struct)[-1]
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = np.sum(y_pred == y) / len(y)
    return accuracy





if __name__ == "__main__":
    
    # use pretrained model ?
    PRETRAIN = False
    # file name to save the pretrained model ; will only be used if PRETRAIN is True
    PRETRAIN_OUT = "dnn_pretrained.pkl"
    SAVE = True
    # file name to save the supervised fully trained model ; will only be used if SAVE is True
    DNN_OUT = "dnn.pkl"
    # digits on which the model is pretrained
    PRETRAIN_DIGITS = (4,5)
    assert all([0 <= x <= 9 for x in PRETRAIN_DIGITS]), "only one-character digits, between 0 and 9"
    # file name if you want to use a pretrained model ; should be set to None otherwise
    PRETRAIN_FILE = "pretrain_2.pkl"
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
        images_path='data\\train-images.idx3-ubyte', 
        labels_path='data\\train-labels.idx1-ubyte'
    )
    X_test, y_test = loadlocal_mnist(
        images_path='data\\t10k-images.idx3-ubyte', 
        labels_path='data\\t10k-labels.idx1-ubyte'
    )
    n_classes = len(np.unique(y_test))

    train_labels = np.zeros(shape=(len(y_train), n_classes))
    X_pretrain = np.zeros(shape=(0, X_train.shape[1]))
    # one-hot encoding
    index = []
    for i in range(len(y_train)):
        train_labels[i][y_train[i]] = 1.
        # extract only chosen data
        if y_train[i] in PRETRAIN_DIGITS:
            index.append(i)
    X_pretrain = X_train[np.array(index)]
    # make the images binary
    for i in range(len(X_pretrain)):
        X_pretrain[i] = (X_pretrain[i] > np.max(X_pretrain[i] / 2)) * 1.
    
    input_dim = X_train.shape[1]
    dbn_size = [input_dim, 256, 128]
    
    # pretrain a dnn
    if PRETRAIN:
        dnn_struct = DNNStruct(dbn_size)
        dnn_struct = pretrain_DNN(X_pretrain, dnn_struct, n_epochs=70, lr=0.01, batch_size=64)
        generer_image_DBN(6, dnn_struct, img_shape='MNIST', n_iter_gibbs=10)
        pkl.dump(dnn_struct,
            open(os.path.join(PATH_TO_STRUCTURES, PRETRAIN_OUT), "wb")
        )
    ################
    
    dnn_size = dbn_size + [n_classes]
    dnn_struct = DNNStruct(dnn_size)
    
    # load a pretrained model set of parameters
    if PRETRAIN_FILE:
        pretrained = pkl.load(
            open(os.path.join(PATH_TO_STRUCTURES, PRETRAIN_FILE), "rb")
        )
        for i in range(len(pretrained.parameters)):
            dnn_struct.update_layer(*pretrained.parameters[i], i)
    ###########################################
    
    dnn_struct = retropropagation(X_train, train_labels, dnn_struct, lr=0.2, n_epochs=10, batch_size=128)
    
    # save model
    if SAVE:
        pkl.dump(dnn_struct,
            open(os.path.join(PATH_TO_STRUCTURES, DNN_OUT), "wb")
        )
    acc = test_DNN(dnn_struct, X_test, y_test)
    print(acc)

    # store the data from the experiment
    """
    if MONITOR:
        if not os.path.exists(PATH_TO_CONFIGS):
            os.mkdir(PATH_TO_CONFIGS)
        monitor_experience(out_file=os.path.join(PATH_TO_CONFIGS, RESULTS_OUT))
    """