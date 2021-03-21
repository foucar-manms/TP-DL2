""" On complètera au fur et à mesure un programme principal principal_DNN_MNIST permettant d’apprendre
    un réseau de neurones profonds pré-entrainé par via un DBN
"""

import sys
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from numpy.matrixlib.defmatrix import matrix

from tqdm import tqdm
from mlxtend.data import loadlocal_mnist

from utils import cross_entropy, vec2img
from principal_RBM_alpha import entree_sortie_RBM, lire_alpha_digit
from principal_DBN_alpha import DNNStruct, generer_image_DBN, pretrain_DNN, PATH_TO_STRUCTURES




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
                ###################

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
    
    PATH = 'data'
    FILES = [
        "t10k-images.idx3-ubyte",
        "t10k-labels.idx1-ubyte",
        "train-images.idx3-ubyte",
        "train-labels.idx1-ubyte"
    ]

    if not os.path.exists(PATH):
        os.mkdir(PATH)
    for f in FILES:
        assert os.path.isfile(os.path.join(PATH, f)), (
            "the file {} does not exist in the {}\\ folder, make sure you have "
            "a {}\\ folder with {} in it (you may download it at "
            "http://yann.lecun.com/exdb/mnist/). You may also want "
            "to check that the file name match yours".format(f, PATH, PATH, f)
        )

    X_train, y_train = loadlocal_mnist(
        images_path='data\\train-images.idx3-ubyte', 
        labels_path='data\\train-labels.idx1-ubyte'
    )
    X_test, y_test = loadlocal_mnist(
        images_path='data\\t10k-images.idx3-ubyte', 
        labels_path='data\\t10k-labels.idx1-ubyte'
    )
    n_classes = 10

    train_labels = np.zeros(shape=(len(y_train), n_classes))
    X = np.zeros(shape=(0, X_train.shape[1]))
    # one-hot encoding
    index = []
    for i in range(len(y_train)):
        train_labels[i][y_train[i]] = 1.
        if y_train[i] == 3:
            index.append(i)
    X = X_train[np.array(index)]
    for i in range(len(X)):
        X[i] = (X[i] != 0) * 1.
    input_dim = X.shape[1]
    size = [input_dim, 300, 200]
    dnn_struct = DNNStruct(size)
    """
    dnn_struct = pretrain_DNN(X, dnn_struct, n_epochs=100, lr=0.01, batch_size=64)
    generer_image_DBN(6, dnn_struct, img_shape='MNIST')
    pkl.dump(dnn_struct, open("DNN_structures\\pretrain_1.pkl", "wb"))
    """
    net_size = size + [n_classes]
    dnn_struct = DNNStruct(net_size)
    pretrained = pkl.load(open("DNN_structures\\pretrain_1.pkl", "rb"))
    for i in range(len(pretrained.parameters)):
        dnn_struct.update_layer(*pretrained.parameters[i], i)
    dnn_struct = retropropagation(X_train, train_labels, dnn_struct, lr=0.2, n_epochs=10, batch_size=64)
    pkl.dump(dnn_struct, open("DNN_structures\\a_dnn.pkl", "wb"))
    acc = test_DNN(dnn_struct, X_test, y_test)
    print(acc)