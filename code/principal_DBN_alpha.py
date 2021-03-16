""" On complètera un script principal_DBN_alpha permettant d’apprendre les caractères de la base Binary
    AlphaDigits de votre choix via un DBN et de générer des caractères similaires à ceux appris. La
    construction de ce programme nécessite les fonctions suivantes
"""

import sys
import os
import pickle as pkl

from utils import index2char
from principal_RBM_alpha import RBMStruct, init_RBM, train_RBM, entree_sortie_RBM, lire_alpha_digit





def init_DNN(size):
    """ construit et initialise (éventuellement aléatoirement) les poids et les biais d’un DNN
    
        Cette fonction retournera une structure DNN, prendra en argument la taille du réseau et
        pourra utiliser de manière itérative la fonction précédente

        args:
            - size: iterable object whose length is the number of layers of the DNN and containing
        the dimension of each layer ; the dimension of the first layer should match the DNN input size

        return:
            dnn_structure: a list of tuples (weights, bias) (arrays of shape (in,out) and (out,)
        respectively) that are the DNN parameters for the corresponding layer
    """

    assert len(size) > 1, (
        "the list of sizes should be at least 2 (input size, output size), found {} "
        .format(len(size))
    )
    
    if not isinstance(size, list):    
        # make a list out of the input
        list_of_sizes = []
        for n_units in size:
            list_of_sizes.append(n_units)
    else: list_of_sizes = size.copy()

    dnn_structure = []
    for i in range(len(list_of_sizes) - 1):
        in_dim, out_dim = list_of_sizes[i], list_of_sizes[i+1]
        _, bias, weights = init_RBM(in_dim, out_dim)
        dnn_structure.append((weights, bias))
    return dnn_structure

def pretrain_DNN(inputs, dnn_struct, n_epochs, lr, batch_size):
    """ apprend de manière non supervisée un DBN (Greedy layer wise procedure)
    
        Cette fonction retournera un DNN pré-entrainé et prendra en argument un DNN, le nombre
        d’itérations de la descente de gradient, le learning rate, la taille du mini-batch, des
        données d’entrées. On rappelle que le pré-entrainement d’un DNN peut être vu comme
        l’entrainement successif de RBM. Cette fonction utilisera donc train_RBM ainsi que
        entree_sortie_RBM
    """

    # the sequence (list) of (wieghts, bias) for each layer
    structure = dnn_struct()
    print([(w.shape, b.shape) for w, b in structure])

    for i in range(len(dnn_struct)):
        # get the dimension of the current layer and the next one
        in_dim = len(dnn_struct[i][0])
        h_dim = len(dnn_struct[i][1])
        # initialize a random rbm structure
        rbm_struct = RBMStruct(in_dim, h_dim)
        print("training the {}-th layer...".format(i+1))
        rbm_struct = train_RBM(inputs, rbm_struct, n_epochs, lr, batch_size)
        print()
        _, bias, weights = rbm_struct()
        dnn_struct.update_layer(weights, bias, i)

        # update inputs (make it the hidden activation of the current RBM)
        inputs = entree_sortie_RBM(inputs, rbm_struct)
    return dnn_struct

def generer_image_DBN():
    """ génère des échantillons suivant un DBN
    
        Cette fonction retournera et affichera les images générées et prendra en argument un DNN
        pré-entrainé, le nombre d’itérations à utiliser dans l’échantillonneur de Gibbs et le
        nombre d’images à générer
    """
    pass




class DNNStruct:
    """ basic structure that contains the parameters of a DNN

        attributes:
            - structure: list of sizes (input dim, output dim) for each dimension
            - size: list of dimensions
            - input_dim: dimension of the input of the DNN ; corresponds to the first index
        of the list of sizes
            - output_dim: dimension of the output of the DNN
    """

    def __init__(self, size):
        assert len(size) > 1, (
            "the list of sizes should be at least 2 (input size, output size), found {} "
            .format(len(size))
        )
        self.structure = init_DNN(size)
        self.size = size
        self.input_dim = size[0]
        self.output_dim = size[-1]

    def __call__(self):
        """ return the parameters """
        return self.structure

    def __getitem__(self, index):
        """ return (weights, bias) for the index-th layer """
        return self.structure[index]

    def __len__(self):
        """ return the number of layers of the DNN """
        return len(self.structure)

    def update_parameters(self, structure):
        """ update the DNN parameters """
        self.structure = structure

    def update_layer(self, weights, bias, index):
        """ update the parameters of the index-th layer """
        self.structure[index] = (weights, bias)

    def get_input_dim(self):
        return self.input_dim
    
    def get_output_dim(self):
        return self.output_dim


if __file__ == "__main__":
    to_mimic = '5'
    if isinstance(to_mimic, int):
        to_mimic = index2char(to_mimic)
    inputs = lire_alpha_digit(char=to_mimic)
    input_dim = inputs.shape[1]
    net_size = [input_dim, 100, 100]
    dnn_struct = DNNStruct(net_size)
    dnn_struct = pretrain_DNN(inputs, dnn_struct, n_epochs=1000, lr=0.1, batch_size=10)
    try:
        pkl.dump(dnn_struct, open("DNN_structures\\DNN_structure_" + to_mimic + ".pkl", "wb"))
    except FileNotFoundError:
        os.mkdir("DNN_structures")
        pkl.dump(dnn_struct, open("DNN_structures\\DNN_structure_" + to_mimic + ".pkl", "wb"))