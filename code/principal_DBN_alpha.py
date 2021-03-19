""" On complètera un script principal_DBN_alpha permettant d’apprendre les caractères de la base Binary
    AlphaDigits de votre choix via un DBN et de générer des caractères similaires à ceux appris. La
    construction de ce programme nécessite les fonctions suivantes
"""

import sys
import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from utils import index2char, sigmoid, vec2img, sigmoid
from principal_RBM_alpha import RBMStruct, init_RBM, train_RBM, entree_sortie_RBM, lire_alpha_digit, sortie_entree_RBM



PATH_TO_STRUCTURES = "DNN_structures"


def init_DNN(size):
    """ construit et initialise (éventuellement aléatoirement) les poids et les biais d’un DNN
    
        Cette fonction retournera une structure DNN, prendra en argument la taille du réseau et
        pourra utiliser de manière itérative la fonction précédente

        args:
            - size: iterable object whose length is the number of layers of the DNN and containing
        the dimension of each layer ; the dimension of the first layer should match the DNN input size

        return:
            dnn_structure: a list of tuples (bias_in, bias_h, weights)
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
        dnn_structure.append(init_RBM(in_dim, out_dim))
    return dnn_structure

def pretrain_DNN(inputs, dnn_struct, n_epochs, lr, batch_size):
    """ apprend de manière non supervisée un DBN (Greedy layer wise procedure)
    
        Cette fonction retournera un DNN pré-entrainé et prendra en argument un DNN, le nombre
        d’itérations de la descente de gradient, le learning rate, la taille du mini-batch, des
        données d’entrées. On rappelle que le pré-entrainement d’un DNN peut être vu comme
        l’entrainement successif de RBM. Cette fonction utilisera donc train_RBM ainsi que
        entree_sortie_RBM
    """

    RBM_stack = []
    for i in range(len(dnn_struct.rbm_stack)):
        # get the dimension of the current layer and the next one
        rbm_struct = dnn_struct.make_rbm(i)
        print("training the {}-th layer...".format(i+1))
        rbm_struct = train_RBM(inputs, rbm_struct, n_epochs, lr, batch_size)
        print()
        dnn_struct.update_rbm(rbm_struct, i)

        # update inputs (make it the hidden activation of the current RBM)
        inputs = entree_sortie_RBM(inputs, rbm_struct)
    return dnn_struct

def generer_image_DBN(n_imgs, dnn_struct, img_shape='binary_alpha_digits', n_iter_gibbs=5):
    """ génère des échantillons suivant un DBN
    
        Cette fonction retournera et affichera les images générées et prendra en argument un DNN
        pré-entrainé, le nombre d’itérations à utiliser dans l’échantillonneur de Gibbs et le
        nombre d’images à générer

        args:
            - n_imgs: number of images to generate
            - dnn_struct: instance of DNNStruct
            - n_iter_gibbs: number of iteration per gibbs sampling process
    """
    
    for _ in range(n_imgs):
        rbm_struct = dnn_struct.make_rbm(-1)
        input_dim = rbm_struct.get_input_dim()
        v = np.expand_dims((np.random.rand(input_dim) < 1/2) * 1, axis=0)
        for _ in range(n_iter_gibbs):
            p_h = entree_sortie_RBM(v, rbm_struct)
            h = (np.random.rand(*p_h.shape) < p_h) * 1
            p_v = sortie_entree_RBM(h, rbm_struct)
            v = (np.random.rand(*p_v.shape) < p_v) * 1
        for i in reversed(range(len(dnn_struct.rbm_stack)-1)):
            rbm_struct = dnn_struct.make_rbm(i)
            p_v = sortie_entree_RBM(v, rbm_struct)
            v = (np.random.rand(*p_v.shape) < p_v) * 1
        img = vec2img(v, matrix_shape=img_shape)
        plt.imshow(img, cmap='gray')
        plt.show()




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
        self.rbm_stack = init_DNN(size)
        self.parameters = [(w, b) for _, b, w in self.rbm_stack]
        self.size = [i for i in size]
        self.input_dim = size[0]
        self.output_dim = size[-1]

    def __call__(self):
        """ return the parameters """
        return self.parameters

    def __getitem__(self, index):
        """ return (weights, bias) for the index-th layer """
        return self.parameters[index]

    def __len__(self):
        """ return the number of layers of the DNN """
        return len(self.parameters)

    def __str__(self):
        string = ("size : " +  str(self.size) + '\n' + 
            "parameters shapes : " + str([(a.shape, b.shape, w.shape) for a, b, w in self.rbm_stack])
        )
        return string

    def update_parameters(self, parameters):
        """ update the DNN parameters """
        self.parameters = parameters

    def update_layer(self, weights, bias, index):
        """ update the parameters of the index-th layer """
        self.parameters[index] = (weights, bias)

    def update_rbm(self, rbm_struct, index):
        """ update the parameters of the index-th RBM """
        self.rbm_stack[index] = rbm_struct()

    def get_input_dim(self):
        return self.input_dim
    
    def get_output_dim(self):
        return self.output_dim

    def make_rbm(self, index):
        """ return an RBMStruct instance based on the parameters of the index-th RBM of the network """
        b_in, b_h, w = self.rbm_stack[index]
        input_dim = len(b_in)
        hidden_dim = len(b_h)
        rbm_struct = RBMStruct(input_dim, hidden_dim).update_parameters(b_in, b_h, w)
        return rbm_struct




if __name__ == "__main__":
    
    to_mimic = '5'
    if isinstance(to_mimic, int):
        to_mimic = index2char(to_mimic)
    inputs = lire_alpha_digit(char=to_mimic)
    input_dim = inputs.shape[1]
    net_size = [input_dim, 256, 128, 64]
    dnn_struct = DNNStruct(net_size)
    print(dnn_struct)
    sys.exit()
    dnn_struct = pretrain_DNN(inputs, dnn_struct, n_epochs=1000, lr=0.1, batch_size=10)
    try:
        pkl.dump(
            dnn_struct, open(os.path.join(PATH_TO_STRUCTURES, "DNN_structure_") + to_mimic + ".pkl", "wb")
        )
    except FileNotFoundError:
        os.mkdir(PATH_TO_STRUCTURES)
        pkl.dump(
            dnn_struct, open(os.path.join(PATH_TO_STRUCTURES, "DNN_structure_") + to_mimic + ".pkl", "wb")
        )
    #dnn_struct = pkl.load(open(os.path.join(PATH_TO_SRUCTURES, "dnn_structure_5.pkl"), "rb"))
    generer_image_DBN(6, dnn_struct, n_iter_gibbs=20)