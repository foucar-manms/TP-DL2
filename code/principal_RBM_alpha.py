""" On complètera au fur et à mesure un script principal_RBM_alpha permettant d’apprendre les caractères de
    la base Binary AlphaDigits de votre choix via un RBM et de générer des caractères similaires à ceux appris.
    La construction de ce programme nécessite les fonctions suivantes
"""

import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy import io

from utils import char2index, index2char, sigmoid, vec2img




def lire_alpha_digit(char, path="data", file="binaryalphadigs.mat"):
    """ make a matrix out of input file, the output rows are flattened input images

        args:
            - char: either a string (one of '0', ..., '9', 'a', ..., 'z') or an integer
        (index between 0 and 35) corresponding to the character we want to learn
    """
    if not os.path.exists(path):
        os.mkdir(path)
    assert os.path.isfile(os.path.join(path, file)), (
        "the file {} does not exist in the {} folder, make sure you have "
        "a {} folder with {} in it (you may download it at "
        "https://cs.nyu.edu/~roweis/data.html)".format(file, path, path, file)
    )
    mat = io.loadmat(os.path.join(path, file))['dat']
    row_shape = mat[0][0].flatten().shape[0]
    imgs_flat = np.empty((0, row_shape), dtype=float)
    if isinstance(char, type('')):
        char = char2index(char)
    imgs = mat[char]
    for j in range(len(imgs)):
        img2vec = imgs[j].flatten()
        imgs_flat = np.vstack((imgs_flat, img2vec))
    return imgs_flat

def init_RBM(input_dim, hidden_dim):
    """ construit et initialise les poids et les biais d’un RBM """

    input_bias, hidden_bias = np.zeros(input_dim), np.zeros(hidden_dim)
    weights = np.random.randn(input_dim, hidden_dim) * np.sqrt(0.01)
    return input_bias, hidden_bias, weights

def entree_sortie_RBM(inputs, rbm_struct):
    """ prend en argument une structure RBM et des données d’entrée et retourne la valeur des unités
        de sortie calculées à partir de la fonction sigmoïde ; les dimensions entre les données
        d'entrée et les paramètres doivent être consistantes

        args:
            - inputs: array or similar object containing input data
            - rbm_struct: should be an instance of class RBMStruct
    """
    _, h_b, w = rbm_struct()
    assert inputs.shape[1] == rbm_struct.get_input_dim(), (
        "the size of the inputs data should match that of input bias ; " 
        "received sizes {} and {}".format(inputs.shape[1], rbm_struct.get_input_dim())
    )
    return sigmoid(np.dot(inputs, w) + h_b)

def sortie_entree_RBM(h_inputs, rbm_struct):
    """ prend en argument un RBM, des données de sortie et retourne la valeur des unités d’entrée
        à partir de la fonction sigmoïde ; les dimensions entre les données d'entrée et les
        paramètres doivent être consistantes

        args:
            - h_inputs: array or similar object containing hidden data
            - rbm_struct: should be an instance of class RBMStruct
    """
    in_b, _, w = rbm_struct()
    assert h_inputs.shape[1] == rbm_struct.get_hidden_dim(), (
        "the size of the hidden inputs data should match that of output bias ; " 
        "received sizes {} and {}".format(h_inputs.shape[1], rbm_struct.get_hidden_dim())
    )
    return sigmoid(np.dot(h_inputs, w.T) + in_b)

def train_RBM(inputs, rbm_struct, n_epochs=10, lr=0.01, batch_size=4):
    """ apprendre de manière non supervisée un RBM par l’algorithme Contrastive-Divergence-1
    
        Cette fonction retournera une structure RBM et prendra en argument une structure RBM, le nombre
        d’itérations de la descente de gradient (epochs), le learning rate, la taille du mini-batch,
        des données d’entrées... À la fin de chaque itération du gradient, on affichera l’erreur
        quadratique entre les données d’entrées et les données reconstruites à partir de l’unité cachée
        afin de mesurer le pouvoir de reconstruction du RBM

        args:
            - inputs: array-like object, the rows are input vectors
            - rbm_struct: instance of RBMStruct
            - n_epochs: number of epochs to iterate over
            - lr: learning rate for the contrastive divergence
            - batch_size: input mini-batch size
        
        returns:
            - rbm_struct: instance of RBMStruct containing trained parameters of the RBM
    """

    in_b, h_b, w = rbm_struct()
    n_examples = len(inputs)
    n_batch = n_examples // batch_size + 1 * (n_examples % batch_size != 0)
    tqdm_dict = {"reconstruction quadratic error": 0.0}
    for epoch in range(n_epochs):
        
        with tqdm(total=n_batch, unit_scale=True, desc="Epoch : %i/%i" % (epoch+1, n_epochs),
                    ncols=100) as pbar:
            
            indexes = np.random.permutation(n_examples)
            
            for i in range(0, n_examples, batch_size):
                batch_indexes = indexes[i:(i+batch_size)]
                batch_in = inputs[batch_indexes]
                # usually equal to batch_size, except for the last batch that may be shorter
                actual_batch_size = len(batch_in)
                
                # one step Gibbs sampling
                #########################
                # hidden units conditional probabilities
                h_batch_units = entree_sortie_RBM(batch_in, rbm_struct)
                # sample hidden
                h_batch = (np.random.rand(*h_batch_units.shape) < h_batch_units) * 1
                # visible units conditional probabilities
                batch_units = sortie_entree_RBM(h_batch, rbm_struct)
                # sample visible
                batch_out = (np.random.rand(*batch_units.shape) < batch_units) * 1
                # recompute hidden unit conditional probabilities
                h_batch_units_out = entree_sortie_RBM(batch_out, rbm_struct)
                #########################

                # update parameters
                db_in_sum = np.sum(batch_in - batch_out, axis=0)
                db_out_sum = np.sum(h_batch_units - h_batch_units_out, axis=0)
                dw_sum = np.dot(batch_in.T, h_batch_units) - np.dot(batch_out.T, h_batch_units_out)
                in_b = in_b + lr * db_in_sum / actual_batch_size
                h_b = h_b + lr * db_out_sum / actual_batch_size
                w = w + lr * dw_sum / actual_batch_size
                rbm_struct.update_parameters(in_b, h_b, w)
                ###################

                reconstruction_err = np.sum(
                    np.sum((batch_in - batch_out)**2, axis=1), axis=0
                ) / actual_batch_size
                tqdm_dict['reconstruction quadratic error'] = reconstruction_err
                pbar.set_postfix(tqdm_dict)
                pbar.update(1)
    return rbm_struct

def generer_image_RBM(n_imgs, rbm_struct, img_shape='binary_alpha_digits', n_iter_gibbs=5):
    """ génère des échantillons suivant un RBM
    
        Cette fonction retournera et affichera les images générées et prendra en argument une structure
        de type RBM, le nombre d’itérations à utiliser dans l’échantillonneur de Gibbs et le nombre
        d’images à générer

        args:
            - n_imgs: number of images to generate
            - rbm_struct: instance of RBMStruct
            - n_iter_gibbs: number of iteration per gibbs sampling process
    """
    
    input_dim = rbm_struct.get_input_dim()
    for _ in range(n_imgs):
        v = np.expand_dims((np.random.rand(input_dim) < 1/2) * 1, axis=0)
        for _ in range(n_iter_gibbs):
            p_h = entree_sortie_RBM(v, rbm_struct)
            h = (np.random.rand(*p_h.shape) < p_h) * 1
            p_v = sortie_entree_RBM(h, rbm_struct)
            v = (np.random.rand(*p_v.shape) < p_v) * 1
        img = vec2img(v, matrix_shape=img_shape)
        plt.imshow(img, cmap='gray')
        plt.show()





class RBMStruct:
    """ basic structure that contains the parameters of an RBM """

    def __init__(self, input_dim, hidden_dim):
        self.input_bias, self.hidden_bias, self.weights = init_RBM(
            input_dim=input_dim, hidden_dim=hidden_dim
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def __call__(self):
        """ return the parameters """
        return self.input_bias, self.hidden_bias, self.weights

    def update_parameters(self, input_bias, hidden_bias, weights):
        """ update the RBM parameters """
        self.input_bias = input_bias
        self.hidden_bias = hidden_bias
        self.weights = weights

    def get_input_dim(self):
        return self.input_dim
    
    def get_hidden_dim(self):
        return self.hidden_dim




if __name__ == "__main__":
    to_mimic = 'b'
    if isinstance(to_mimic, int):
        to_mimic = index2char(to_mimic)
    inputs = lire_alpha_digit(to_mimic)
    input_dim = inputs.shape[1]
    hidden_dim = 100
    rbm_struct = RBMStruct(input_dim, hidden_dim)
    rbm_struct = train_RBM(inputs, rbm_struct, lr=0.1, n_epochs=3000, batch_size=10)
    
    try:
        pkl.dump(rbm_struct, open("RBM_structures\\RBM_structure_" + to_mimic + ".pkl", "wb"))
    except FileNotFoundError:
        os.mkdir("RBM_structures")
        pkl.dump(rbm_struct, open("RBM_structures\\RBM_structure_" + to_mimic + ".pkl", "wb"))
    
    #rbm_struct = pkl.load(open("RBM_structures\\RBM_structure_h.pkl", "rb"))
    generer_image_RBM(n_imgs=6, rbm_struct=rbm_struct)