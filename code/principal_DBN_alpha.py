""" On complètera un script principal_DBN_alpha permettant d’apprendre les caractères de la base Binary
    AlphaDigits de votre choix via un DBN et de générer des caractères similaires à ceux appris. La
    construction de ce programme nécessite les fonctions suivantes
"""


def init_DNN():
    """ construit et initialise (éventuellement aléatoirement) les poids et les biais d’un DNN
    
        Cette fonction retournera une structure DNN, prendra en argument la taille du réseau et
        pourra utiliser de manière itérative la fonction précédente
    """
    pass

def pretrain_DNN():
    """ apprend de manière non supervisée un DBN (Greedy layer wise procedure)
    
        Cette fonction retournera un DNN pré-entrainé et prendra en argument un DNN, le nombre
        d’itérations de la descente de gradient, le learning rate, la taille du mini-batch, des
        données d’entrées. On rappelle que le pré-entrainement d’un DNN peut être vu comme
        l’entrainement successif de RBM. Cette fonction utilisera donc train_RBM ainsi que
        entree_sortie_RBM
    """
    pass

def generer_image_DBN():
    """ génère des échantillons suivant un DBN
    
        Cette fonction retournera et affichera les images générées et prendra en argument un DNN
        pré-entrainé, le nombre d’itérations à utiliser dans l’échantillonneur de Gibbs et le
        nombre d’images à générer
    """
    pass

