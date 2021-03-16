""" On complètera au fur et à mesure un programme principal principal_DNN_MNIST permettant d’apprendre
    un réseau de neurones profonds pré-entrainé par via un DBN
"""


def calcul_softmax():
    """ prend en argument un RBM, des données d’entrée et retourne des probabilités sur les
        unités de sortie à partir de la fonction softmax
    """
    pass

def entree_sortie_reseau():
    """ prend en argument un DNN, des données en entrée du réseau et retourne dans un tableau
        les sorties sur chaque couche cachées du réseau ainsi que les probabilités sur les
        unités de sortie.
        
        Cette fonction pourra utiliser les fonctions entree_sortie_RBM et calcul_softmax
    """
    pass

def retropropagation():
    """ estime les poids/biais du réseau à partir de données labellisées, retourne un DNN et
        prend en argument un DNN, le nombre d’itérations de la descente de gradient, le learning
        rate, la taille du mini-batch, des données d’entrée, leur label,...
        
        On pensera à calculer à la fin de chaque epoch, après la mise à jour des paramètres, la
        valeur de l’entropie croisée que l’on cherche à minimiser afin de s’assurer que l’on
        minimise bien cette entropie
    """
    pass

def test_DNN():
    """ teste les performances du réseau appris, prend en argument un DNN appris, un jeu de données
        test, et les vrais labels associés
        
        Elle commencera par estimer le label associé à chaque donnée test (on pourra utiliser
        entree_sortie_reseau) puis comparera ces labels estimés aux vrais labels. Elle retournera
        enfin le taux d’erreur
    """
    pass