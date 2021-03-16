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