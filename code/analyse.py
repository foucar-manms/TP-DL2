"""
Voici enfin un plan pour vous aider à comparer les performances d’un réseau pré-entrainé avec un réseau
initialisé aléatoirement. À partir des données ’train’ de la base MNIST :
    1. initialiser deux réseaux identiques;
    2. pré-apprendre un des deux réseau en le considérant comme un empilement de RBM (apprentissage non
supervisé);
    3. apprendre le réseau pré-appris préalablement avec l’algorithme de rétro-propagation;
    4. apprendre le second réseau qui a été initialisé aléatoirement avec l’algorithme de rétro-propagation;
    5. Calculer les taux de mauvaises classifications avec le réseau 1 (pré-entrainé + entraîné) et le
réseau 2 (entraîné) à partir du jeu ’train’ et du jeu ’test’
    3 figures suffisent à analyser vos résultats :
        - Fig 1 : 2 courbes exprimant le taux d’erreur des 2 réseaux en fonction du nombre de couches (par
            exemple 2 couches de 200, puis 3 couches de 200, ... puis 5 couches de 200);
        - Fig 2 : 2 courbes exprimant le taux d’erreur des 2 réseaux en fonction du nombre de neurones par
            couches (par exemple 2 couches de 100, puis 2 couches de 300, ...puis 2 couches de 700,...);
        - Fig 3 : 2 courbes exprimant le taux d’erreur des 2 réseaux en fonction du nombre de données train
            (par exemple on fixe 2 couches de 200 puis on utilise 1000 données train, 3000, 7000, 10000,
            30000, 60000).
"""