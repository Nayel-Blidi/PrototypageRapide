# PrototypageRapide

I - Spécifications :

Problématique : générer un réseau de neurones permettant le décodage de signaux binaires perturbés par des interférences de télécommunications.

Les spécifications sont divisées en deux parties : 
Les interférences/perturbations, coeur du problème, qui décident à quel point notre étude de cas est réaliste
Les conditions de réalisation, qui fixent les objectifs du projet pour un signal donné
Les deux spécifications sont considérées comme indépendantes, et il s’agit de trouver un équilibre entre ces deux contraintes pour parvenir à un résultat convaincant.

Interférences
Atmosphère
Autres signaux
a) et b) peuvent probablement être asimilés à un AWGN (additive white gaussian noise) : bruit blanc, normalement distribué, centré en 0 (moyenne) et de paramètre sigma (écart type)
Ionosphère
ToDo : étudier comment simuler statistiquement des perturbations de la ionosphère, et si un modèle général peut être appliqué (a contrario, les conditions ionosphériques peuvent avoir des impacts trop importants pour ne pas être traités à part)
Autres perturbations et couches atmosphériques
ToDo : étudier les autres perturbations majeurs de signaux pouvant intervenir, et comment les modéliser 

Réalisation 
Réseau de neurone Pytorch (python)
Le réseau de neurones devrait être une version améliorée de celle en Keras. Il faudrait intégrer des techniques de traitement du signal pour améliorer les points suivants :
Réduire la complexité et le temps de calcul
Augmenter la vitesse de convergence et la fiabilité
Augmenter la taille des dictionnaires de mots binaires décodables
Simulateur (simulink/python/C++)
b : Simulation de l’encodeur physique couplé à son décodeur neuronal et/ou physique
b-bis : Simulation de l’encodeur physique sur python
b-ter : Simulation de l’encodeur physique sur python, interface graphique et portabilité C++ (exécutable)
Simulateur physique 
Réaliser (ou utiliser) un prototype d’émetteur/récepteur de signal (laser, antenne, générateur…) qui traverse un canal perturbé (espace vide, résistance, liquide…) dans le but d’être ensuite décodé.



II - État de l’art :



https://en.wikipedia.org/wiki/Phase-shift_keying

Insérer shéma blocs codeur/décodeur des PSK

Encodeurs
- Gray code :

- Polar code :

- Hamming code :





Décodeurs

- MAP decoding techniques (Maximum A-Posteriori) : méthode statistique de décodage de bits

- BPSK/QPSK : encodage sur 1 ou 2 bits, distance dans le plan complexe identique, même BER


- 8/16-PSK : mots binaires plus longs, plus dur à séparer sur le disque unité


- 8/16 QAM : mots binaires plus longs, relativement durs à séparer dans le plan complexe








Réseaux de neurones

- Ancienne version Tensorflow/Keras : 4 fully connected dense layers

- Nouvelle version Pytorch : 
	- Commencer par refaire le même réseau 
	- Implémenter un réseau de convolution 1D (CNN)



