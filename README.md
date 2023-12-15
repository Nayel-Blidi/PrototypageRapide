# Noise cancelling using deep learning
A proof-of-concept applied to telecoms

## Plan
### Introduction
L’émission de messages en binaires dans la télécommunication terrestre et/ou spatiale, les techniques d’encodage et leur coût calculatoire.
### Problématique	
Comment générer un réseau de neurones permettant le décodage de signaux binaires perturbés par des interférences de télécommunications ?
### Etat de l’art
- MAP
- PSK, QAM
- Encodage (Polar, Hamming, Gray)
- Thèse originale (8-bits, fully connected nn)
### Spécifications
- Intensité du bruit (en dB)
- Complexité du réseau de neurones
- Réussite et taille de dictionnaires
- Langage/bibliothèques
### Réseaux de neurone
- V1 : structure, perfs
- V2 : structure, perfs
- V3: …
### Simulation ordinateur
- PSK Class (carrier phase) (low quality video data)
- QAM Class (carrier phase and amplitude) (hd video data eg: 4096 QAM)
### Simulation électronique
- Domaine d’application
- Bruit simulé/apparenté
- Résultat du décodage avec le NN
### Conclusion
### Bibliographie / Sitographie
   
## Structure app
   
### main.py  
- calls imageHandler functions  
- offers input selections / folder / images  
### imageHandler.py  
- loads images  
- applies class modulation   
- applies class demodulation and/or nn demodulation  
- evaluates performances either way  
### PSK.py  
- applies Bpsk mod  
- applies Qpsk mod?   
- proposes pipelines methods  
### QAM.py  
- applies 16qam mod  
- applies 64-4096qam mod?  
- proposes pipelines methods  
### mainDeepPSK  
- trains/tests psk nn  
- proposes evaluation method  
### mainDeepQAM  
- trains/tests QAM nn  
- proposes evaluation methods   
  
