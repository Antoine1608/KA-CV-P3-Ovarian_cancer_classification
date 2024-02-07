# Compétition Kaggle : Classification d'Images de Cancers Ovariens
Voici ma contribution à la compétition Kaggle sur la classification d'images de cancers ovariens! Dans ce projet, je vais développer un modèle de classification capable de distinguer différents types de cancers ovariens à partir d'images histologiques.

## Ma Stratégie
Ma stratégie pour aborder cette compétition est la suivante :

Préparation des Images d'Entraînement : Je prendrai en compte les différents zooms présents dans les images. Les images TMA ont un zoom de 40x, tandis que les images WSI ont un zoom de 20x mais ont été converties en thumbnails, ce qui a entraîné une variation de zoom pour chaque image WSI, compris entre 1 et 18. Pour uniformiser les zooms, j'ai découpé les images thumbnail WSI en tuiles de zoom 40x.

Entraînement d'un Modèle VGG16 : J'utiliserai le modèle VGG16 pré-entrainé pour classer les images en sous-types de cancer ovariens. Les sous-types de cancer considérés sont HGSC, CC, MC, LGSC, EC. De plus, j'ai ajouté une classe supplémentaire "other" pour les tuiles ne représentant pas de tissu tumoral, comme les zones noires ou blanches.

Évaluation de la Performance : Je mesurerai la performance de la classification sur les données de validation en utilisant l'accuracy comme métrique principale. Après l'entraînement du modèle sur les images TMA, je ferai des prédictions sur les tuiles des images WSI et attribuerai le label de la classe majoritaire à l'image complète WSI.

## Contributions
Les contributions sont les bienvenues! Si vous avez des suggestions d'amélioration, des rapports de bugs ou des idées pour de nouvelles fonctionnalités, n'hésitez pas à ouvrir une issue ou à soumettre une demande de pull request.

Merci pour votre lecture !# KA-CV-P3-Ovarian_cancer_classification
