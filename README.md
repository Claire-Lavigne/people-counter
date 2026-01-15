# People Counter – Application autonome de comptage de personnes

Cette application détecte et compte en temps réel le nombre de personnes présentes dans une salle à partir d’une caméra. Elle affiche le comptage, l’historique, les statistiques (moyenne, min, max) et permet l’export PDF des résultats. Aucun lien avec OBS n’est requis, mais l’affichage peut être capturé dans n’importe quel logiciel si besoin.

## Fonctionnalités principales

- Détection et comptage de personnes en temps réel via webcam ou caméra réseau
- Affichage du flux vidéo, du nombre de personnes, de l’historique et des statistiques
- Export PDF du graphique et des données
- Respect total de la confidentialité (aucune donnée collectée ou transmise)

## Distribution et installation

1. Double-cliquez sur le fichier `build_exe.bat` dans le dossier du projet pour générer l’exécutable autonome.
2. Récupérez le fichier `main.exe` dans le dossier `dist/`.
3. Distribuez le fichier `main.exe`.

## Utilisation

1. Lancez `main.exe` sur un PC relié à une caméra (double-cliquer pour lancer l’application)
2. Sélectionnez la caméra et cliquez sur « Démarrer ».
3. Visualisez le comptage, l’historique et les statistiques en temps réel.
4. Exportez un rapport PDF si besoin.
5. Fermez l’application quand vous avez terminé.

## Confidentialité

Cette application ne collecte, ne stocke, ni ne transmet aucune donnée utilisateur ou vidéo. Tout le traitement s’effectue localement sur votre appareil.

## Licence

MIT
