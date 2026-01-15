# People Counter – Application de comptage de personnes

## Présentation

People Counter détecte et compte en temps réel le nombre de personnes dans une salle à partir d’une caméra. L’application affiche le flux vidéo, le comptage, l’historique, les statistiques (moyenne, min, max) et permet l’export PDF des résultats.

## Fonctionnalités principales

- Détection et comptage de personnes en temps réel via webcam ou caméra réseau (basé sur YOLOv8 – ultralytics)
- Contrôle du zoom sur caméras réseau compatibles ONVIF (si disponible)
- Affichage du flux vidéo, du nombre de personnes, de l’historique et des statistiques
- Export PDF et Excel des résultats
- Respect total de la confidentialité (aucune donnée collectée ou transmise)

## Pour les utilisateurs

L'installation la plus simple se fait via un exécutable Windows prêt à l'emploi :

1. Téléchargez le fichier `main.exe` fourni par le développeur (généré dans le dossier `dist/`).
2. Placez-le dans le dossier de votre choix sur votre PC Windows.
3. Double-cliquez sur `main.exe` pour lancer l'application.
4. Sélectionnez la caméra et cliquez sur « Démarrer ».
5. Visualisez le comptage, l’historique et les statistiques en temps réel.
6. Exportez un rapport PDF si besoin.
7. Fermez l’application quand vous avez terminé.

## Pour les développeurs

### Prérequis

- Windows 10 ou 11
- Python 3.8 ou supérieur
  - **Vérifier la version installée :**
    Ouvrez un terminal (cmd) et tapez :
    ```
    python --version
    ```
    Si la version affichée est 3.8 ou supérieure, c'est bon.
  - **Installer Python si besoin :**
    - [Télécharger ici](https://www.python.org/downloads/)
    - Ou via le terminal Windows :
      ```
      winget install Python.Python.3
      ```

### Installation

1. Ouvrez un terminal Windows (cmd) dans le dossier du projet.
2. Lancez le script d'installation automatique :

```
setup.bat
```

Ce script crée un environnement virtuel et installe toutes les dépendances nécessaires.

### Mode développement

Après installation :

1. Activez l’environnement virtuel (si ce n’est pas déjà fait) :

```
venv\Scripts\activate
```

2. Lancez l’application :

```
python main.py
```

### Génération de l'exécutable

Pour créer le fichier .exe à distribuer aux utilisateurs :

-

1. Activez l’environnement virtuel (si ce n'est pas déjà fait) :

```
venv\Scripts\activate
```

2. Installez PyInstaller dans l'environnement virtuel (si ce n'est pas déjà fait) :

```
pip install pyinstaller
```

2. Compilez avec :

```
build_exe.bat
```

L'exécutable autonome sera généré dans le dossier `dist/`.

### Fermer l'environnement virtuel

```
deactivate
```

### Publier une release

Pour déclencher la génération automatique et la publication de main.exe sur GitHub, utilisez ces commandes dans le dossier du projet :

1. Préparez et validez vos modifications :

```
git add .
git commit -m "Préparer la release v1.0"
```

2. Créez un nouveau tag (remplacez v1.0 par votre version) :

```
git tag v1.0
```

3. Poussez le code et le tag sur GitHub :

```
git push
git push origin v1.0
```

## Confidentialité

Cette application ne collecte, ne stocke, ni ne transmet aucune donnée utilisateur ou vidéo. Tout le traitement s’effectue localement sur votre appareil.

## Licence

MIT
