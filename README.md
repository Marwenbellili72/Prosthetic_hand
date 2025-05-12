
## Installation et Lancement Local

1.  **Cloner le dépôt :**
    ```bash
    git clone https://github.com/Marwenbellili72/Prosthetic_hand.git
    cd Prosthetic_hand
    ```

2.  **Créer un environnement virtuel (recommandé) :**
    ```bash
    python -m venv venv
    # Sur Windows:
    .\venv\Scripts\activate
    # Sur macOS/Linux:
    source venv/bin/activate
    ```

3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
    *Note : Assurez-vous que les versions des bibliothèques (notamment Scikit-learn) sont compatibles avec les fichiers `.pkl` et le modèle `.json`, ou regénérez ces fichiers si nécessaire.*

4.  **Vérifier les fichiers requis :** Assurez-vous que `emg_xgboost_model.json`, `emg_scaler.pkl`, et le répertoire `static/models/hand_model.glb` sont présents.

5.  **Lancer le serveur FastAPI :**
    ```bash
    uvicorn main_api:app --reload --host 127.0.0.1 --port 8000
    ```
    *(Utilisez `--host 0.0.0.0` si vous exécutez dans Docker ou avez besoin d'y accéder depuis une autre machine sur le réseau local)*

6.  **Accéder à l'application :** Ouvrez votre navigateur et allez à `http://127.0.0.1:8000`.

**(Alternative avec Docker)**

Si un `Dockerfile` est configuré :

1.  **Construire l'image Docker :**
    ```bash
    docker build -t prosthetic-hand-app .
    ```
2.  **Lancer le conteneur Docker :**
    ```bash
    docker run -d -p 8000:8000 --name my-prosthetic-app prosthetic-hand-app
    ```
3.  **Accéder à l'application :** `http://localhost:8000`

## Utilisation

1.  Ouvrez l'interface web dans votre navigateur.
2.  Cliquez sur le bouton de sélection de fichier ("Choisir un fichier" ou similaire).
3.  Sélectionnez un fichier `.mat` contenant les données EMG prétraitées (la structure attendue est une variable nommée `emg` dans le fichier `.mat`).
4.  Cliquez sur le bouton "Prédire et Visualiser".
5.  Attendez que le traitement soit terminé. Le statut s'affichera.
6.  Les résultats apparaîtront :
    *   Dans la section "Informations", vous verrez le nom du fichier et le nombre de fenêtres traitées.
    *   Dans la section "Mouvements Détectés", une liste des mouvements prédits (avec leur compte et pourcentage) sera affichée. Le mouvement majoritaire sera mis en évidence.
7.  Le modèle 3D dans la section "Visualisation 3D" devrait afficher la pose correspondant au mouvement prédit majoritaire (ou le dernier mouvement cliqué dans la liste). Vous pouvez faire tourner/zoomer le modèle.
8.  (Optionnel) Utilisez les boutons "Tout jouer" et "Arrêter" pour voir une animation séquentielle des mouvements prédits sur le modèle 3D.

## Déploiement

Une version de cette application est déployée sur Hugging Face Spaces et accessible ici :
[https://huggingface.co/spaces/Marwenbellili72/prosthetic_hand](https://huggingface.co/spaces/Marwenbellili72/prosthetic_hand)
*(Vérifiez et confirmez ce lien)*

## Travaux Futurs

*   **Traitement en Temps Réel :** Adapter le système pour traiter les signaux EMG en streaming via WebSocket ou une autre technologie, plutôt que des fichiers préenregistrés.
*   **Amélioration du Modèle :** Explorer d'autres architectures de modèles (ex: LSTMs, CNNs) pour potentiellement améliorer la précision et la robustesse de la classification.
*   **Plus de Mouvements :** Entraîner le modèle à reconnaître un plus grand nombre de gestes et de préhensions.
*   **Intégration Hardware :** Connecter le système à un microcontrôleur (ex: ESP32, Arduino) pour acquérir des données EMG réelles et potentiellement contrôler une prothèse physique.
*   **Calibration Utilisateur :** Implémenter une procédure de calibration pour adapter le modèle aux signaux spécifiques de chaque utilisateur.
*   **Feedback Haptique :** Explorer l'ajout de retours haptiques.
*   **Interface d'Entraînement :** Créer une interface permettant aux utilisateurs d'enregistrer de nouvelles données et de ré-entraîner/affiner le modèle.

## Licence

Ce projet est distribué sous la licence [Nom de Votre Licence, ex: MIT]. Voir le fichier `LICENSE` pour plus de détails. *(Vous devez choisir une licence et ajouter un fichier LICENSE si ce n'est pas déjà fait).*
