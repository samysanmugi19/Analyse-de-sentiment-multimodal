# Analyse-de-sentiment-multimodal

### **Description**
Ce projet combine plusieurs modalités (vidéo, audio et texte) pour prédire le sentiment en utilisant des modèles d'apprentissage profond, inspirés de DEVA. L'objectif est d'analyser les émotions exprimées dans les données multimédia en fusionnant les signaux textuels, audio et visuels. Le système utilise des modèles basés sur BERT pour l'encodage du texte, OpenSMILE pour les caractéristiques audio et OpenFace pour l'analyse des expressions faciales. Le dataset utilisé et CMU - MOSI (Segmented)
# **Prérequis**

*   Python 3.8+
*   PyTorch
*   Transformers (HuggingFace)
*   OpenSMILE (pour l'extraction des caractéristiques audio)
*  OpenFace (pour l'analyse des expressions faciales)
*   Librosa (pour le traitement audio)
*   MoviePy (pour l'extraction de l'audio à partir des vidéos)
*   Pandas, Numpy (pour la manipulation des données)
*   Élément de liste

**Structure du Répertoire**

 analyse_sentiment_multimodale.ipynb : Modèles entraînés et checkpoints.
 bert.py : la class BertTextEncoder


Installez les bibliothèques requises avec pip :
```
pip install torch transformers librosa pandas numpy moviepy
```


### Dataset MOSI

Le **MOSI (Multimodal Opinion Sentiment Intensity)** est un dataset multimodal utilisé pour la prédiction des émotions et des sentiments exprimés à partir de données vidéo, audio et textuelles. Le dataset MOSI contient des vidéos de personnes exprimant des opinions sur des sujets variés, ainsi que les transcriptions textuelles et les caractéristiques audio associées.

Le lien vers le dataset : https://www.kaggle.com/datasets/mathurinache/cmu-mosi

#### Structure du dataset

Le dataset MOSI est composé de plusieurs fichiers, principalement les suivants :
- **Vidéos** : Les vidéos contiennent les expressions faciales et les mouvements corporels des individus exprimant leurs opinions.
- **Audio** : Les fichiers audio contiennent l'enregistrement des voix des personnes dans les vidéos.
- **Textes** : Les transcriptions textuelles des vidéos.


###  Fichier de Labels

Le **fichier de labels** contient les annotations des émotions pour chaque vidéo, accompagnées des identifiants des vidéos et des segments correspondants. Ce fichier permet d'aligner les données multimodales (texte, audio, vidéo) sur la même échelle d'annotation pour pouvoir entraîner un modèle de prédiction des émotions.

#### Structure du fichier `label.csv`

Le fichier `label.csv` contient les colonnes suivantes :
- **video_id** : Identifiant unique de la vidéo.
- **clip_id** : Identifiant du segment vidéo (peut correspondre à une portion spécifique de la vidéo).
- **text** : La transcription textuelle du segment vidéo.
- **label** : Score d'émotion allant de **-3 à 3**, représentant l'intensité de l'émotion exprimée dans le texte de la vidéo.
- **label_T**, **label_A**, **label_V** : Étiquettes spécifiques pour les données de texte, audio et vidéo, respectivement. Ces étiquettes sont utilisées pour l'alignement des données multimodales.
- **annotation** : Annotation de l'émotion, catégorisée comme **Positif**, **Négatif** ou **Neutre**.
- **mode** : Le mode de l'échantillon, typiquement **train** ou **test**, pour l'assignation à l'ensemble d'entraînement ou de test.


# **Prétraitement**
## **Traitement du video**
Pour l'analyse des expressions faciales, nous utilisons OpenFace pour extraire les unités d'action faciale à partir des images extraites des vidéos.

Pour exécuter OpenFace, téléchargez et exécutez le logiciel :
Téléchargez OpenFace depuis ici :https://sourceforge.net/projects/openface.mirror/

Extrayez et exécutez OpenFace dans le shell.
```
# Définir les chemins
$exePath = "E:\OpenFace_2.2.0_win_x86\OpenFace_2.2.0_win_x86\FeatureExtraction.exe"
$videoFolder = "E:\Raw\Video\Segmented"  # Répertoire contenant les vidéos
$outDir = "E:\OpenFace_Results\processed"  # Répertoire de sortie

# Créer le répertoire de sortie s'il n'existe pas déjà
if (-not (Test-Path -Path $outDir)) {
    New-Item -Path $outDir -ItemType Directory
}

# Récupérer tous les fichiers vidéo (formats .mp4, .avi, .mov)
$videoFiles = Get-ChildItem -Path $videoFolder -Recurse -File | Where-Object { $_.Extension -in @(".mp4", ".avi", ".mov") }

# Boucle sur chaque vidéo et exécution de FeatureExtraction
foreach ($video in $videoFiles) {
    $videoPath = $video.FullName
    Write-Host "Processing: $videoPath"
    
    # Exécuter FeatureExtraction sur chaque vidéo
    & $exePath -f $videoPath -out_dir $outDir
}

Write-Host "Traitement terminé !"
```
À partir des fichiers .csv générés par OpenFace (une ligne par frame, une colonne par AU), nous :


1.   détectons les Action Units (AUs) actives,
2.   sélectionnons les AUs les plus importantes,
3.   convertissons ces AUs en texte lisible,
4.   générons un identifiant unique pour chaque segment,
5.   sauvegardons l’ensemble dans un fichier vision_text.pkl.

Ce fichier sera ensuite utilisé dans la modalité Vision de notre modèle multimodal.

## **Traitement du audio**
Dans ce projet, nous extrayons l'audio des vidéos à l'aide de **MoviePy**, une bibliothèque Python qui permet de traiter et de manipuler les fichiers multimédia. Ensuite, nous utilisons **OpenSMILE**, un outil de traitement audio, pour extraire plusieurs caractéristiques importantes des fichiers audio, notamment :

- **Loudness** : Mesure de l'intensité du son.
- **Jitter** : Variation de la fréquence fondamentale.
- **Shimmer** : Variation de l'amplitude du signal audio.
- **F0 (Fréquence fondamentale)** : Valeur de la fréquence fondamentale du signal.

Ces caractéristiques sont ensuite traitées et classées en trois niveaux (faible, normal, élevé) pour chaque dimension sonore. Ces descriptions sont ensuite converties en texte pour chaque vidéo. Et puis on génère l'id pour chaque segment

Le code pour cette extraction et transformation des caractéristiques audio est présent dans le notebook **`analyse_sentiment_multiomodal.ipynb`**, où chaque étape est détaillée et appliquée aux fichiers audio des vidéos.

## **Traitement du Texte**
Nous traitons le texte avec BERT, en utilisant des modèles pré-entraînés pour la tokenisation et la génération des embeddings. Ces embeddings sont utilisés pour prédire le sentiment du texte.

1. **Chargement des données**
   - Nous chargeons les fichiers **.pkl** contenant les textes bruts associés à chaque segment vidéo, audio ou visuel et les données textuel du dataset MOSI
   - Les fichiers pickle audio et video contiennent les textes associés aux segments audio et visuel respectivement.
 

2. **Tokenisation**
   - Les textes sont tokenisés à l’aide du **tokenizer BERT** (modèle pré-entrainé `bert-base-uncased`).
   - Chaque texte est transformé en une séquence de tokens compatible avec BERT, avec un maximum de **128 tokens**.

3. **Encodage avec BERT**
   - Le texte tokenisé est ensuite traité par le modèle **BERT**, qui génère des **embeddings** de taille 768 pour chaque token.
   - Ces embeddings sont ensuite utilisés dans le **TextEncoder**.

4. **Application du TextEncoder**
   - Le **TextEncoder** est appliqué sur les embeddings de BERT. Il ajoute un token spécial **Eₘ** pour indiquer le début de la séquence de la modalité.
   - La sortie du **TextEncoder** est une séquence de **8 tokens** représentée par un vecteur de taille **768**.
   - **Seuls les 8 premiers tokens** générés par le **TextEncoder** sont utilisés pour l'entraînement, représentant ainsi les informations clés de chaque séquence.

5. **Fusion des embeddings**
   - Après avoir généré les embeddings pour chaque modalité (texte, audio et vidéo), les embeddings sont fusionnés pour être utilisés dans le modèle multimodal.
  
   - ## **Entraînement du Modèle**
   ### Chargement des données

Les données utilisées dans ce projet sont stockées sous forme de fichiers `.pkl` pour les embeddings des trois modalités : texte, audio et vidéo. Ces fichiers sont chargés à l'aide de la fonction `load_pkl()` qui lit les fichiers `.pkl` et récupère les embeddings ainsi que les identifiants (IDs) associés.

Les données sont ensuite extraites et préparées pour l'entraînement du modèle multimodal :

- **Texte** : Les embeddings textuels sont chargés .
- **Audio** : Les embeddings audio sont chargés `.
- **Vidéo** : Les embeddings vidéo sont chargés `.

Les identifiants (IDs) associés à chaque modalité sont également extraits et utilisés pour l'alignement avec les IDs du fichier CSV des labels.

### Alignement des données multimodales

Les embeddings de chaque modalité (texte, audio et vidéo) sont alignés avec les IDs correspondants à partir du fichier CSV des labels. Un **projecteur d'embeddings** est utilisé pour transformer chaque ensemble d'embeddings en séquences alignées, permettant ainsi d'avoir des données cohérentes pour l'entraînement du modèle.

La classe `AudioVisualFeatureProjector` permet d'aligner les embeddings audio, vidéo et texte en utilisant les IDs du CSV. Les données sont ensuite alignées avec les IDs du CSV, garantissant que chaque échantillon de données multimodales (texte, audio, vidéo) correspond à une étiquette spécifique.

### Création du Dataset pour l'entraînement

Un dataset multimodal est créé en combinant les embeddings alignés de chaque modalité avec les étiquettes issues du fichier CSV. Ce dataset est utilisé pour entraîner, valider et tester le modèle.

Les données sont divisées en trois ensembles :

- **Ensemble d'entraînement** : 70% des données
- **Ensemble de validation** : 15% des données
- **Ensemble de test** : 15% des données

La division est réalisée à l'aide de la fonction `train_test_split()` de Scikit-learn, permettant ainsi de séparer les données de manière aléatoire tout en préservant la distribution des labels.

### Statistiques finales sur les données

Après avoir préparé les données, nous vérifions les dimensions des embeddings et leur distribution sur les ensembles d'entraînement, de validation et de test. Les dimensions des embeddings sont affichées, et la répartition des labels (positifs et négatifs) est également vérifiée pour chaque ensemble.

Les résultats des statistiques finales sont les suivants :

- Nombre d'échantillons dans chaque ensemble (train, val, test).
- Dimensions des embeddings pour chaque modalité (texte, audio, vidéo).
- Distribution des labels dans chaque ensemble.


# Définir les chemins
$exePath = "E:\OpenFace_2.2.0_win_x86\OpenFace_2.2.0_win_x86\FeatureExtraction.exe"
$videoFolder = "E:\Raw\Video\Segmented"  # Répertoire contenant les vidéos
$outDir = "E:\OpenFace_Results\processed"  # Répertoire de sortie

# Créer le répertoire de sortie s'il n'existe pas déjà
if (-not (Test-Path -Path $outDir)) {
    New-Item -Path $outDir -ItemType Directory
}

# Récupérer tous les fichiers vidéo (formats .mp4, .avi, .mov)
$videoFiles = Get-ChildItem -Path $videoFolder -Recurse -File | Where-Object { $_.Extension -in @(".mp4", ".avi", ".mov") }

# Boucle sur chaque vidéo et exécution de FeatureExtraction
foreach ($video in $videoFiles) {
    $videoPath = $video.FullName
    Write-Host "Processing: $videoPath"
    
    # Exécuter FeatureExtraction sur chaque vidéo
    & $exePath -f $videoPath -out_dir $outDir
}

Write-Host "Traitement terminé !"

À partir des fichiers .csv générés par OpenFace (une ligne par frame, une colonne par AU), nous :


1.   détectons les Action Units (AUs) actives,
2.   sélectionnons les AUs les plus importantes,
3.   convertissons ces AUs en texte lisible,
4.   générons un identifiant unique pour chaque segment,
5.   sauvegardons l’ensemble dans un fichier vision_text.pkl.

Ce fichier sera ensuite utilisé dans la modalité Vision de notre modèle multimodal.

## **Traitement du audio**

Dans ce projet, nous extrayons l'audio des vidéos à l'aide de **MoviePy**, une bibliothèque Python qui permet de traiter et de manipuler les fichiers multimédia. Ensuite, nous utilisons **OpenSMILE**, un outil de traitement audio, pour extraire plusieurs caractéristiques importantes des fichiers audio, notamment :

- **Loudness** : Mesure de l'intensité du son.
- **Jitter** : Variation de la fréquence fondamentale.
- **Shimmer** : Variation de l'amplitude du signal audio.
- **F0 (Fréquence fondamentale)** : Valeur de la fréquence fondamentale du signal.

Ces caractéristiques sont ensuite traitées et classées en trois niveaux (faible, normal, élevé) pour chaque dimension sonore. Ces descriptions sont ensuite converties en texte pour chaque vidéo. Et puis on génère l'id pour chaque segment

Le code pour cette extraction et transformation des caractéristiques audio est présent dans le notebook **`analyse_sentiment_multiomodal.ipynb`**, où chaque étape est détaillée et appliquée aux fichiers audio des vidéos.

## **Traitement du Texte**
Nous traitons le texte avec BERT, en utilisant des modèles pré-entraînés pour la tokenisation et la génération des embeddings. Ces embeddings sont utilisés pour prédire le sentiment du texte.

### Chargement des données

Les données utilisées dans ce projet sont stockées sous forme de fichiers `.pkl` pour les embeddings des trois modalités : texte, audio et vidéo. Ces fichiers sont chargés à l'aide de la fonction `load_pkl()` qui lit les fichiers `.pkl` et récupère les embeddings ainsi que les identifiants (IDs) associés.

Les données sont ensuite extraites et préparées pour l'entraînement du modèle multimodal :

- **Texte** : Les embeddings textuels sont chargés .
- **Audio** : Les embeddings audio sont chargés `.
- **Vidéo** : Les embeddings vidéo sont chargés `.

Les identifiants (IDs) associés à chaque modalité sont également extraits et utilisés pour l'alignement avec les IDs du fichier CSV des labels.

### Alignement des données multimodales

Les embeddings de chaque modalité (texte, audio et vidéo) sont alignés avec les IDs correspondants à partir du fichier CSV des labels. Un **projecteur d'embeddings** est utilisé pour transformer chaque ensemble d'embeddings en séquences alignées, permettant ainsi d'avoir des données cohérentes pour l'entraînement du modèle.

La classe `AudioVisualFeatureProjector` permet d'aligner les embeddings audio, vidéo et texte en utilisant les IDs du CSV. Les données sont ensuite alignées avec les IDs du CSV, garantissant que chaque échantillon de données multimodales (texte, audio, vidéo) correspond à une étiquette spécifique.

### Création du Dataset pour l'entraînement

Un dataset multimodal est créé en combinant les embeddings alignés de chaque modalité avec les étiquettes issues du fichier CSV. Ce dataset est utilisé pour entraîner, valider et tester le modèle.

Les données sont divisées en trois ensembles :

- **Ensemble d'entraînement** : 70% des données
- **Ensemble de validation** : 15% des données
- **Ensemble de test** : 15% des données

La division est réalisée à l'aide de la fonction `train_test_split()` de Scikit-learn, permettant ainsi de séparer les données de manière aléatoire tout en préservant la distribution des labels.

### Statistiques finales sur les données

Après avoir préparé les données, nous vérifions les dimensions des embeddings et leur distribution sur les ensembles d'entraînement, de validation et de test. Les dimensions des embeddings sont affichées, et la répartition des labels (positifs et négatifs) est également vérifiée pour chaque ensemble.

Les résultats des statistiques finales sont les suivants :

- Nombre d'échantillons dans chaque ensemble (train, val, test).
- Dimensions des embeddings pour chaque modalité (texte, audio, vidéo).
- Distribution des labels dans chaque ensemble.

## 🔥 Entraînement du Modèle Multimodal (DEVANet)

Cette section décrit comment le modèle multimodal a été entraîné, régularisé et évalué après l’alignement des embeddings texte–audio–vidéo.

---

### 1️⃣ Normalisation des données

Avant l’entraînement, les embeddings de chaque modalité (Texte, Audio, Vidéo) sont **normalisés** en utilisant :

- la **moyenne** et l’**écart-type** des données d’entraînement uniquement  
- une normalisation appliquée ensuite aux ensembles **train**, **validation** et **test**

Cette étape stabilise l’entraînement et permet au modèle de converger plus rapidement.

---

### 2️⃣ Dataset avec augmentation

Pour rendre le modèle plus robuste, une **augmentation légère** est appliquée pendant l’entraînement :

- ajout d’un bruit gaussien aux embeddings (- texte, audio, vidéo -)
- probabilité de 50%
- standard deviation du bruit = **0.05**

➡️ Cela simule des variations naturelles (bruit audio, micro-expression instable, variation textuelle).

---

### 3️⃣ Architecture : Cross-Modal Attention

Le cœur du modèle repose sur une **attention croisée robuste** qui permet au texte d’aller chercher des informations pertinentes dans :

- les embeddings **audio**
- les embeddings **vidéo**

L’architecture utilisée comprend :

#### 🔹 RobustCrossModalAttention  
Un module d'attention qui calcule :
- Query (texte)
- Keys/Values (audio ou vidéo)
- Matrice d’attention + dropout

#### 🔹 SimplifiedMFU  
(Multimodal Fusion Unit simplifiée)

- effectue une double attention croisée T→A et T→V  
- réalise un **pooling temporel** sur les séquences  
- concatène les modalités texte/audio/vidéo  
- applique une couche fully connected + LayerNorm

#### 🔹 DEVANet Régularisé

Le modèle final contient :

- **MFU** → fusion multimodale
- **Classifier** → MLP (2 couches) qui prédit le score d’émotion (-3 à 3)

Régularisation utilisée :
- Dropout = **0.4**
- Weight decay = **1e-3**
- Gradient clipping = **0.3**

---

### 4️⃣ Fonction de perte hybride

Nous utilisons une **loss hybride** spécialement conçue pour les labels MOSI :

#### 🔸 MSE Loss  
Pour l'aspect continu : prédiction du score d’émotion (-3 → 3)

#### 🔸 BCE With Logits + Label Smoothing  
Pour la classification binaire :

- score > 0  → **positif**
- score ≤ 0 → **négatif**

Label smoothing = **0.1**

➡️ Cela stabilise l’apprentissage lorsque les labels sont bruités.

La loss finale :  
**0.5 × MSE + 0.5 × BCE_smooth**

---

### 5️⃣ Métriques d’évaluation

Comme MOSI est un dataset **continu**, mais souvent évalué en binaire, nous utilisons :

| Métrique | Description |
|---------|-------------|
| **Acc-2** | Accuracy binaire (score > 0 vs score ≤ 0) |
| **F1-Weighted** | Évalue l’équilibre Positif/Négatif |
| **MAE** | Mesure l’erreur absolue sur les scores (-3 → 3) |
| **Pearson Correlation** | Corrélation entre prédictions et labels MOSI |

---

### 6️⃣ Entraînement

Hyperparamètres clés :

- Optimizer : **AdamW**
- LR : **5e-5**
- Scheduler : **Cosine Annealing**
- Epochs : **80**
- Patience : **20**
- Batch size : **16**

L’entraînement inclut un **early stopping**, basé sur la meilleure Acc-2 en validation.

---

### 7️⃣ Évaluation finale et sauvegarde

À la fin de l’entraînement :

- Le meilleur modèle (selon **Acc-2** validation) est chargé
- Les performances sont évaluées sur le **test set**
- Le modèle final est sauvegardé sous :



les poids du modèle
les métriques test
les statistiques de normalisation
la configuration du modèle
➡️ nécessaire pour une inférence cohérente
Double-cliquez (ou appuyez sur Entrée) pour modifier

## 📊 Résultats — Modèle Baseline (BERT-base-uncased)

Nous avons entraîné un premier modèle **baseline** en utilisant les embeddings texte provenant de  
**BERT-base-uncased**, combinés avec les embeddings audio (OpenSMILE) et vidéo (OpenFace).  
Ce modèle utilise notre version simplifiée de **DEVANet** avec attention croisée multimodale.

L’entraînement s’est arrêté automatiquement grâce à l’**early stopping** à l’epoch 21.

### 🔥 Performances finales sur le Test Set

| Metric | Score |
|--------|--------|
| **Acc-2 (Binary)** | **0.8273** |
| **F1-weighted** | **0.8273** |
| **MAE** | **0.9959** |
| **Pearson Correlation** | **0.7374** |

➡️ **Acc-2** et **F1-Weighted** au-dessus de **82%**.

### 📌 Observations importantes

- Le modèle apprend rapidement, atteignant une précision binaire de **94%** sur le train set avant régularisation.
- Les résultats en validation tournent autour de **0.73–0.75**, ce qui est cohérent avec MOSI.
- Le test set montre une bonne généralisation (Acc-2 ≈ 0.8273).
- Le **MAE ≈ 0.99** montre que le modèle reste stable pour de la régression émotionnelle continue.
- La **corrélation de Pearson ≈ 0.74** indique une bonne cohérence entre labels réels et prédictions.

                   +--------------------------------+
                   |   1. Téléchargement des données |
                   |     CMU-MOSI (vidéos, audio,    |
                   |     transcriptions, labels)     |
                   +--------------------------------+
                                   |
                                   v
        +--------------------------------------------------------+
        | 2. Extraction & Encodage des caractéristiques          |
        |                                                        |
        |  🔹 VISUEL : OpenFace                                  |
        |      → landmarks, Action Units, embeddings             |
        |                                                        |
        |  🔹 AUDIO : OpenSMILE + encodage audio                 |
        |      → F0, jitter, shimmer, loudness, MFCC, etc.       |
        |      → passage dans un encodeur pour obtenir           |
        |        un embedding audio fixe                         |
        |                                                        |
        |  🔹 TEXTE : BERT-base-uncased                          |
        |      → embeddings textuels 768d                        |
        +--------------------------------------------------------+
                                   |
                                   v
        +--------------------------------------------------------+
        |              3. Alignement temporel MOSI               |
        |  - Alignement audio/texte/vidéo                        |
        |  - Segments synchronisés de longueur T=8               |
        +--------------------------------------------------------+
                                   |
                                   v
        +--------------------------------------------------------+
        |            4. Prétraitement & Normalisation            |
        |  - Normalisation séparée par modalité (T, A, V)        |
        |  - Encodage du label :                                |
        |       • Valeur continue : [-3, 3]                      |
        |       • Label binaire : (score > 0)                    |
        |       • Label tri-class (Pos / Neg / Neutre)           |
        |  - Construction des DataLoaders                        |
        +--------------------------------------------------------+
                                   |
                                   v
        +--------------------------------------------------------+
        |      5. Fusion Multimodale (MFU – Baseline)           |
        |  - Attention croisée T→A et T→V                        |
        |  - Pooling temporel (moyenne)                          |
        |  - Fusion T + A + V                                    |
        +--------------------------------------------------------+
                                   |
                                   v
        +--------------------------------------------------------+
        |                  6. Modèle Baseline                    |
        |  - DEVANet (couche dense)                              |
        |  - Sortie : prédiction continue ∈ [-3, 3]              |
        +--------------------------------------------------------+
                                   |
                                   v
        +--------------------------------------------------------+
        |                 7. Entraînement                         |
        |  - HybridLoss (MSE + BCE binaire)                      |
        |  - AdamW + Early Stopping                              |
        |  - Suivi des métriques de validation                   |
        +--------------------------------------------------------+
                                   |
                                   v
        +--------------------------------------------------------+
        |           8. Prédiction & Évaluation finale            |
        |  - Prédiction : score continu + classe binaire         |
        |  - Métriques : Acc-2, F1-weighted, MAE, Pearson        |
        +--------------------------------------------------------+

  
## 🔁 Variante RoBERTa (expérimentation supplémentaire)

En complément de la baseline avec **BERT-base-uncased**, nous avons testé une variante où :

- Les descriptions audio/vidéo (prompts générés à partir de loudness, jitter, shimmer, F0, etc.)  
  sont encodées avec **RoBERTa-base** au lieu de BERT.
- Un **TextEncoder transformer** (T = 8, d = 768) projette ces sorties en blocs de taille fixe.
- Les embeddings texte, audio et vidéo sont ensuite alignés avec le fichier `label.csv`
  et injectés dans le même modèle **RegularizedDEVANet** (cross-modal attention + fusion).

L’entraînement et l’évaluation sont identiques à la baseline (même split MOSI, mêmes métriques).

### 📊 Résultats (RoBERTa)

Sur le **test set**, nous obtenons environ :

- **Acc-2 (Binary)** ≈ **0.74**
- (autres métriques dans le notebook d’entraînement)

Ces résultats sont **inférieurs** à ceux de la baseline BERT-base-uncased  
(Acc-2 ≈ 0.83), donc **la baseline BERT** reste notre modèle de référence officiel.

> 💡 Le code complet de cette variante RoBERTa (encodage + alignement + entraînement)  
> est disponible dans le notebook du projet.

### 1. Meilleur encodeur texte

- Baseline : `bert-base-uncased` (BERT généraliste).
- Modèle optimisé : **`ayoubkirouane/BERT-Emotions-Classifier`**, un BERT pré-entraîné spécifiquement pour la classification des émotions.
- Objectif : obtenir des embeddings textuels plus discriminants pour la valence.

### 2. Attention croisée multi-tête stabilisée

- Baseline : attention croisée simple (une seule tête, sans normalisation).
- Modèle optimisé :
  - **Multi-Head Cross-Modal Attention** (4 têtes) entre texte–audio et texte–vidéo.
  - **Residual connection + LayerNorm** dans le bloc d’attention.
- Objectif : mieux capturer les interactions fines entre modalités et stabiliser l’entraînement.

### 3. MFU (fusion multimodale) amélioré

- Baseline : pooling temporel par moyenne seule, puis concaténation et projection.
- Modèle optimisé :
  - Pooling **moyenne + max** pour chaque modalité (texte, audio, vidéo).
  - Fusion via un bloc linéaire + ReLU + LayerNorm.
- Objectif : garder à la fois la tendance globale et les pics émotionnels dans chaque séquence.

### 4. Fonction de perte hybride réajustée

- Baseline :
  - `MSELoss` + `BCEWithLogitsLoss` avec pondération 50% / 50%.
- Modèle optimisé :
  - **`SmoothL1Loss` (Huber)** pour la partie régression (valence continue).
  - `BCEWithLogitsLoss` avec **label smoothing** pour la partie binaire.
  - Pondération **30% MSE / 70% BCE**.
- Objectif : mieux équilibrer la régression de la valence et la classification binaire (Acc-2), tout en rendant la perte moins sensible aux outliers.

### 5. Optimisation & entraînement

- Optimizer : toujours **AdamW**, mais avec :
  - `lr = 2e-5` (plus stable),
  - `weight_decay = 5e-4`,
  - `amsgrad=True` pour une meilleure convergence.
- Scheduler :
  - Baseline : `CosineAnnealingLR`.
  - Modèle optimisé : **`CosineAnnealingWarmRestarts`** pour mieux explorer l’espace de paramètres.
- Entraînement :
  - **AMP (mixed precision)** avec `torch.cuda.amp` pour accélérer l’entraînement et améliorer la stabilité numérique.
  - **Early stopping** plus agressif (`patience = 12`) pour limiter l’overfitting.

### 6. Normalisation & réplicabilité

- Normalisation systématique des embeddings texte/audio/vidéo à partir des statistiques du train.
- Fixation d’un **seed global (42)** pour PyTorch, NumPy et Python, afin de garantir la réplicabilité des résultats.

## 📊 Résultats finaux sur le Test Set

Après l’entraînement du modèle DEVANet optimisé et la sélection du meilleur checkpoint
(basé sur la métrique Acc-2 en validation), nous avons évalué les performances sur le
jeu de test MOSI.

### 🧪 Métriques obtenues

| Métrique                 | Valeur |
|--------------------------|--------|
| **Acc-2** (accuracy binaire) | **0.8363** |
| **F1-weighted**          | **0.8363** |
| **MAE** (erreur absolue moyenne) | **0.7624** |
| **Corrélation de Pearson** | **0.7757** |

### ✅ Interprétation des résultats

- **Acc-2 = 83.63%** → Le modèle discrimine efficacement les sentiments *positifs vs négatifs*.  
- **F1-weighted ≈ 0.836** → Les performances sont équilibrées malgré le déséquilibre de classes.  
- **MAE ≈ 0.76** → L’erreur moyenne entre la prédiction de sentiment continu et la vérité terrain reste faible.  
- **Corrélation de Pearson ≈ 0.776** → Le modèle suit bien la tendance de l’intensité émotionnelle réelle.

Ces résultats montrent que **notre version optimisée de DEVANet** (attention multi-têtes, pooling amélioré, hybrid loss ajustée) obtient de meilleures performances que notre baseline.


               +--------------------------------+
               |   1. Téléchargement des données |
               |     CMU-MOSI (vidéo, audio,     |
               |     texte, labels [-3,3])       |
               +--------------------------------+
                               |
                               v
    +----------------------------------------------------------+
    | 2. Extraction & Encodage des caractéristiques             |
    |                                                          |
    |  🔹 VISUEL : OpenFace                                    |
    |      → AU (Action Units), landmarks, embeddings          |
    |                                                          |
    |  🔹 AUDIO : OpenSMILE + encodeur audio                   |
    |      → MFCC, loudness, jitter, shimmer, F0               |
    |      → encodage → vecteur audio fixe                     |
    |                                                          |
    |  🔹 TEXTE : BERT amélioré                                 |
    |      → Modèle utilisé :                                  |
    |            **"ayoubkirouane/BERT-Emotions-Classifier"**  |
    |      → embeddings émotionnels optimisés (768d)           |
    +----------------------------------------------------------+
                               |
                               v
    +----------------------------------------------------------+
    |            3. Alignement temporel (MOSi segmenté)         |
    |  - Alignement Texte / Audio / Vidéo                       |
    |  - Fenêtres synchronisées T = 8 frames                    |
    +----------------------------------------------------------+
                               |
                               v
    +----------------------------------------------------------+
    |    4. Prétraitement & Normalisation multimodale          |
    |  - Normalisation indépendante (texte, audio, vidéo)      |
    |  - Augmentation légère : ajout de bruit                  |
    |  - Encodage du label :                                   |
    |        • Valeur continue ∈ [-3,3]                        |
    |        • Binaire : (label > 0)                           |
    |        • Tri-class (Pos / Neg / Neu)                     |
    +----------------------------------------------------------+
                               |
                               v
    +----------------------------------------------------------+
    |     5. Fusion Multimodale Améliorée (Optimized MFU)     |
    |                                                          |
    |  🔸 Multi-Head Cross-Modal Attention                      |
    |       • 4 têtes parallèle pour chaque modalité           |
    |       • T → A et T → V                                   |
    |                                                          |
    |  🔸 Pooling avancé : mean + max                          |
    |       → meilleure capture des pics émotionnels           |
    |                                                          |
    |  🔸 Fusion : concat(T, A, V) → couche dense              |
    +----------------------------------------------------------+
                               |
                               v
    +----------------------------------------------------------+
    |             6. Modèle Optimisé : DEVANetOptim             |
    |  - MFU amélioré + classifier profond                     |
    |  - Sortie : score émotionnel continu ∈ [-3,3]            |
    +----------------------------------------------------------+
                               |
                               v
    +----------------------------------------------------------+
    |         7. Entraînement Optimisé                         |
    |  🔹 Loss hybrides améliorée                               |
    |       • SmoothL1 (MSE robuste)                            |
    |       • BCE avec label smoothing                          |
    |       • pondération (0.3/0.7) optimisée                   |
    |                                                          |
    |  🔹 Optimisateur : AdamW + amsgrad                        |
    |  🔹 Mixed precision : AMP (autocast + GradScaler)         |
    |  🔹 Scheduler : CosineAnnealingWarmRestarts               |
    |  🔹 Gradient clipping : 0.5                               |
    +----------------------------------------------------------+
                               |
                               v
    +----------------------------------------------------------+
    |         8. Évaluation & Sauvegarde du meilleur modèle    |
    |  - Métriques : Acc-2, F1-weighted, MAE, Pearson          |
    |  - Early stopping                                         |
    |  - Sauvegarde : devanet_optimized_final.pth               |
    |    → inclut normalisation + métriques + poids            |
    +----------------------------------------------------------+
