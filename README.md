# Détecteur de Comportements Suspects : Rapport Technique

## Résumé Exécutif

Ce projet implémente un système de surveillance en temps réel capable de détecter et d'analyser des comportements suspects chez les personnes à partir d'un flux vidéo. Le système utilise la détection de personnes basée sur YOLOv4-tiny combinée à un suivi centroïde et à des heuristiques comportementales sophistiquées. L'architecture est optimisée pour s'exécuter sur des ressources CPU limitées, ce qui la rend particulièrement adaptée aux déploiements sur Raspberry Pi.

---

## 1. Introduction et Contexte

### 1.1 Objectifs du Projet

Le système vise à :
- **Détecter en temps réel** la présence de personnes dans un flux vidéo à partir d'une webcam ou d'un module caméra
- **Suivre individuellement** chaque personne détectée tout au long de la séquence vidéo
- **Analyser les comportements** à travers des heuristiques basées sur la cinématique et la géométrie
- **Générer des alertes** pour les comportements anormaux ou suspects
- **Journaliser les événements** avec suppression intelligente des doublons pour éviter la saturation des logs

### 1.2 Architecture Générale

Le système repose sur trois piliers techniques :

1. **Détection d'Objets** : YOLOv4-tiny via OpenCV DNN
2. **Suivi Multi-Objets** : Algorithme de suivi centroïde simplifié
3. **Analyse Comportementale** : Heuristiques basées sur l'accélération, la stase et la géométrie

---

## 2. Architecture Technique

### 2.1 Composants Principaux

```
suspicious_behavior.py      [Script principal]
    ├── Chargement du modèle YOLOv4-tiny
    ├── Capture vidéo (webcam ou fichier)
    ├── Boucle de traitement temps réel
    │   ├── Inférence YOLO
    │   ├── Mise à jour du suivi
    │   └── Analyse comportementale
    └── Visualisation et journalisation
```

**Fichiers du Projet :**

| Fichier | Description |
|---------|-------------|
| `suspicious_behavior.py` | Script principal (~450 lignes) |
| `requirements.txt` | Dépendances Python |
| `models/` | Répertoire contenant les fichiers du modèle (téléchargés automatiquement) |
| `logs/suspicious.log` | Journaux des alertes détectées |

### 2.2 Dépendances

```
opencv-python-headless==4.6.0.66    # Traitement vidéo et inférence DNN
numpy>=1.21                          # Calculs numériques et opérations matricielles
requests>=2.28                       # Téléchargement automatique des modèles
```

### 2.3 Fichiers de Modèle

Le système utilise **YOLOv4-tiny**, une version légère du détecteur YOLO :

| Fichier | Taille | Rôle |
|---------|--------|------|
| `yolov4-tiny.weights` | ~23 MB | Poids du réseau de neurones pré-entraîné |
| `yolov4-tiny.cfg` | ~16 KB | Architecture du réseau (configuration) |
| `coco.names` | ~5 KB | Libellés des 80 classes COCO (dont "person") |

Les fichiers sont téléchargés automatiquement lors de la première exécution.

---

## 3. Détection et Suivi

### 3.1 Processus de Détection

Pour chaque image du flux vidéo :

1. **Préparation du blob** : Redimensionnement en 416×416 pixels, normalisation des valeurs de pixels
2. **Inférence** : Passage du blob dans le réseau YOLOv4-tiny
3. **Post-traitement** :
   - Extraction des boîtes englobantes pour la classe "person"
   - Seuillage par confiance (`CONF_THRESHOLD = 0.4`)
   - Suppression des chevauchements (NMS) avec seuil `NMS_THRESHOLD = 0.4`

### 3.2 Suivi Centroïde

Un algorithme de suivi centroïde simple mais efficace associe les détections d'une image à l'autre :

**Principes :**
- Chaque personne détectée est représentée par son centroïde (centre de sa boîte englobante)
- Les centroïdes sont appariés entre images successives en minimisant la distance euclidienne
- Les appariements au-delà d'une distance seuil (`MAX_DISTANCE = 80 px`) créent une nouvelle trace

**Historique :**
- Chaque objet suivi conserve un historique des 32 derniers centroïdes avec leurs timestamps et dimensions
- Cet historique permet le calcul de vélocité, accélération et analyse temporelle

---

## 4. Détection des Comportements Suspects

Le système analyse quatre catégories de comportements suspects. Pour chacune, nous présentons la méthode de détection, les paramètres de configuration et l'interprétation.

### 4.1 Accélération Soudaine

**Description :** Détecte les mouvements rapides et imprévisibles, potentiellement indicatifs de panique ou de menace.

**Méthode de Détection :**

L'accélération est calculée à partir de l'historique des centroïdes :

$$a(t) = \frac{\|\vec{v}(t) - \vec{v}(t-\Delta t)\|}{\Delta t}$$

où $\vec{v}(t)$ est la vélocité instantanée (en pixels/seconde) et $\Delta t$ l'intervalle entre mesures.

**Étapes Algorithmiques :**
1. Extraction de l'historique des 4 derniers centroïdes
2. Calcul de deux vélocités successives à partir des paires de centroïdes
3. Mesure de la variation de vélocité rapportée au temps écoulé
4. Comparaison au seuil d'accélération

**Paramètres Configurables :**
```python
SUDDEN_ACCELERATION_THRESHOLD = 1500.0  # pixels/s²
```

**Interprétation :** Une accélération > 1500 px/s² sur une caméra positionnée à 2 mètres de hauteur correspond à approximativement 0.5–0.8 g (accélération gravitationnelle), signalant une action très brusque.

---

### 4.2 Immobilité Prolongée (Stase)

**Description :** Détecte les personnes qui restent stationnaires pendant une période anormalement longue, pouvant indiquer un malaise, une chute ou un refus de bouger.

**Méthode de Détection :**

Le système mesure la durée durant laquelle une personne reste dans un rayon spatial limité :

$$\text{stillness\_time} = \max(t : \forall \tau \in [t-T, t], \|\Delta p(\tau)\| < d_{\text{seuil}})$$

où :
- $T$ est la fenêtre temporelle d'observation
- $\Delta p(\tau)$ est le déplacement spatial entre deux frames
- $d_{\text{seuil}}$ est la distance maximale tolérée pour considérer la personne immobile

**Étapes Algorithmiques :**
1. Parcours rétroactif de l'historique depuis la frame courante
2. Pour chaque intervalle entre centroïdes successifs, calcul de la distance parcourue
3. Arrêt du parcours si une distance > seuil est détectée
4. Accumulation du temps depuis le début du parcours

**Paramètres Configurables :**
```python
STILLNESS_TIME_THRESHOLD = 8.0          # secondes
STILLNESS_DISTANCE_THRESHOLD = 10.0     # pixels
```

**Interprétation :** Une personne restant immobile (< 10 pixels de mouvement) pendant 8 secondes est signalée.

---

### 4.3 Détection de Chute

**Description :** Identifie les chutes ou pertes d'équilibre, parmi les événements les plus critiques en surveillance de sécurité.

**Méthode de Détection :**

La détection de chute repose sur trois critères complémentaires :

#### Critère 1 : Descente Rapide Verticale

$$v_y > v_{\text{seuil}} \quad \text{(descendant)}$$

où $v_y$ est la composante verticale de la vélocité (positive vers le bas).

```python
FALL_DESCENT_SPEED = 200.0  # pixels/s
```

#### Critère 2 : Réduction de Hauteur

Comparaison du rapport de hauteur entre la détection courante et celle d'environ 1 seconde auparavant :

$$\text{height\_ratio} = \frac{h(t)}{h(t-1s)} < \text{FALL\_HEIGHT\_RATIO}$$

```python
FALL_HEIGHT_RATIO = 0.6  # La personne descend à 60% de sa hauteur antérieure
```

#### Critère 3 : Orientation Horizontale

Détection d'une boîte englobante avec un rapport largeur/hauteur anormal :

$$\text{aspect\_ratio} = \frac{w}{h} > \text{ASPECT\_RATIO\_THRESHOLD}$$

```python
ASPECT_RATIO_THRESHOLD = 1.2
```

**Confirmation par Immobilité :**

Pour éviter les faux positifs, une chute est confirmée seulement après que la personne soit restée immobile pendant :

```python
FALL_STILLNESS_TIME = 3.0  # secondes
```

**Processus Complet :**
1. Vérifier si $v_y > 200$ px/s ET $\text{height\_ratio} < 0.6$, OU rapport hauteur/largeur > 1.2
2. Attendre que la personne soit immobile pendant 3 secondes
3. Si confirmed, générer alerte "FALL"

---

### 4.4 Entrée en Zone Restreinte

**Description :** Déclenche une alerte lorsqu'une personne pénètre dans une région spatiale définie (par ex., zone interdite, bureau privé).

**Méthode de Détection :**

La zone restreinte est définie comme un rectangle en coordonnées relatives (fractions du cadre vidéo) :

$$\text{RESTRICTED\_ZONE} = (x_1, y_1, x_2, y_2) \in [0,1]^4$$

À chaque détection, le centroïde $(c_x, c_y)$ en pixels est comparé à la zone convertie :

$$\text{alert} \iff \left( \frac{x_1 \cdot W}{1} \leq c_x \leq \frac{x_2 \cdot W}{1} \right) \land \left( \frac{y_1 \cdot H}{1} \leq c_y \leq \frac{y_2 \cdot H}{1} \right)$$

où $W$ et $H$ sont les dimensions du cadre vidéo.

**Paramètres Configurables :**
```python
RESTRICTED_ZONE = (0.3, 0.6, 0.7, 0.95)
# Zone au centre-bas de l'écran (30-70% en largeur, 60-95% en hauteur)
```

**Interprétation :** Ce mécanisme permet de protéger des zones comme un bureau de direction, un équipement sensible ou une zone d'exclusion.

---

## 5. Système de Journalisation et Suppression de Doublons

### 5.1 Gestion des Alertes

Le système génère des alertes avec un mécanisme de suppression des doublons pour éviter une saturation des logs :

**Règles de Suppression :**

1. **Cooldown Global** : Si le même message exact est généré deux fois dans un intervalle < 5 secondes, le second est omis.
2. **Historique par Clé** : Chaque type d'alerte unique (par contenu) dispose de son propre historique de timestamp.

```python
LOG_COOLDOWN_SECONDS = 5.0
```

**Format de Log :**
```
[2025-11-18 14:32:45] ALERT: fall detected (ID 5) height_ratio=0.58 vy=215.3 px/s
[2025-11-18 14:32:51] ALERT: sudden acceleration (ID 3) acc=1620.5 px/s^2
[2025-11-18 14:33:02] ALERT: long stillness (ID 7) time=8.2s
[2025-11-18 14:33:15] ALERT: restricted zone entered (ID 2)
```

---

## 6. Configuration et Optimisation

### 6.1 Paramètres Clés

| Paramètre | Valeur Par Défaut | Description |
|-----------|-------------------|-------------|
| `YOLO_INPUT_SIZE` | 416 | Dimension d'entrée du réseau (pixels) |
| `CONF_THRESHOLD` | 0.4 | Seuil de confiance minimum pour une détection |
| `NMS_THRESHOLD` | 0.4 | Seuil de suppression des chevauchements |
| `MAX_DISTANCE` | 80 px | Distance max pour associer deux détections |

### 6.2 Recommandations d'Optimisation

- **Pour une performance réduite** : Diminuer `YOLO_INPUT_SIZE` à 320 pour Raspberry Pi
- **Pour une sensibilité accrue** : Réduire `CONF_THRESHOLD` à 0.3–0.35
- **Pour un environnement bruyant** : Augmenter `STILLNESS_TIME_THRESHOLD` et `SUDDEN_ACCELERATION_THRESHOLD`

---

## 7. Installation et Utilisation

### 7.1 Installation sur Windows

```powershell
cd $HOME\Desktop
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 7.2 Lancement du Système

```powershell
python suspicious_behavior.py
```

Le script télécharge automatiquement les fichiers du modèle (~23 MB) lors de la première exécution.

**Arguments optionnels :**
```
python suspicious_behavior.py --camera 0 --conf 0.4 --size 416 --input <video_file>
```

---

## 8. Déploiement sur Raspberry Pi

### 8.1 Matériel Recommandé

| Plateforme | RAM | CPU | Recommandation |
|-----------|-----|-----|----------------|
| Raspberry Pi 3B+ | 1 GB | ARM32 | Marginal, réduction taille image |
| Raspberry Pi 4 | 4-8 GB | ARM64 | **Recommandé** |
| Raspberry Pi 5 | 4-8 GB | ARM64 | **Optimal** |

### 8.2 Installation sur Raspberry Pi OS

```bash
# 1. Créer virtualenv
python3 -m venv venv
source venv/bin/activate

# 2. Installer dépendances
pip install --upgrade pip
pip install numpy

# 3. OpenCV depuis source (optimisé NEON/VFPv4) ou prébuild
pip install opencv-python-headless
# OU compiler depuis source avec NEON activé:
# - Voir documentation Raspberry Pi pour build optimisé

pip install requests
```

### 8.3 Optimisations pour Pi

**Réduction de la taille d'entrée :**
```python
# Dans suspicious_behavior.py, ligne ~100
YOLO_INPUT_SIZE = 320  # au lieu de 416
```

**Augmentation de la swap (temporaire) :**
```bash
sudo dphys-swapfile swapoff
# Éditer /etc/dphys-swapfile et augmenter CONF_SWAPSIZE
sudo dphys-swapfile swapon
```

### 8.4 Module Caméra Raspberry Pi

Pour utiliser le module caméra Pi Camera V2/V3 (à la place d'une USB) :

**Option 1 : libcamera + OpenCV (Pi OS moderne)**
```python
# Utiliser directement /dev/video0 après activation de libcamera
cap = cv2.VideoCapture("/dev/video0")
```

**Option 2 : Bibliothèque picamera2 (recommandée)**
```python
from picamera2 import Picamera2

picam = Picamera2()
config = picam.create_preview_configuration(
    main={'format': 'XRGB8888', 'size': (640, 480)}
)
picam.configure(config)
picam.start()

# Dans la boucle principale :
frame = picam.capture_array()
```

### 8.5 Performance Attendue sur Pi 4

Avec `YOLO_INPUT_SIZE = 320` :
- **FPS** : 4–8 fps (selon surcharge système)
- **Latence** : 120–250 ms par frame
- **Consommation CPU** : 80–95%

---

## 9. Cas d'Usage et Intégrations Futures

### 9.1 Applications Actuelles

- Surveillance de résidences privées
- Monitoring de pièces de stockage
- Détection de chutes en établissements de santé
- Alertes d'accès non autorisé

### 9.2 Améliorations Possibles

1. **Intégration accélérateurs matériels**
   - Coral USB TPU (Edge TPU) : conversion en TFLite requise
   - Intel NCS2 : conversion en OpenVINO requise

2. **Interface Web de Visualisation**
   - Flux vidéo en temps réel
   - Historique des alertes avec filtrage
   - Configuration dynamique des seuils

3. **Persistance des Alertes**
   - Base de données SQLite pour archivage
   - Intégration webhook pour systèmes externes
   - Notifications par email/SMS

4. **Suivi Avancé**
   - Algorithme Hungarian pour apparier détections (au lieu du greedy)
   - Kalman Filter pour prédiction de trajectoire
   - Analyse de trajectoire (loitering, pacing)

---

## 10. Conclusion

Ce système offre une solution légère mais complète de surveillance comportementale adaptée aux environnements avec ressources limitées. Grâce à son architecture modulaire et sa configurabilité, il peut être déployé aussi bien sur des postes de travail que sur des périphériques embarqués comme le Raspberry Pi, tout en maintenant des performances de détection acceptables.

---

## Références

- **YOLO** : Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection"
- **OpenCV DNN** : https://docs.opencv.org/master/d6/d0f/group__dnn.html
- **Centroid Tracking** : Éducation en vision par ordinateur, tracking basique multi-objet
