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

### 2.0 Initialisation du Système

Avant de commencer la boucle principale, le système effectue les étapes d'initialisation suivantes :

**Code : Initialisation Complète**

```python
if __name__ == '__main__':
    args = parse_args()
    
    # ÉTAPE 1 : Assurer que tous les fichiers de modèle sont présents
    ensure_models()  # Télécharge automatiquement si nécessaire

    # ÉTAPE 2 : Charger les classes COCO
    with open(COCO_NAMES_PATH, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    person_class_id = classes.index('person') if 'person' in classes else 0
    print(f"Loaded {len(classes)} classes, person_class_id = {person_class_id}")

    # ÉTAPE 3 : Charger le réseau YOLOv4-tiny
    net = load_yolo(YOLO_CFG_PATH, YOLO_WEIGHTS_PATH)
    output_layer_names = get_output_layer_names(net)
    print(f"Loaded YOLO network with {len(output_layer_names)} output layers")

    # ÉTAPE 4 : Initialiser le tracker centroïde
    tracker = CentroidTracker(max_distance=MAX_DISTANCE)
    print(f"Centroid tracker initialized (max_distance={MAX_DISTANCE}px)")

    # ÉTAPE 5 : Ouvrir la source vidéo
    if args.input:
        cap = cv2.VideoCapture(args.input)
        print(f"Opened video file: {args.input}")
    else:
        cap = cv2.VideoCapture(args.camera)
        print(f"Opened camera (index {args.camera})")
    
    if not cap.isOpened():
        print('ERROR: Unable to open video source')
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Input FPS: {fps}")

    # ÉTAPE 6 : Lire la première frame et calculer zone restreinte
    ret, frame = cap.read()
    if not ret:
        print('ERROR: Unable to read from source')
        sys.exit(1)
    
    H, W = frame.shape[:2]
    print(f"Video resolution: {W}x{H}")
    
    rx1 = int(RESTRICTED_ZONE[0] * W)
    ry1 = int(RESTRICTED_ZONE[1] * H)
    rx2 = int(RESTRICTED_ZONE[2] * W)
    ry2 = int(RESTRICTED_ZONE[3] * H)
    print(f"Restricted zone: ({rx1},{ry1}) to ({rx2},{ry2})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Remettre au début du flux

    print('=' * 60)
    print('Suspicious Behavior Detector initialized successfully!')
    print('Press Q to quit the application')
    print('=' * 60)

    main()
```

**Explication :**
- **Téléchargement automatique** : `ensure_models()` garantit que les poids et config sont disponibles
- **Classe "person"** : On cherche l'indice de la classe "person" dans les 80 classes COCO
- **Couches de sortie** : YOLOv4-tiny a 3 couches de sortie à différentes échelles (multiscale detection)
- **Tracker** : Initialisé une seule fois avec le paramètre de distance maximale
- **Première frame** : Nécessaire pour connaître la résolution et convertir les coordonnées relatives de zone restreinte
- **Logs informatifs** : Affichage de tous les paramètres clés pour le débogage

---

### 2.2 Composants Principaux

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

### 2.3 Dépendances

```
opencv-python-headless==4.6.0.66    # Traitement vidéo et inférence DNN
numpy>=1.21                          # Calculs numériques et opérations matricielles
requests>=2.28                       # Téléchargement automatique des modèles
```

### 2.4 Fichiers de Modèle

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

**Code : Inférence YOLO et Extraction des Détections**

```python
# Préparation du blob et inférence
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (args.size, args.size), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(output_layer_names)

H, W = frame.shape[:2]
boxes = []
confidences = []
class_ids = []

# Collecte des détections
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = int(np.argmax(scores))
        conf = float(scores[class_id])
        
        # Filtrer par confiance et classe "person"
        if conf > args.conf and class_id == person_class_id:
            # Mise à l'échelle aux coordonnées du frame
            cx = int(detection[0] * W)
            cy = int(detection[1] * H)
            w = int(detection[2] * W)
            h = int(detection[3] * H)
            x = int(cx - w/2)
            y = int(cy - h/2)
            boxes.append([x, y, w, h])
            confidences.append(conf)
            class_ids.append(class_id)

# Application de la suppression des chevauchements (NMS)
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.conf, args.nms)
```

**Explication :**
- `blobFromImage()` : Normalise l'image (division par 255, redimensionnement, inversion RGB↔BGR)
- Chaque `detection` contient : [cx, cy, w, h, conf_background, conf_person, ..., conf_other_classes]
- Les coordonnées sont en valeurs normalisées (0-1), d'où la multiplication par W et H
- `NMSBoxes()` : Supprime les boîtes englobantes hautement chevauchantes, gardant celle avec la plus haute confiance

---

### 3.2 Suivi Centroïde

Un algorithme de suivi centroïde simple mais efficace associe les détections d'une image à l'autre :

**Principes :**
- Chaque personne détectée est représentée par son centroïde (centre de sa boîte englobante)
- Les centroïdes sont appariés entre images successives en minimisant la distance euclidienne
- Les appariements au-delà d'une distance seuil (`MAX_DISTANCE = 80 px`) créent une nouvelle trace

**Historique :**
- Chaque objet suivi conserve un historique des 32 derniers centroïdes avec leurs timestamps et dimensions
- Cet historique permet le calcul de vélocité, accélération et analyse temporelle

**Code : Classe de Suivi Centroïde**

```python
class CentroidTracker:
    def __init__(self, max_distance=MAX_DISTANCE):
        self.next_object_id = 1
        self.objects = {}  # id -> centroïde courant
        self.history = {}  # id -> deque de (timestamp, centroïde, dimensions)
        self.max_distance = max_distance

    def update(self, input_centroids, input_sizes=None):
        """Apparie les centroïdes détectés à la frame courante avec les objets existants"""
        now = time.time()
        
        # Cas initial : aucun objet existant
        if len(self.objects) == 0:
            for idx, c in enumerate(input_centroids):
                oid = self.next_object_id
                self.objects[oid] = c
                size = input_sizes[idx] if input_sizes is not None else None
                self.history[oid] = deque([(now, c, size)], maxlen=32)
                self.next_object_id += 1
            return self.objects

        # Construction de la matrice de distances
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[i] for i in object_ids]

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=float)
        for i, oc in enumerate(object_centroids):
            for j, ic in enumerate(input_centroids):
                D[i, j] = math.hypot(oc[0] - ic[0], oc[1] - ic[1])

        # Appariement glouton : sélectionner les paires avec les distances minimales
        assigned_objects = set()
        assigned_inputs = set()
        pairs = [(D[i, j], i, j) for i in range(D.shape[0]) for j in range(D.shape[1])]
        pairs.sort(key=lambda x: x[0])
        
        for dist, i, j in pairs:
            if i in assigned_objects or j in assigned_inputs:
                continue
            if dist > self.max_distance:  # Distance trop grande : pas d'appariement
                continue
            oid = object_ids[i]
            self.objects[oid] = input_centroids[j]
            size = input_sizes[j] if input_sizes is not None else None
            self.history[oid].append((now, input_centroids[j], size))
            assigned_objects.add(i)
            assigned_inputs.add(j)

        # Créer de nouveaux objets pour les centroïdes non appariés
        for j, ic in enumerate(input_centroids):
            if j in assigned_inputs:
                continue
            oid = self.next_object_id
            self.objects[oid] = ic
            size = input_sizes[j] if input_sizes is not None else None
            self.history[oid] = deque([(now, ic, size)], maxlen=32)
            self.next_object_id += 1

        return self.objects
```

**Explication :**
- La matrice `D` stocke la distance euclidienne entre chaque objet existant et chaque détection courante
- L'appariement glouton garantit une correspondance biunivoque : chaque objet ↔ une détection maximum
- `maxlen=32` : limite l'historique à 32 entrées pour économiser la mémoire
- Les centroïdes non appariés créent de nouveaux objets avec des IDs incrémentés

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

**Code : Calcul de l'Accélération**

```python
def get_velocity(self, oid, samples=3):
    """Retourne la vélocité approximative (vx, vy) en px/sec"""
    hist = list(self.history.get(oid, []))
    if len(hist) < 2:
        return (0.0, 0.0)
    
    pts = hist[-samples:]  # Derniers 3 échantillons
    if len(pts) < 2:
        return (0.0, 0.0)
    
    (t0, p0, _s0) = pts[0]
    (t1, p1, _s1) = pts[-1]
    dt = t1 - t0
    
    if dt <= 0:
        return (0.0, 0.0)
    
    vx = (p1[0] - p0[0]) / dt  # Composante horizontale
    vy = (p1[1] - p0[1]) / dt  # Composante verticale
    return (vx, vy)

def get_acceleration(self, oid, samples=4):
    """Retourne la magnitude d'accélération en px/sec²"""
    hist = list(self.history.get(oid, []))
    if len(hist) < 3:
        return 0.0
    
    # Calculer les vélocités successives
    velocities = []
    for i in range(1, len(hist)):
        t0, p0, _s0 = hist[i-1]
        t1, p1, _s1 = hist[i]
        dt = t1 - t0
        
        if dt <= 0:
            continue
        
        vx = (p1[0] - p0[0]) / dt
        vy = (p1[1] - p0[1]) / dt
        velocities.append((t1, (vx, vy)))
    
    if len(velocities) < 2:
        return 0.0
    
    # Accélération entre les deux derniers échantillons de vélocité
    (t_prev, v_prev) = velocities[-2]
    (t_last, v_last) = velocities[-1]
    dt = t_last - t_prev
    
    if dt <= 0:
        return 0.0
    
    ax = (v_last[0] - v_prev[0]) / dt
    ay = (v_last[1] - v_prev[1]) / dt
    return math.hypot(ax, ay)  # Magnitude du vecteur accélération
```

**Détection de l'Alerte :**

```python
# Calcul des métriques de comportement
acc = tracker.get_acceleration(best_id)
speed = tracker.get_speed(best_id)

# Vérification du seuil d'accélération soudaine
if acc > SUDDEN_ACCELERATION_THRESHOLD:
    text = f"ALERT: sudden acceleration (ID {best_id}) acc={acc:.1f} px/s^2"
    log_alert(text)
    cv2.putText(frame, "SUDDEN", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
```

**Explication :**
- `get_velocity()` : Calcule la pente entre le premier et le dernier point sur une fenêtre glissante
- `get_acceleration()` : Mesure le changement de vélocité sur le temps, détectant les changements rapides de direction/vitesse
- La magnitud `math.hypot(ax, ay)` combine les deux composantes (horizontale et verticale)
- Une alerte est générée si l'accélération dépasse 1500 px/s², indiquant un mouvement brusque et anormal

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

**Code : Calcul du Temps d'Immobilité**

```python
def time_still(self, oid, time_window=STILLNESS_TIME_THRESHOLD, dist_threshold=STILLNESS_DISTANCE_THRESHOLD):
    """Retourne la durée pendant laquelle l'objet est resté immobile"""
    now = time.time()
    hist = list(self.history.get(oid, []))
    
    if not hist:
        return 0.0
    
    # Parcours rétroactif jusqu'à détecter un mouvement > seuil
    count_time = 0.0
    for i in range(len(hist)-1, 0, -1):
        t_curr, p_curr, _s_curr = hist[i]
        t_prev, p_prev, _s_prev = hist[i-1]
        
        # Distance parcourue entre deux frames
        dist = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        dt = t_curr - t_prev
        
        # Si mouvement détecté au-delà du seuil, arrêter
        if dist > dist_threshold:
            break
        
        # Accumuler le temps d'immobilité
        count_time += dt
        
        # Arrêter si on a atteint la fenêtre temporelle
        if count_time >= time_window:
            break
    
    return count_time
```

**Détection de l'Alerte :**

```python
# Calcul de l'immobilité
still_time = tracker.time_still(best_id)

# Vérification du seuil d'immobilité prolongée
if still_time >= STILLNESS_TIME_THRESHOLD:
    text = f"ALERT: long stillness (ID {best_id}) time={still_time:.1f}s"
    log_alert(text)
    cv2.putText(frame, "STILLNESS", (x, y+h+35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
```

**Explication :**
- Parcours **rétroactif** : on remonte dans l'historique depuis la frame actuelle
- Si un mouvement > 10 px est détecté, on arrête (la personne s'est déplacée)
- **Accumulation** : on additionne tous les intervalles de temps durant lesquels la personne est restée immobile
- Cas d'usage : détection de malaise, personne tombée, personne figée ou incapacitée

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

$$\text{height ratio} = \frac{h(t)}{h(t-1s)} < \text{FALL HEIGHT RATIO}$$

```python
FALL_HEIGHT_RATIO = 0.6  # La personne descend à 60% de sa hauteur antérieure
```

#### Critère 3 : Orientation Horizontale

Détection d'une boîte englobante avec un rapport largeur/hauteur anormal :

$$\text{aspect ratio} = \frac{w}{h} > \text{ASPECT RATIO THRESHOLD}$$

```python
ASPECT_RATIO_THRESHOLD = 1.2
```

**Confirmation par Immobilité :**

Pour éviter les faux positifs, une chute est confirmée seulement après que la personne soit restée immobile pendant :

```python
FALL_STILLNESS_TIME = 3.0  # secondes
```

**Code : Calcul du Rapport de Hauteur et Détection de Chute**

```python
def get_height_ratio_change(self, oid, seconds=1.0):
    """Retourne h(t) / h(t - seconds) pour détecter une réduction de hauteur"""
    hist = list(self.history.get(oid, []))
    if len(hist) < 2:
        return 1.0
    
    latest_t, _latest_p, latest_s = hist[-1]
    
    # Chercher l'entrée datant d'environ 1 seconde auparavant
    target_t = latest_t - seconds
    earlier_s = None
    for (t, p, s) in reversed(hist):
        if t <= target_t:
            earlier_s = s
            break
    
    if earlier_s is None:
        # Fallback : première entrée de l'historique
        earlier_s = hist[0][2]
    
    if earlier_s is None or latest_s is None:
        return 1.0
    
    # Extraire les hauteurs (stockées comme (largeur, hauteur))
    h_latest = latest_s[1] if isinstance(latest_s, (list, tuple)) else latest_s
    h_earlier = earlier_s[1] if isinstance(earlier_s, (list, tuple)) else earlier_s
    
    if h_earlier <= 0:
        return 1.0
    
    return float(h_latest) / float(h_earlier)
```

**Détection de l'Alerte de Chute :**

```python
# Récupération des vélocités et dimensions
vx, vy = tracker.get_velocity(best_id)
height_ratio = tracker.get_height_ratio_change(best_id, seconds=1.0)

# Critères de candidate chute
fall_candidate = False

# Critère 1 & 2 : Descente rapide ET réduction de hauteur
if vy > FALL_DESCENT_SPEED and height_ratio < FALL_HEIGHT_RATIO:
    fall_candidate = True

# Critère 3 : Aspect ratio horizontal (personne allongée)
last_size = tracker.history.get(best_id, [])[-1][2] if tracker.history.get(best_id) else None
if last_size is not None:
    w_last, h_last = last_size
    if h_last > 0 and (float(w_last) / float(h_last)) > ASPECT_RATIO_THRESHOLD:
        fall_candidate = True

# Confirmation par immobilité prolongée
if fall_candidate and tracker.time_still(best_id, 
                                         time_window=FALL_STILLNESS_TIME, 
                                         dist_threshold=STILLNESS_DISTANCE_THRESHOLD) >= FALL_STILLNESS_TIME:
    text = f"ALERT: fall detected (ID {best_id}) height_ratio={height_ratio:.2f} vy={vy:.1f} px/s"
    log_alert(text)
    cv2.putText(frame, "FALL", (x, y+h+75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
```

**Explication :**
- **Composante verticale (`vy`)** : La vélocité positive vers le bas > 200 px/s indique une chute rapide
- **Réduction de hauteur** : La personne qui tombe voit sa hauteur de boîte englobante diminuer (le haut du corps descend)
- **Aspect ratio** : Une personne allongée par terre a une largeur > hauteur (rapport > 1.2)
- **Confirmation par immobilité** : On attend 3 secondes d'immobilité pour éviter les faux positifs (ex. flexion rapide)
- Cette approche multi-critère réduit significativement les faux positifs tout en détectant les vrais chutes

---

### 4.4 Entrée en Zone Restreinte

**Description :** Déclenche une alerte lorsqu'une personne pénètre dans une région spatiale définie (par ex., zone interdite, bureau privé).

**Méthode de Détection :**

La zone restreinte est définie comme un rectangle en coordonnées relatives (fractions du cadre vidéo) :

$$\text{RESTRICTED ZONE} = (x_1, y_1, x_2, y_2) \in [0,1]^4$$

À chaque détection, le centroïde $(c_x, c_y)$ en pixels est comparé à la zone convertie :

$$\text{alert} \iff \left( \frac{x_1 \cdot W}{1} \leq c_x \leq \frac{x_2 \cdot W}{1} \right) \land \left( \frac{y_1 \cdot H}{1} \leq c_y \leq \frac{y_2 \cdot H}{1} \right)$$

où $W$ et $H$ sont les dimensions du cadre vidéo.

**Paramètres Configurables :**
```python
RESTRICTED_ZONE = (0.3, 0.6, 0.7, 0.95)
# Zone au centre-bas de l'écran (30-70% en largeur, 60-95% en hauteur)
```

**Interprétation :** Ce mécanisme permet de protéger des zones comme un bureau de direction, un équipement sensible ou une zone d'exclusion.

**Code : Initialisation et Vérification de Zone Restreinte**

```python
# Initialisation au démarrage (dans main())
ret, frame = cap.read()
if not ret:
    print('Unable to read from camera')
    sys.exit(1)

H, W = frame.shape[:2]

# Conversion des coordonnées relatives en pixels
rx1 = int(RESTRICTED_ZONE[0] * W)  # x_min en pixels
ry1 = int(RESTRICTED_ZONE[1] * H)  # y_min en pixels
rx2 = int(RESTRICTED_ZONE[2] * W)  # x_max en pixels
ry2 = int(RESTRICTED_ZONE[3] * H)  # y_max en pixels

# Visualisation : rectangle de zone restreinte
cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
cv2.putText(frame, "Restricted Zone", (rx1, ry1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
```

**Vérification d'Entrée en Zone Restreinte :**

```python
# Pour chaque détection de personne
for (x, y, w, h, conf) in final_boxes:
    cx = int(x + w/2)
    cy = int(y + h/2)
    best_id = ...  # ID de la personne suivie
    
    # Vérification : le centroïde est-il dans la zone restreinte ?
    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
        text = f"ALERT: restricted zone entered (ID {best_id})"
        log_alert(text)
        cv2.putText(frame, "RESTRICTED", (x, y+h+55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
```

**Explication :**
- Les coordonnées **relatives** (0-1) facilitent l'adaptation à différentes résolutions de caméra
- La conversion en **pixels** se fait une seule fois au démarrage pour efficacité
- Le test de **containment simple** : `rx1 <= cx <= rx2 and ry1 <= cy <= ry2`
- Visualisation : un **rectangle rouge** aide l'opérateur à comprendre la zone protégée
- Cas d'usage : entrées de locaux sensibles, zones d'exclusion, périmètres de sécurité

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

**Code : Fonction de Journalisation avec Suppression**

```python
# Variables globales pour suppression de doublons
_last_log_text = None
_last_log_time = 0.0
_last_log_times = {}

def log_alert(text):
    """Enregistre une alerte avec suppression des doublons proches dans le temps"""
    global _last_log_text, _last_log_time, _last_log_times
    
    now_ts = time.time()
    key = text.strip()

    # Vérifier si le message exact a été écrit récemment (cooldown global)
    if _last_log_text == key and (now_ts - _last_log_time) < LOG_COOLDOWN_SECONDS:
        return

    # Vérifier si ce type spécifique d'alerte a été enregistré récemment
    last_time_for_key = _last_log_times.get(key, 0.0)
    if (now_ts - last_time_for_key) < LOG_COOLDOWN_SECONDS:
        return

    # Mise à jour de l'état de suppression
    _last_log_text = key
    _last_log_time = now_ts
    _last_log_times[key] = now_ts

    # Écriture dans le fichier de log avec timestamp
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {text}\n"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line)
    
    # Affichage en console
    print(line.strip())
```

**Explication :**
- **Cooldown global** : `_last_log_text` conserve le dernier message et son timestamp
- **Historique par clé** : `_last_log_times` est un dictionnaire { alerte_type → timestamp }
- **Double vérification** : évite à la fois les répétitions du même message ET d'autres messages similaires trop proches
- **Avantages** : Logs concis et lisibles, pas de saturation même avec détections instables

---

## 6. Helpers YOLO et Initialisation

### 6.1 Chargement du Modèle

**Code : Fonction de Chargement YOLO**

```python
def load_yolo(net_cfg, net_weights):
    """Charge le réseau YOLOv4-tiny avec configuration CPU"""
    net = cv2.dnn.readNetFromDarknet(net_cfg, net_weights)
    
    # Forcer l'utilisation de la CPU (pas de GPU)
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception:
        pass  # Certaines versions d'OpenCV ne supportent pas ces paramètres
    
    return net

def get_output_layer_names(net):
    """Récupère les noms des couches de sortie (gère les différentes versions OpenCV)"""
    layer_names = net.getLayerNames()
    
    try:
        # Récupérer les indices des couches non connectées
        outs = net.getUnconnectedOutLayers()
        
        # Gérer différents formats de sortie selon la version OpenCV
        if hasattr(outs, 'flatten'):
            idxs = outs.flatten().tolist()
        else:
            idxs = list(outs)

        # Convertir les indices en noms (les indices sont 1-based)
        cleaned = []
        for i in idxs:
            if isinstance(i, (list, tuple, np.ndarray)):
                ii = int(np.array(i).flatten()[0])
            else:
                ii = int(i)
            cleaned.append(ii)

        out_layers = [layer_names[i - 1] for i in cleaned]
        return out_layers
    
    except Exception:
        # Fallback : certaines versions d'OpenCV modernes
        try:
            return net.getUnconnectedOutLayersNames()
        except Exception:
            # Dernier recours : retourner tous les noms (moins efficace)
            return layer_names
```

**Explication :**
- `readNetFromDarknet()` : Charge la configuration (.cfg) et les poids (.weights) du réseau Darknet
- `setPreferableTarget(DNN_TARGET_CPU)` : Force l'inférence sur CPU pour la compatibilité Raspberry Pi
- `getUnconnectedOutLayers()` : Retourne les indices des couches de sortie (il y en a 3 pour YOLOv4-tiny)
- **Gestion des versions** : OpenCV a eu plusieurs changements d'API ; le code gère les différentes variantes

### 6.2 Téléchargement Automatique des Modèles

**Code : Téléchargement des Fichiers de Modèle**

```python
def download_file(url, dest_path, chunk_size=8192):
    """Télécharge un fichier depuis une URL avec affichage de la progression"""
    
    # Vérifier si le fichier existe déjà
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"Found existing {os.path.basename(dest_path)} — skipping download")
        return
    
    print(f"Downloading {url} -> {dest_path}")
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()  # Lever une exception en cas d'erreur HTTP
    
    total = int(r.headers.get('content-length', 0))
    
    # Écriture par chunks avec barre de progression
    with open(dest_path, 'wb') as f:
        downloaded = 0
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 / total
                    print(f"\r{pct:3.0f}% ", end='', flush=True)
    
    print("\nDownload completed")

def ensure_models():
    """Assure que tous les fichiers du modèle sont présents (les télécharge si nécessaire)"""
    try:
        if not os.path.exists(COCO_NAMES_PATH):
            download_file(COCO_NAMES_URL, COCO_NAMES_PATH)
        if not os.path.exists(YOLO_CFG_PATH):
            download_file(YOLO_CFG_URL, YOLO_CFG_PATH)
        if not os.path.exists(YOLO_WEIGHTS_PATH):
            download_file(YOLO_WEIGHTS_URL, YOLO_WEIGHTS_PATH)
    except Exception as e:
        print("Error downloading models:", e)
        print("Please check your internet connection or download the files manually into 'models/'")
        raise
```

**Explication :**
- **Vérification préalable** : Si le fichier existe déjà avec taille > 0, skip le téléchargement
- **Progression** : Affichage du pourcentage en temps réel
- **Chunked download** : Par défaut 8192 bytes à la fois, économe en mémoire
- **Gestion des erreurs** : Les exceptions sont relayées pour arrêter le programme si échec

---

## 7. Boucle Principale de Traitement

Le cœur du système : capture vidéo, inférence, suivi et analyse comportementale.

**Code : Boucle Principale**

```python
def main():
    args = parse_args()
    
    # Charger les classes COCO
    with open(COCO_NAMES_PATH, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]
    person_class_id = classes.index('person') if 'person' in classes else 0

    # Charger le réseau YOLO
    net = load_yolo(YOLO_CFG_PATH, YOLO_WEIGHTS_PATH)
    output_layer_names = get_output_layer_names(net)

    # Initialiser le tracker centroïde
    tracker = CentroidTracker(max_distance=MAX_DISTANCE)

    # Ouvrir la source vidéo (webcam ou fichier)
    if args.input:
        cap = cv2.VideoCapture(args.input)
    else:
        cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print('Unable to open video source')
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Input FPS (approx): {fps}")

    # Lire la première frame et convertir les coordonnées de zone restreinte
    ret, frame = cap.read()
    if not ret:
        print('Unable to read from camera')
        sys.exit(1)
    
    H, W = frame.shape[:2]
    rx1 = int(RESTRICTED_ZONE[0] * W)
    ry1 = int(RESTRICTED_ZONE[1] * H)
    rx2 = int(RESTRICTED_ZONE[2] * W)
    ry2 = int(RESTRICTED_ZONE[3] * H)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Remettre au début
    print('Starting detection. Press q to quit.')

    # ===== BOUCLE PRINCIPALE =====
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ÉTAPE 1 : Préparation du blob et inférence YOLO
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (args.size, args.size), 
                                     swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layer_names)

        H, W = frame.shape[:2]
        boxes = []
        confidences = []
        class_ids = []

        # ÉTAPE 2 : Collecte des détections de personnes
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])
                
                if conf > args.conf and class_id == person_class_id:
                    cx = int(detection[0] * W)
                    cy = int(detection[1] * H)
                    w = int(detection[2] * W)
                    h = int(detection[3] * H)
                    x = int(cx - w/2)
                    y = int(cy - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(conf)
                    class_ids.append(class_id)

        # ÉTAPE 3 : Suppression des chevauchements (NMS)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.conf, args.nms)
        centroids = []
        final_boxes = []
        
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y, w, h = boxes[i]
                cx = int(x + w/2)
                cy = int(y + h/2)
                centroids.append((cx, cy))
                final_boxes.append((x, y, w, h, confidences[i]))

        sizes = [(w, h) for (x, y, w, h, conf) in final_boxes]

        # ÉTAPE 4 : Mise à jour du tracker
        objects = tracker.update(centroids, input_sizes=sizes)

        # ÉTAPE 5 : Analyse comportementale et visualisation
        for (x, y, w, h, conf) in final_boxes:
            cx = int(x + w/2)
            cy = int(y + h/2)
            
            # Trouver l'ID de l'objet correspondant
            best_id = None
            best_dist = float('inf')
            for oid, cent in objects.items():
                d = math.hypot(cent[0]-cx, cent[1]-cy)
                if d < best_dist:
                    best_dist = d
                    best_id = oid

            # Dessiner la boîte englobante
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"ID {best_id} {conf:.2f}" if best_id is not None else f"{conf:.2f}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            # Analyse comportementale
            if best_id is not None:
                acc = tracker.get_acceleration(best_id)
                still_time = tracker.time_still(best_id)
                height_ratio = tracker.get_height_ratio_change(best_id, seconds=1.0)
                vx, vy = tracker.get_velocity(best_id)

                # Détection de chute
                fall_candidate = False
                if vy > FALL_DESCENT_SPEED and height_ratio < FALL_HEIGHT_RATIO:
                    fall_candidate = True
                
                last_size = tracker.history.get(best_id, [])[-1][2] if tracker.history.get(best_id) else None
                if last_size is not None:
                    w_last, h_last = last_size
                    if h_last > 0 and (float(w_last) / float(h_last)) > ASPECT_RATIO_THRESHOLD:
                        fall_candidate = True

                if fall_candidate and tracker.time_still(best_id, FALL_STILLNESS_TIME, STILLNESS_DISTANCE_THRESHOLD) >= FALL_STILLNESS_TIME:
                    log_alert(f"ALERT: fall detected (ID {best_id}) height_ratio={height_ratio:.2f} vy={vy:.1f} px/s")
                    cv2.putText(frame, "FALL", (x, y+h+75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # Accélération soudaine
                if acc > SUDDEN_ACCELERATION_THRESHOLD:
                    log_alert(f"ALERT: sudden acceleration (ID {best_id}) acc={acc:.1f} px/s^2")
                    cv2.putText(frame, "SUDDEN", (x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # Immobilité prolongée
                if still_time >= STILLNESS_TIME_THRESHOLD:
                    log_alert(f"ALERT: long stillness (ID {best_id}) time={still_time:.1f}s")
                    cv2.putText(frame, "STILLNESS", (x, y+h+35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

                # Zone restreinte
                if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                    log_alert(f"ALERT: restricted zone entered (ID {best_id})")
                    cv2.putText(frame, "RESTRICTED", (x, y+h+55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Visualisation de la zone restreinte
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        cv2.putText(frame, "Restricted Zone", (rx1, ry1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # Affichage et gestion de la sortie
        cv2.imshow('Suspicious Behavior Monitor', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Nettoyage
    cap.release()
    cv2.destroyAllWindows()
```

**Explication du Flux :**

1. **Initialisation** : Charger modèles, classes, tracker, source vidéo
2. **Inférence YOLO** : Créer blob, passer au réseau, récupérer outputs
3. **Post-traitement** : Filtrer par confiance, scaling des coordonnées, NMS
4. **Suivi** : Mettre à jour le tracker avec les centroïdes actuels
5. **Analyse** : Pour chaque détection, calculer accélération, immobilité, hauteur, position
6. **Alertes** : Vérifier les seuils et enregistrer les alertes
7. **Visualisation** : Dessiner les boîtes, IDs, textes d'alerte, zone restreinte
8. **Affichage** : Montrer la frame annotée, vérifier la touche 'q' pour quitter

---

## 8. Configuration et Optimisation

### 8.1 Paramètres Clés

| Paramètre | Valeur Par Défaut | Description |
|-----------|-------------------|-------------|
| `YOLO_INPUT_SIZE` | 416 | Dimension d'entrée du réseau (pixels) |
| `CONF_THRESHOLD` | 0.4 | Seuil de confiance minimum pour une détection |
| `NMS_THRESHOLD` | 0.4 | Seuil de suppression des chevauchements |
| `MAX_DISTANCE` | 80 px | Distance max pour associer deux détections |

### 8.2 Recommandations d'Optimisation

- **Pour une performance réduite** : Diminuer `YOLO_INPUT_SIZE` à 320 pour Raspberry Pi
- **Pour une sensibilité accrue** : Réduire `CONF_THRESHOLD` à 0.3–0.35
- **Pour un environnement bruyant** : Augmenter `STILLNESS_TIME_THRESHOLD` et `SUDDEN_ACCELERATION_THRESHOLD`

---

## 9. Installation et Utilisation

### 9.1 Installation sur Windows

```powershell
cd $HOME\Desktop
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 9.2 Lancement du Système

```powershell
python suspicious_behavior.py
```

Le script télécharge automatiquement les fichiers du modèle (~23 MB) lors de la première exécution.

**Arguments optionnels :**
```
python suspicious_behavior.py --camera 0 --conf 0.4 --size 416 --input <video_file>
```

---

## 10. Déploiement sur Raspberry Pi

### 10.1 Matériel Recommandé

| Plateforme | RAM | CPU | Recommandation |
|-----------|-----|-----|----------------|
| Raspberry Pi 3B+ | 1 GB | ARM32 | Marginal, réduction taille image |
| Raspberry Pi 4 | 4-8 GB | ARM64 | **Recommandé** |
| Raspberry Pi 5 | 4-8 GB | ARM64 | **Optimal** |

### 10.2 Installation sur Raspberry Pi OS

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

### 10.3 Optimisations pour Pi

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

### 10.4 Module Caméra Raspberry Pi

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

### 10.5 Performance Attendue sur Pi 4

Avec `YOLO_INPUT_SIZE = 320` :
- **FPS** : 4–8 fps (selon surcharge système)
- **Latence** : 120–250 ms par frame
- **Consommation CPU** : 80–95%

---

## 11. Cas d'Usage et Intégrations Futures

### 11.1 Applications Actuelles

- Surveillance de résidences privées
- Monitoring de pièces de stockage
- Détection de chutes en établissements de santé
- Alertes d'accès non autorisé

### 11.2 Améliorations Possibles

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

## 12. Conclusion

Ce système offre une solution légère mais complète de surveillance comportementale adaptée aux environnements avec ressources limitées. Grâce à son architecture modulaire et sa configurabilité, il peut être déployé aussi bien sur des postes de travail que sur des périphériques embarqués comme le Raspberry Pi, tout en maintenant des performances de détection acceptables.

---

## Références

- **YOLO** : Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection"
- **OpenCV DNN** : https://docs.opencv.org/master/d6/d0f/group__dnn.html
- **Centroid Tracking** : Éducation en vision par ordinateur, tracking basique multi-objet
