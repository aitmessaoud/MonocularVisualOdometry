import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from numpy.linalg import inv, det
import scipy.io
import json
import pandas as pd
# Charger le fichier .mat
mat = scipy.io.loadmat('IntrinsicMatrix.mat')
# Extraire la matrice intrinséque K
K = mat['intrinsicMatrix'].T
# Création du détecteur SIFT
# Paramètres SIFT modifiables
nfeatures = 0 # Nombre de meilleurs points à conserver
contrastThreshold = 0.04  # Seuil de contraste
# Création du détecteur SIFT avec des paramètres personnalisés
# sift = cv2.SIFT_create(nfeatures, contrastThreshold=contrastThreshold, edgeThreshold=edgeThreshold)
sift = cv2.SIFT_create()
# Utiliser BFMatcher avec la norme L2
bf = cv2.BFMatcher(cv2.NORM_L2)

def select_interest_points(image,idx):
    # Créer une fenêtre pour l'affichage de l'image
    cv2.namedWindow("Selectionnez les points d'interet", cv2.WINDOW_NORMAL)

    # Copier l'image pour éviter de dessiner sur l'originale
    img_copy = image.copy()

    # Liste pour stocker les points d'intérêt sélectionnés
    points = []

    # Fonction de rappel pour la souris
    def mouse_callback(event, x, y, flags, param):
        nonlocal points

        if event == cv2.EVENT_LBUTTONDOWN:
            # Ajouter le point à la liste
            points.append((x, y))
            # Dessiner le point sur l'image copiée
            cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)

            # Afficher l'image mise à jour
            cv2.imshow("Selectionnez les points d'interet", img_copy)

    # Attacher la fonction de rappel à la fenêtre
    cv2.setMouseCallback("Selectionnez les points d'interet", mouse_callback)

    # Attendre jusqu'à ce que l'utilisateur appuie sur la touche "Entrée"
    while True:
        cv2.imshow("Selectionnez les points d'interet", img_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Appuyez sur la touche Entrée pour terminer la sélection
            break
        if key == ord('q'):  # Appuyez sur la touche 'q' pour quitter
            return None 
    # Fermer la fenêtre après la sélection
    cv2.destroyAllWindows()

    # Convertir les points en un tableau numpy
    points = np.array(points, dtype=np.float32)
    return points
def compute_homography(src_pts, dst_pts): 
    """
    Calcule la matrice de homographie entre deux ensembles de points correspondants de deux images successives.
    Cette fonction utilise une méthode manuelle pour calculer la homographie en résolvant un système d'équations linéaires.
    Elle n'est cependant pas utilisée dans le programme principal car la fonction findHomography de OpenCV fournit une solution
    plus robuste et précise grâce à des algorithmes comme RANSAC pour gérer les erreurs de correspondance.

    :param src_pts: Points dans la première image (source).
    :param dst_pts: Points correspondants dans la seconde image (destination).
    :return: Matrice de homographie H qui transforme les points de src_pts en dst_pts.
    """
    nbp = len(src_pts)
    A = np.zeros((2 * nbp, 9))
    for i in range(nbp):
        u0, v0 = src_pts[i][0]
        u1, v1 = dst_pts[i][0]
        A[2*i] = [0, 0, 0, -u1, -v1, -1, v0*u1, v0*v1, v0]
        A[2*i+1] = [u1, v1, 1, 0, 0, 0, -u0*u1, -u0*v1, -u0]

    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    h /= h[-1]
    H = h.reshape(3, 3)

    return H

def find_feature_matches(frame1, frame2, sift, bf, selected_points):
    # Détecter et décrire les caractéristiques avec SIFT dans les deux images
    kp1, des1 = sift.detectAndCompute(frame1, None)  # Pour la première image
    kp2, des2 = sift.detectAndCompute(frame2, None)  # Pour la deuxième image

    # Apparier les caractéristiques entre les deux images en utilisant BFMatcher
    raw_matches = bf.knnMatch(des1, des2, k=2)  # Trouver les 2 meilleurs matches pour chaque descripteur

    # Appliquer le test de ratio de Lowe pour filtrer les correspondances
    # Ce test permet d'éliminer les correspondances faibles
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.4 * n.distance:  # Si la distance est moins de 40% de la seconde meilleure correspondance
            good_matches.append(m)  # Ajouter à la liste des bonnes correspondances

    # Vérifier s'il y a suffisamment de bonnes correspondances
    if len(good_matches) >= 10:  # Un seuil minimal de correspondances pour poursuivre
        # Extraire les points de correspondance des bonnes correspondances
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # Points dans la première image
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # Points correspondants dans la seconde image

        # Calculer la matrice de homographie en utilisant RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Appliquer la transformation de perspective pour aligner les points sélectionnés
        points_transformed = cv2.perspectiveTransform(selected_points.reshape(-1, 1, 2), H)

        # Retourner les points transformés et autres informations pertinentes
        return points_transformed, src_pts, dst_pts, kp1, kp2, good_matches
    else:
        # Si pas assez de bonnes correspondances, retourner None pour tous les paramètres
        return None, None, None, None, None, []

 
# Paramètres de calibration de la caméra et points dans le monde réel
# ... [Initialisez ici vos variables world_points, K, etc.] ...

def compare_des(des_current, des_face):
    """
    Fonction pour comparer les descripteurs entre deux frames (image courante et image de référence)
    et renvoyer le nombre de bonnes correspondances.
    """
    # Utilisation de knnMatch pour trouver les deux meilleures correspondances pour chaque descripteur
    matches = bf.knnMatch(des_current, des_face, k=2)

    # Application du test de ratio pour filtrer les correspondances
    # Seules les correspondances dont la distance est inférieure à 30% de la seconde meilleure correspondance sont conservées
    good_matches = [m for m, n in matches if m.distance < 0.3 * n.distance]

    # Retourne le nombre de bonnes correspondances
    return len(good_matches)

def compare_des_faces(des_current, des_faces):
    """
    Fonction pour comparer le descripteur de la frame courante avec les descripteurs de chaque face de la boîte
    et renvoyer le nombre maximum de correspondances parmi toutes les faces.
    """
    # Liste pour stocker le nombre de correspondances pour chaque face
    num_match = []

    # Boucle sur chaque ensemble de descripteurs de face
    for des in des_faces:
        # Comparer le descripteur courant avec chaque descripteur de face et stocker le nombre de bonnes correspondances
        num_match.append(compare_des(des_current, des))

    # Retourne le nombre maximum de correspondances trouvées parmi toutes les faces
    return max(num_match)

def load_interest_points(file_path):
    """
    Fonction pour charger les points d'intérêt à partir d'un fichier JSON.
    :param file_path: Chemin vers le fichier JSON contenant les points d'intérêt.
    :return: Dictionnaire des données chargées à partir du fichier JSON.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)  # Chargement des données JSON dans une variable
    return data  # Retour des données chargées

# Charger les données des points d'intérêt
data = load_interest_points("interest_points.json")

# Accéder et convertir en array numpy aux points d'intérêt pour chaque face de la boîte
# Image Points - Points dans l'image où les caractéristiques sont détectées
# World Points - Points correspondants dans l'espace monde (réel)

# Pour la face 1
image_points_face1 = np.array(data["face1"]["image_points"], dtype=np.float32)  # Points d'image pour la face 1
world_points_face1 = np.array(data["face1"]["world_points"], dtype=np.float32)  # Points du monde réel pour la face 1

# Pour la face 2
image_points_face2 = np.array(data["face2"]["image_points"], dtype=np.float32)  # Points d'image pour la face 2
world_points_face2 = np.array(data["face2"]["world_points"], dtype=np.float32)  # Points du monde réel pour la face 2

# Pour la face 3
image_points_face3 = np.array(data["face3"]["image_points"], dtype=np.float32)  # Points d'image pour la face 3
world_points_face3 = np.array(data["face3"]["world_points"], dtype=np.float32)  # Points du monde réel pour la face 3

# Pour la face 4
image_points_face4 = np.array(data["face4"]["image_points"], dtype=np.float32)  # Points d'image pour la face 4
world_points_face4 = np.array(data["face4"]["world_points"], dtype=np.float32)  # Points du monde réel pour la face 4

#%%
# Chemin vers le dossier contenant les images
folder_path = "sequence_images"

# Trier les fichiers d'images du dossier
image_files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('.')[0]))

# Traitement de la première face
first_image_path = os.path.join(folder_path, image_files[0])  # Chemin de l'image représentant la face 1
first_frame = cv2.imread(first_image_path)                    # Lecture de l'image de la face 1
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)  # Conversion en niveaux de gris
kp_face1, des_face1 = sift.detectAndCompute(first_frame_gray, None)  # Détection SIFT pour la face 1
# image_points_face1 = select_interest_points(first_frame, 1)  # Optionnel : Sélection manuelle des points d'intérêt

# Traitement de la deuxième face
second_image_path = os.path.join(folder_path, image_files[9])  # Chemin de l'image représentant la face 2
second_frame = cv2.imread(second_image_path)                   # Lecture de l'image de la face 2
second_frame_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)  # Conversion en niveaux de gris
kp_face2, des_face2 = sift.detectAndCompute(second_frame_gray, None)  # Détection SIFT pour la face 2
# image_points_face2 = select_interest_points(second_frame, 2)  # Optionnel : Sélection manuelle des points d'intérêt

# Traitement de la troisième face
third_image_path = os.path.join(folder_path, image_files[17])  # Chemin de l'image représentant la face 3
third_frame = cv2.imread(third_image_path)                     # Lecture de l'image de la face 3
third_frame_gray = cv2.cvtColor(third_frame, cv2.COLOR_BGR2GRAY)  # Conversion en niveaux de gris
kp_face3, des_face3 = sift.detectAndCompute(third_frame_gray, None)  # Détection SIFT pour la face 3
# image_points_face3 = select_interest_points(third_frame, 3)  # Optionnel : Sélection manuelle des points d'intérêt

# Traitement de la quatrième face
fourth_image_path = os.path.join(folder_path, image_files[25])  # Chemin de l'image représentant la face 4
fourth_frame = cv2.imread(fourth_image_path)                    # Lecture de l'image de la face 4
fourth_frame_gray = cv2.cvtColor(fourth_frame, cv2.COLOR_BGR2GRAY)  # Conversion en niveaux de gris
kp_face4, des_face4 = sift.detectAndCompute(fourth_frame_gray, None)  # Détection SIFT pour la face 4
# image_points_face4 = select_interest_points(fourth_frame, 4)  # Optionnel : Sélection manuelle des points d'intérêt

 
#%%
# Création d'une liste contenant les descripteurs de toutes les faces
des_faces = [des_face1, des_face2, des_face3, des_face4]

# Calcul des seuils de changement de face
# Ces seuils représentent le nombre de bonnes correspondances lors de la comparaison des descripteurs d'une face avec elle-même.
# Ces seuils sont utilisés plus tard pour déterminer si une image correspond à l'une des faces de la boîte.

# Seuil pour la face 1
CHANGE_FACE_1_THRESHOLD = compare_des(des_face1, des_face1)  # Comparaison des descripteurs de la face 1 avec elle-même

# Seuil pour la face 2
CHANGE_FACE_2_THRESHOLD = compare_des(des_face2, des_face2)  # Comparaison des descripteurs de la face 2 avec elle-même

# Seuil pour la face 3
CHANGE_FACE_3_THRESHOLD = compare_des(des_face3, des_face3)  # Comparaison des descripteurs de la face 3 avec elle-même

# Seuil pour la face 4
CHANGE_FACE_4_THRESHOLD = compare_des(des_face4, des_face4)  # Comparaison des descripteurs de la face 4 avec elle-même

# Stocker tous les seuils dans une liste pour un accès facile
CHANGE_FACE_THRESHOLD = [CHANGE_FACE_1_THRESHOLD, CHANGE_FACE_2_THRESHOLD, CHANGE_FACE_3_THRESHOLD, CHANGE_FACE_4_THRESHOLD]

#%% 
 
 
# Initialisation des variables nécessaires
old_frame = None  # Pour stocker l'image précédente dans la boucle
camera_position = []  # Liste pour stocker les positions calculées de la caméra
H_total = np.eye(4)  # Matrice d'homographie initiale (matrice identité 4x4)

# Sélection initiale des points d'image et des points du monde pour la première face
image_points = image_points_face1
world_points = world_points_face1

# Boucle sur les fichiers d'images
for file in image_files:
    frame_path = os.path.join(folder_path, file)  # Chemin du fichier d'image actuel
    frame = cv2.imread(frame_path)  # Lecture de l'image

    # Vérification si l'image est lue correctement
    if frame is None:
        continue

    # Traitement si une ancienne image existe
    if old_frame is not None:
        gray_old = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)  # Conversion de l'ancienne image en niveaux de gris
        gray_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Conversion de l'image actuelle en niveaux de gris

        # Détecter les caractéristiques dans l'image actuelle
        kp_current, des_current = sift.detectAndCompute(gray_new, None)

        # Comparaison avec les faces connues pour déterminer la correspondance
        num_matches = compare_des_faces(des_current, des_faces)

        # Vérification du seuil de changement de face et mise à jour des points d'intérêt
        if num_matches in CHANGE_FACE_THRESHOLD:
            face = CHANGE_FACE_THRESHOLD.index(num_matches)
            if face == 0: 
                print("Changement de face vers face 1 au frame", file)
                world_points = world_points_face1
                image_points = image_points_face1
            elif face == 1:
                print("Changement de face vers face 2 au frame", file)
                world_points = world_points_face2
                image_points = image_points_face2
            elif face == 2:
                print("Changement de face vers face 3 au frame", file)
                world_points = world_points_face3
                image_points = image_points_face3
            elif face == 3:
                print("Changement de face vers face 4 au frame", file)
                world_points = world_points_face4
                image_points = image_points_face4
        else: 
            # Trouver les correspondances de caractéristiques et mettre à jour les points d'image
            image_points, old_pts, new_pts, kp1, kp2, matches = find_feature_matches(gray_old, gray_new, sift, bf, image_points)

        # Traitement si des points d'image sont trouvés
        if image_points is not None:
            # Affichage des correspondances de caractéristiques
            matching_result = cv2.drawMatches(gray_old, kp1, gray_new, kp2, matches, None)
            cv2.namedWindow("Appariements SIFT", cv2.WINDOW_NORMAL)
            cv2.imshow("Appariements SIFT", matching_result)

            # Calcul de la position de la caméra
            ret, rvec, tvec = cv2.solvePnP(world_points, image_points, K, None)
            H_total = np.eye(4)
            R_init, _ = cv2.Rodrigues(rvec)
            H_total[:3, :3] = R_init
            H_total[:3, 3] = tvec.ravel()
            H_total = np.linalg.inv(H_total)
            camera_position.append(H_total[:3, 3])
        else:
            print("Points d'image non trouvés !!")

        gray_old = gray_new.copy()  # Mise à jour de l'ancienne image pour le prochain cycle
    else:
        # Traitement pour la première image
        if image_points is not None:
            # Calcul initial de la position de la caméra
            ret, rvec, tvec = cv2.solvePnP(world_points, image_points, K, None)
            R_init, _ = cv2.Rodrigues(rvec)
            H_total[:3, :3] = R_init
            H_total[:3, 3] = tvec.ravel()
            H_total = np.linalg.inv(H_total)
            camera_position.append(H_total[:3, 3])
        else:
            break  # Sortie de la boucle si aucun point d'image n'est trouvé

    old_frame = frame  # Mise à jour de l'image précédente

    # Sortie de la boucle si la touche 'q' est enfoncée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#%%
# Charger la trajectoire réelle
file_path = 'trajectoire_reelle.csv'
trajectoire_df = pd.read_csv(file_path)
trajectoire_reelle = trajectoire_df.to_numpy()

# Convertir les données en unités compatibles (si nécessaire)
trajectoire_mesuree = np.array(camera_position) / 10


# Tracer les deux trajectoires
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot(trajectoire_mesuree[:, 0], trajectoire_mesuree[:, 1], trajectoire_mesuree[:, 2], label='Trajectoire Mesurée', marker='o')
ax.plot(trajectoire_reelle[:, 0], trajectoire_reelle[:, 1], trajectoire_reelle[:, 2], label='Trajectoire Réelle', linestyle='-')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Comparaison des Trajectoires')
ax.legend()
plt.show()
 

# # Calcul des erreurs
erreurs = np.linalg.norm(trajectoire_mesuree - trajectoire_reelle, axis=1)

# Tracer la courbe d'erreur
plt.figure()
plt.plot(erreurs)
plt.xlabel('Index du Point')
plt.ylabel('Erreur')
plt.title('Courbe d’Erreur entre Trajectoires Réelle et Mesurée')
plt.show()

# Histogramme des erreurs
plt.figure()
plt.hist(erreurs, bins=20)
plt.xlabel('Erreur')
plt.ylabel('Fréquence')
plt.title('Histogramme des Erreurs')
plt.show()
#%%
# Tracé de la trajecoitre dans le plan 2d
plt.figure()
plt.plot(trajectoire_mesuree[:, 0], trajectoire_mesuree[:, 2], label='Trajectoire Mesurée', marker='o')
plt.plot(trajectoire_reelle[:, 0], trajectoire_reelle[:, 2], label='Trajectoire Réelle', linestyle='-')
plt.xlabel('X')
 
plt.ylabel('Z')
plt.title('Comparaison des Trajectoires de la caméra dans le plan 2d')
plt.legend()
plt.show()