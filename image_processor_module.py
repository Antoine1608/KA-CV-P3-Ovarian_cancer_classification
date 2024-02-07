#image_processor_module.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytest
from PIL import Image
from skimage.exposure import match_histograms
import os

class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = self.load_image()

    def load_image(self):
        # Charger l'image à partir du chemin spécifié
        return cv2.imread(self.image_path, 1)

    def display_images(self, images, titles):
        # Afficher les images dans une figure matplotlib
        plt.figure(figsize=(8, 4))
        for i in range(len(images)):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')
        plt.show()

    def apply_clahe_rgb(self, image):
        # Appliquer l'amélioration du contraste (CLAHE) en espace couleur RGB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced_img

    def apply_clahe_grayscale(self):
        # Appliquer l'amélioration du contraste (CLAHE) en niveaux de gris
        image_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        image_resized = cv2.resize(image_gray, (224, 224), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(2, 2))
        clahe_img = clahe.apply(image_resized)
        return clahe_img

    def apply_gabor_grayscale(self):
        # Appliquer le filtre de Gabor en niveaux de gris
        image_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        image_resized = cv2.resize(image_gray, (224, 224), interpolation=cv2.INTER_AREA)
        kernel_size = 5
        theta = 0
        sigma = 2.0
        gamma = 0.5
        psi = 0
        gabor_kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambd=10.0, gamma=gamma, psi=psi)
        gabor_img = cv2.filter2D(image_resized, cv2.CV_8UC3, gabor_kernel)
        return gabor_img

    def normalization_processing(self):
        # Appliquer une normalisation de l'image sur le modèle d'une image de référence ref_path
        ref_path = r"C:\Users\John\Desktop\KA-CL-P2-Ovarian_Cancer_Classification\test_img.jpg"
        
        # Charger l'image en couleur
        ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        
        # Resize des images pour avoir la même taille
        ref_img = cv2.resize(ref_img, (224, 224), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Appliquer la normalisation
        aft_img = match_histograms(image, ref_img)
        
        # Convert the image to uint8 and BGR
        aft_img = cv2.convertScaleAbs(aft_img, cv2.COLOR_LAB2BGR)
        
        return aft_img
    
    def process_and_display(self):
        # Appliquer les traitements et afficher les résultats
        processed_clahe_gray = self.apply_clahe_grayscale()
        processed_gabor = self.apply_gabor_grayscale()
        processed_normalization = self.normalization_processing()
        processed_tiling = self.tiling()

        self.display_images([self.img, processed_clahe_gray, processed_gabor, processed_normalization, processed_tiling],
                            ['Image originale', 'Image modifiée avec CLAHE gris', 'Image modifiée avec Gabor gris', 'Image normalisée', 'Image tiling'])

    def pad_to_size(self, tile, target_size):
        # Calcule la quantité de padding nécessaire
        pad_height = max(0, target_size[0] - tile.shape[0])
        pad_width = max(0, target_size[1] - tile.shape[1])
    
        # Ajoute des colonnes de zéros à gauche et à droite
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
    
        # Ajoute des lignes de zéros en haut et en bas
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
    
        # Applique le padding
        padded_tile = np.pad(tile, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
    
        return padded_tile

    def tiling(self, tile_size=256, pix_threshold=0.6):
        # Découper l'image en petits carreaux
        # 1. Charger l'image
        #image_path = r"C:\Users\John\Desktop\KA-CL-P2-Ovarian_Cancer_Classification\img_paris.png"
        image = Image.open(self.image_path)
        
        # 2. Convertir en numpy array
        image_array = np.array(image)
        
        # 3. Tiling de l'image en carreaux de tile_size par tile_size
        #tile_size = 256
        tiles = [self.pad_to_size(image_array[i:i+tile_size, j:j+tile_size], (tile_size, tile_size, 3)) for i in range(0, image_array.shape[0], tile_size) for j in range(0, image_array.shape[1], tile_size)]
        
        # 4. Élimination des carreaux de taille inférieure à un certain threshold
        threshold = pix_threshold * tile_size**2 * 3 # Vous pouvez ajuster ce seuil en fonction de vos besoins
        filtered_tiles = [tile for tile in tiles if np.sum(tile) > threshold]
        
        # 5. Reconstruction d'un numpy array carré avec les tiles restant
        num_tiles_side = int(np.sqrt(len(filtered_tiles)))
    
        if (len(filtered_tiles) - num_tiles_side**2) != 0:
            num_blank_tiles = (num_tiles_side+1)**2 - num_tiles_side**2
            num_tiles_side = num_tiles_side + 1
        else:
            num_blank_tiles = 0
            
        # Création de tiles blancs
        blank_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        
        # Remplissage des tiles manquants avec des tiles blanches
        filled_tiles = filtered_tiles + [blank_tile] * num_blank_tiles
        
        # Reconstruction du numpy array carré
        reconstructed_array = np.vstack([np.hstack(filled_tiles[i*num_tiles_side:(i+1)*num_tiles_side]) for i in range(num_tiles_side)])
        
        # 6. Enregistrement au format png
        reconstructed_image = Image.fromarray(reconstructed_array)
        
        base_name = os.path.basename(self.image_path)
        # Create the folder if it doesn't exist
        os.makedirs(r"reconstructed/", exist_ok=True)
        folder = r"reconstructed/"
        full_name = ''.join([folder, base_name])
        reconstructed_image.save(full_name)
        return reconstructed_image

# Tests pytest

def test_load_image():
    # Vérifier que la fonction load_image renvoie une image non nulle
    image_processor = ImageProcessor(r".\test_img.jpg")
    assert image_processor.load_image() is not None

def test_apply_clahe_rgb():
    # Vérifier que la fonction apply_clahe_rgb renvoie une image non nulle
    image_processor = ImageProcessor(r".\test_img.jpg")
    image_processor.img = image_processor.load_image()  # Charger l'image ici
    assert image_processor.img is not None  # Vérifier que l'image est chargée correctement
    assert image_processor.apply_clahe_rgb(image_processor.img) is not None

def test_apply_clahe_grayscale():
    # Vérifier que la fonction apply_clahe_grayscale renvoie une image non nulle
    image_processor = ImageProcessor(r".\test_img.jpg")
    image_processor.img = image_processor.load_image()  # Charger l'image ici
    assert image_processor.img is not None  # Vérifier que l'image est chargée correctement
    assert image_processor.apply_clahe_grayscale() is not None

def test_apply_gabor_grayscale():
    # Vérifier que la fonction apply_gabor_grayscale renvoie une image non nulle
    image_processor = ImageProcessor(r".\test_img.jpg")
    image_processor.img = image_processor.load_image()  # Charger l'image ici
    assert image_processor.img is not None  # Vérifier que l'image est chargée correctement
    assert image_processor.apply_gabor_grayscale() is not None

def test_normalization_processing():
    # Vérifier que la fonction apply_gabor_grayscale renvoie une image non nulle
    image_processor = ImageProcessor(r".\test_img.jpg")
    image_processor.img = image_processor.load_image()  # Charger l'image ici
    assert image_processor.img is not None  # Vérifier que l'image est chargée correctement
    assert image_processor.normalization_processing() is not None

def test_tiling():
    # Vérifier que la fonction apply_clahe_rgb renvoie une image non nulle
    image_processor = ImageProcessor(r".\test_img.jpg")
    image_processor.img = image_processor.load_image()  # Charger l'image ici
    assert image_processor.img is not None  # Vérifier que l'image est chargée correctement
    assert image_processor.tiling(tile_size=25, pix_threshold=255*0.4) is not None

if __name__ == "__main__":
    # Exécuter les tests avec pytest si le script est exécuté en tant que programme principal
    pytest.main([__file__])

