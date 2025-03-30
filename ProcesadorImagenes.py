import os
import numpy as np
from PIL import Image

def convert_txt_to_tif(folder_path):
    """
    Convierte todos los archivos .txt en la carpeta especificada a imágenes .tif y luego elimina los archivos .txt.
    
    :param folder_path: Ruta de la carpeta donde están los archivos .txt.
    """
    if not os.path.exists(folder_path):
        print("La carpeta especificada no existe.")
        return
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            old_path = os.path.join(folder_path, filename)
            new_filename = filename.rsplit(".", 1)[0] + ".tif"
            new_path = os.path.join(folder_path, new_filename)
            
            # Leer los datos del archivo de texto
            with open(old_path, "r") as file:
                data = [list(map(float, line.split())) for line in file]
            
            # Convertir en un array de NumPy
            image_array = np.array(data)
            
            # Normalizar los datos a valores entre 0 y 255
            image_array_normalized = (255 * (image_array - np.min(image_array)) / 
                                      (np.max(image_array) - np.min(image_array))).astype(np.uint8)
            
            # Crear imagen y guardarla como TIFF
            image = Image.fromarray(image_array_normalized)
            image.save(new_path)
            print(f"Convertido: {filename} -> {new_filename}")
            
            # Eliminar el archivo .txt original
            os.remove(old_path)
            print(f"Eliminado: {filename}")

# Uso
directory = r"C:\Users\admin\Desktop\Andes\Semestre 8\Experimental\Red Neuronal\Neural-Network-Optimization-of-Speckle-Based-Phase-Contrast-X-ray-Imaging\Data_augmentation\drive-download-20250224T040520Z-001"
convert_txt_to_tif(directory)
