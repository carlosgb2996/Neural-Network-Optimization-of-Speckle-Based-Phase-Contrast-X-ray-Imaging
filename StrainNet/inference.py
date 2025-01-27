# -*- coding: utf-8 -*-

# Importacion de bibliotecas necesarias para la ejecucion del script
# argparse: Manejo de argumentos desde la linea de comandos.
# pathlib: Facilita la manipulacion de rutas en un estilo mas orientado a objetos.
# torch: Biblioteca principal para aprendizaje profundo y manejo de modelos.
# tqdm: Biblioteca para mostrar barras de progreso en bucles largos.
# numpy: Utilizado para operaciones numericas y manipulacion de datos matriciales.
# warnings: Para manejar y suprimir advertencias especificas.

import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models  # Importa modelos personalizados definidos en un modulo externo
from tqdm import tqdm
import numpy as np
import warnings

# Suprime advertencias relacionadas con futuras implementaciones en PyTorch.
# Esto se hace para evitar mensajes innecesarios al ejecutar el script.
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Configuracion del analizador de argumentos para permitir la ejecucion desde la linea de comandos.
# Este parser facilita al usuario especificar parametros como la arquitectura de la red,
# la ubicacion de los datos y el modelo preentrenado.
parser = argparse.ArgumentParser(
    description='StrainNet Inference',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Da formato a los argumentos con valores predeterminados.
)

# Definicion de argumentos que se pasaran al script.
parser.add_argument('--arch', default='StrainNet_f', choices=['StrainNet_f', 'StrainNet_h', 'StrainNet_l'],
                    help='Arquitectura de la red a utilizar (e.g., StrainNet_f, StrainNet_h, StrainNet_l)')
parser.add_argument('data', metavar='DIR', help='Ruta a la carpeta que contiene las imagenes a procesar.')
parser.add_argument('--pretrained', metavar='PTH', required=True,
                    help='Ruta al archivo del modelo preentrenado en formato .pth')
parser.add_argument('--output', '-o', metavar='DIR', default=None,
                    help='Carpeta donde se guardaran los resultados. Si no se especifica, se creara en la carpeta de datos.')
parser.add_argument("--img-exts", metavar='EXT', default=['csv'], nargs='*', type=str,
                    help="Extensiones de archivo para buscar imagenes (predeterminadamente: .csv)")

# Funcion principal del script
def main():
    """
    La funcion principal maneja todo el flujo de inferencia, incluyendo:
    - La busqueda y validacion de archivos de entrada (imagenes de referencia y de muestra).
    - La carga del modelo preentrenado.
    - El procesamiento de imagenes para predecir mapas de desplazamiento.
    - El guardado de resultados en archivos CSV.
    """
    global args, save_path
    args = parser.parse_args()  # Procesa los argumentos pasados al script desde la linea de comandos.

    # Define la carpeta que contiene las imagenes proporcionadas por el usuario.
    data_dir = Path(args.data)

    # Configuracion de la carpeta de salida donde se guardaran los resultados.
    if args.output is None:
        save_path = data_dir / 'flow'  # Crea una subcarpeta predeterminada llamada 'flow' dentro de 'data'.
    else:
        save_path = Path(args.output)
    save_path.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe, incluyendo directorios padres.

    # Inicializacion para buscar pares de imagenes RefXXX y DefXXX
    img_pairs = []  # Almacena los pares de imagenes para inferencia.
    ref_files = []  # Lista de archivos de referencia encontrados.

    # Recorre las extensiones de archivo especificadas para buscar imagenes que coincidan.
    for ext in args.img_exts:
        # Busca archivos con el prefijo 'Ref' y la extension especificada.
        test_files = list(data_dir.glob(f'Ref*_*.{ext}'))
        ref_files.extend(test_files)

    # Asocia cada archivo Ref con su correspondiente archivo Def.
    for file in ref_files:
        # Busca el archivo 'Def' correspondiente reemplazando 'Ref' en el nombre del archivo.
        def_filename = file.name.replace('Ref', 'Def')
        def_file = file.parent / def_filename
        if def_file.is_file():
            img_pairs.append([file, def_file])  # Agrega el par Ref-Def a la lista de pares.
        else:
            print(f"Advertencia: No se encontro el archivo Def correspondiente a {file.name}")

    # Si no se encuentran pares de imagenes, finaliza el script con un mensaje de error.
    if len(img_pairs) == 0:
        print("No se encontraron pares de imagenes. Verifica los archivos en el directorio de datos.")
        return

    # Determina el dispositivo disponible para la inferencia (GPU o CPU).
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Mensaje indicando el dispositivo utilizado para la inferencia.
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Usando GPU: {gpu_name}")
    else:
        print("Usando CPU")

    # Carga el modelo preentrenado desde el archivo proporcionado.
    try:
        network_data = torch.load(args.pretrained, map_location=device)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    # Inicializa el modelo con los datos cargados y lo prepara para inferencia.
    try:
        model = models.__dict__[args.arch](network_data).to(device)
        model.eval()  # Cambia el modelo al modo de evaluacion (desactiva dropout y batchnorm).
        cudnn.benchmark = True  # Habilita optimizaciones para entradas de tamaño constante.
    except Exception as e:
        print(f"Error al inicializar el modelo: {e}")
        return

    # Procesa cada par de imagenes (Ref y Def) utilizando el modelo.
    for (img1_file, img2_file) in tqdm(img_pairs, desc="Procesando pares de imagenes"):
        try:
            # Carga las imagenes desde archivos CSV.
            img1 = np.loadtxt(img1_file, delimiter=',')
            img2 = np.loadtxt(img2_file, delimiter=',')

            # Normaliza las imagenes (valores entre 0 y 1).
            img1 = img1 / 255.0
            img2 = img2 / 255.0

            # Añade una dimension de canal si las imagenes son 2D.
            if img1.ndim == 2:
                img1 = img1[np.newaxis, ...]
                img2 = img2[np.newaxis, ...]

            # Convierte las imagenes normalizadas en tensores de PyTorch.
            img1 = torch.from_numpy(img1).float()
            img2 = torch.from_numpy(img2).float()

            # Duplica los canales para arquitecturas que lo requieran.
            if args.arch in ['StrainNet_h', 'StrainNet_f']:
                img1 = img1.repeat(3, 1, 1)
                img2 = img2.repeat(3, 1, 1)

            # Concatena las imagenes Ref y Def para formar el tensor de entrada.
            input_var = torch.cat([img1, img2], dim=0).unsqueeze(0).to(device)

            # Realiza la inferencia para calcular los desplazamientos.
            with torch.no_grad():
                output = model(input_var)

            # Ajusta la salida para que coincida con el tamaño de entrada si es necesario.
            input_size = img1.shape[1:]
            output_size = output.shape[-2:]
            if output_size != input_size:
                output = F.interpolate(input=output, size=input_size, mode='bilinear', align_corners=False)

            # Convierte la salida a matrices numpy y desnormaliza los desplazamientos.
            output_to_write = output.cpu().detach().numpy()
            disp_x = output_to_write[0, 0, :, :] * 2.0 - 1.0
            disp_y = output_to_write[0, 1, :, :] * 2.0 - 1.0

            # Genera nombres de archivo para guardar los resultados.
            filenamex = save_path / f"{img1_file.stem.replace('Ref', 'Def')}_disp_x.csv"
            filenamey = save_path / f"{img1_file.stem.replace('Ref', 'Def')}_disp_y.csv"

            # Guarda los mapas de desplazamiento en archivos CSV.
            np.savetxt(filenamex, disp_x, delimiter=',')
            np.savetxt(filenamey, disp_y, delimiter=',')

        except Exception as e:
            print(f"Error procesando el par {img1_file.name} y {img2_file.name}: {e}")

    print("=> Inferencia completada exitosamente.")

if __name__ == '__main__':
    main()
