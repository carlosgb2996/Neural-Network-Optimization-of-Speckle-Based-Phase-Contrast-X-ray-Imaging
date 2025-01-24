# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm
import numpy as np
import warnings

# Supress specific warnings (optional)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# Argument parser
parser = argparse.ArgumentParser(description='StrainNet Inference',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--arch', default='StrainNet_f', choices=['StrainNet_f', 'StrainNet_h', 'StrainNet_l'],
                    help='Network architecture to use')
parser.add_argument('data', metavar='DIR',
                    help='Path to images folder')
parser.add_argument('--pretrained', metavar='PTH', required=True, help='Path to pre-trained model')
parser.add_argument('--output', '-o', metavar='DIR', default=None,
                    help='Path to output folder. If not set, will be created in data folder')
parser.add_argument("--img-exts", metavar='EXT', default=['csv'], nargs='*', type=str,
                    help="Image file extensions to search for")

def main():
    global args, save_path
    args = parser.parse_args()

    data_dir = Path(args.data)

    print(f"=> Searching for image pairs in '{args.data}'")
    if args.output is None:
        save_path = data_dir / 'flow'
    else:
        save_path = Path(args.output)
    print(f"=> Saving all outputs to {save_path}")
    save_path.mkdir(parents=True, exist_ok=True)

    img_pairs = []
    ref_files = []
    for ext in args.img_exts:
        # Corrected glob pattern to match RefXXX_XX.csv
        test_files = list(data_dir.glob(f'Ref*_*.{ext}'))
        ref_files.extend(test_files)

    print(f"Number of Ref files found: {len(ref_files)}")

    for file in ref_files:
        # Replace 'Ref' with 'Def' in the filename to find the corresponding Def file
        def_filename = file.name.replace('Ref', 'Def')
        def_file = file.parent / def_filename
        if def_file.is_file():
            img_pairs.append([file, def_file])
        else:
            print(f"Warning: Def file corresponding to {file.name} not found")

    print(f"Total number of image pairs found: {len(img_pairs)}")

    if len(img_pairs) == 0:
        print("No image pairs found. Please check the files in the data directory.")
        return  # Exit the script if no pairs are found

    # Determine the device to use
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Message indicating whether GPU or CPU is being used
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Running on GPU: {gpu_name}")
    else:
        print("Running on CPU")

    # Load the model (load on GPU or CPU based on availability)
    try:
        if device.type == 'cuda':
            network_data = torch.load(args.pretrained)
        else:
            network_data = torch.load(args.pretrained, map_location=device)
        print(f"=> Model successfully loaded from '{args.pretrained}'")
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    try:
        # Inicializar el modelo con los datos de la red
        model = models.__dict__[args.arch](network_data).to(device)
        model.eval()
        cudnn.benchmark = True
        print(f"=> Using the pre-trained model '{args.arch}'")
    except Exception as e:
        print(f"Error initializing the model: {e}")
        return

    for (img1_file, img2_file) in tqdm(img_pairs, desc="Processing image pairs"):
        try:
            # Load images from CSV files
            img1 = np.loadtxt(img1_file, delimiter=',')
            img2 = np.loadtxt(img2_file, delimiter=',')

            # Normalize the input images the same way as during the training
            img1 = img1 / 255.0
            img2 = img2 / 255.0

            if img1.ndim == 2:
                img1 = img1[np.newaxis, ...]  # Añadir dimensión de canal, shape: [1, H, W]
                img2 = img2[np.newaxis, ...]

            # Convert to PyTorch tensors
            img1 = torch.from_numpy(img1).float()
            img2 = torch.from_numpy(img2).float()

            # If the model requires 3 channels, replicate the channel
            if args.arch in ['StrainNet_h', 'StrainNet_f']:
                img1 = img1.repeat(3, 1, 1)  # shape: [3, H, W]
                img2 = img2.repeat(3, 1, 1)

            # Concatenate the reference and deformed images along the channel dimension
            input_var = torch.cat([img1, img2], dim=0).unsqueeze(0).to(device)  # shape: [1, 6, H, W]

            # Debug: Verificar la forma del input
            # print(f"Input shape: {input_var.shape}")

            # Compute the output
            with torch.no_grad():
                output = model(input_var)

            # Debug: Verificar la forma de la salida del modelo
            print(f"Processing pair {img1_file.name} and {img2_file.name}: Output shape {output.shape}")

            # Check if upsampling is needed
            input_size = img1.shape[1:]  # (H, W)
            output_size = output.shape[-2:]  # (H_out, W_out)
            if output_size != input_size:
                print(f"Upsampling output from {output_size} to {input_size}")
                output = F.interpolate(input=output, size=input_size, mode='bilinear', align_corners=False)

            # Process the output for saving
            output_to_write = output.cpu().detach().numpy()

            # Asegúrate de que output_to_write tenga la forma [1, 2, H, W]
            if output_to_write.ndim != 4 or output_to_write.shape[1] != 2:
                raise ValueError(f"Unexpected output shape: {output_to_write.shape}")

            # Invert the normalization applied during the training
            # Según Train.py: Dispx_normalized = (Dispx - mean1) / std1 => Dispx = Dispx_normalized * std1 + mean1
            # Donde mean1 = -1.0 y std1 = 2.0
            disp_x = output_to_write[0, 0, :, :] * 2.0 - 1.0  # shape: [H, W]
            disp_y = output_to_write[0, 1, :, :] * 2.0 - 1.0  # shape: [H, W]

            # Generar nombres de archivo de salida
            filenamex = save_path / f"{img1_file.stem.replace('Ref', 'Def')}_disp_x.csv"
            filenamey = save_path / f"{img1_file.stem.replace('Ref', 'Def')}_disp_y.csv"

            # Save the outputs as CSV files
            np.savetxt(filenamex, disp_x, delimiter=',')
            np.savetxt(filenamey, disp_y, delimiter=',')

            print(f"Input image shape: {img1.shape}")  # Debería ser [C, H, W]
            print(f"Model output shape: {output.shape}")  # Debería ser [1, 2, H, W]

            # Opcional: Visualizar los desplazamientos para verificar
            """
            plt.figure(figsize=(15,5))
            
            plt.subplot(1,2,1)
            plt.imshow(disp_x, cmap='gray')
            plt.title('Displacement X')
            plt.colorbar()
            
            plt.subplot(1,2,2)
            plt.imshow(disp_y, cmap='gray')
            plt.title('Displacement Y')
            plt.colorbar()
            
            plt.show()
            """

        except Exception as e:
            print(f"Error processing pair {img1_file.name} and {img2_file.name}: {e}")

    print("=> Inference completed successfully.")

if __name__ == '__main__':
    main()
