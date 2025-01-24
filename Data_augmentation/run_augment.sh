#!/bin/bash

###### Zona de Parámetros de solicitud de recursos a SLURM ############################
#
#SBATCH --job-name=AugmentData           # Nombre del job
#SBATCH -p gpu                           # Cola a usar, Default=short (Ver colas y límites en /hpcfs/shared/README/partitions.txt)
#SBATCH -N 1                             # Nodos requeridos
#SBATCH -n 1                             # Tasks paralelos
#SBATCH --cpus-per-task=4                # Cores por task
#SBATCH --mem=16G                        # Memoria en Gb
#SBATCH --gres=gpu:1                     # Solicita 1 GPU
#SBATCH --time=10:00:00                  # Tiempo máximo de corrida
#SBATCH --mail-user=s.naranjob@uniandes.edu.co
#SBATCH --mail-type=ALL                  
#SBATCH -o AugmentData.o%j               # Archivo de salida
#SBATCH -e AugmentData_error.e%j         # Archivo de error
#
########################################################################################

# ################## Zona Carga de Módulos ############################################
module load anaconda

# Activar el entorno de Anaconda
source activate StrainNet
pip install numpy scipy scikit-image pandas


########################################################################################

###### Zona de Ejecución de código y comandos a ejecutar secuencialmente #############
host=`/bin/hostname`
date=`/bin/date`

echo "Host: $(/bin/hostname)" 
echo "Started at: $(/bin/date)"

# Navegar al directorio donde está el script de Python
cd /hpcfs/home/fisica/s.naranjob/StrainNet/Data_augmentation             # Reemplaza con la ruta correcta

# Ejecutar el script de Python
python augment_data.py

echo -e "Finalicé la ejecución del script \n"
########################################################################################