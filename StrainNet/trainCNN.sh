#!/bin/bash

# ###### Zona de Parámetros de solicitud de recursos a SLURM ############################
#
#SBATCH --job-name=TrainCNN           # Nombre del job
#SBATCH -p gpu                        # Cola a usar, Default=short (Ver colas y límites en /hpcfs/shared/README/partitions.txt)
#SBATCH -N 1                          # Nodos requeridos, Default=1
#SBATCH -n 1                          # Tasks paralelos, recomendado para MPI, Default=1
#SBATCH --cpus-per-task=1             # Cores requeridos por task, recomendado para multi-thread, Default=1
#SBATCH --mem=24G                     # Memoria en Mb por CPU, Default=2048
#SBATCH --gres=gpu:1                  # Solicita 1 GPU
#SBATCH --time=10-00:00:00            # Tiempo máximo de corrida, Default=2 horas
#SBATCH --mail-user=s.naranjob@uniandes.edu.co
#SBATCH --mail-type=ALL               
#SBATCH -o Train.o%j			            #Nombre de archivo de salida
#SBATCH -e Train_error.e%j			      #Nombre de archivo de salida
#
########################################################################################

# ################## Zona Carga de Módulos ############################################
module load anaconda
source activate StrainNet
pip install 'numpy<2.0'

########################################################################################


# ###### Zona de Ejecución de código y comandos a ejecutar secuencialmente #############
host=`/bin/hostname`
date=`/bin/date`

echo "Host: $(/bin/hostname)" 
echo "Started at: $(/bin/date)"

#Codigo a correr

source activate StrainNet
python Train.py

echo -e "Finalicé la ejecución del script \n"
########################################################################################

