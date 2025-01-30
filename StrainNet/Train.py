# -*- coding: utf-8 -*-

# Importación de librerías necesarias para el entrenamiento de la red neuronal
import argparse  # Manejo de argumentos de línea de comandos
import os  # Manejo de archivos y directorios
import time  # Medición del tiempo de ejecución
import torch  # Biblioteca principal para redes neuronales
import torch.nn.functional as F  # Funciones de activación y pérdidas
import torch.nn.parallel  # Utilidad para entrenar en múltiples GPUs
import torch.backends.cudnn as cudnn  # Optimizaciones para aceleración en GPU
import torch.optim  # Optimizadores (Adam, SGD, etc.)
import torch.utils.data  # Carga de datos y manejo de datasets
import torchvision.transforms as transforms  # Transformaciones para preprocesamiento
from torch.utils.data import Dataset, DataLoader  # Clases para definir conjuntos de datos
import models  # Módulo que contiene las arquitecturas de redes neuronales
import pandas as pd  # Manejo de datasets en formato CSV
import numpy as np  # Cálculos numéricos y manejo de arrays
from multiscaleloss import multiscaleEPE, realEPE  # Importación de funciones de pérdida
import datetime  # Manejo de fechas y tiempos
from tensorboardX import SummaryWriter  # Para registrar métricas en TensorBoard
from util import AverageMeter, save_checkpoint  # Utilidades para métricas y guardado de modelos

# Definir la ruta base donde se almacenan los datos del proyecto
BASE_PATH = '/hpcfs/home/fisica/s.naranjob/StrainNet'  # Ruta en un cluster HPC
# BASE_PATH = os.path.join('content')  # Alternativa para ejecución en Google Colab

# Configuración del dispositivo: se usa GPU si está disponible, de lo contrario CPU
if torch.cuda.is_available():
    device = torch.device('cuda')  # Usar la GPU disponible
    print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')  # Usar la CPU
    print("CUDA no está disponible, usando CPU.")

# Definir los modelos disponibles en el módulo models
model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))

# Definir los argumentos de línea de comandos para configurar la ejecución del entrenamiento
parser = argparse.ArgumentParser(description='Entrenamiento de StrainNet en un conjunto de datos de moteado',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--arch', default='StrainNet_f', choices=['StrainNet_f', 'StrainNet_h'],
                    help='Selección de arquitectura de la red neuronal')
parser.add_argument('--solver', default='adam', choices=['adam', 'sgd'],
                    help='Algoritmo de optimización a usar (Adam o SGD)')
parser.add_argument('-j', '--workers', default=1, type=int,
                    help='Número de procesos en paralelo para cargar los datos')
parser.add_argument('--epochs', default=300, type=int,
                    help='Número total de épocas de entrenamiento')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='Número de época en el que se empieza el entrenamiento (útil para reanudar entrenamientos)')
parser.add_argument('--epoch-size', default=0, type=int,
                    help='Número de muestras por época (se usa tamaño del dataset si es 0)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    help='Tamaño del mini-batch para el entrenamiento')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    help='Tasa de aprendizaje inicial')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momento para SGD o parámetro alpha en Adam')
parser.add_argument('--beta', default=0.999, type=float,
                    help='Parámetro beta para Adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    help='Factor de decaimiento de peso para regularización')
parser.add_argument('--bias-decay', default=0, type=float,
                    help='Factor de decaimiento de sesgo')
parser.add_argument('--multiscale-weights', '-w', default=[0.005,0.01,0.02,0.08,0.32], type=float, nargs=5,
                    help='Pesos de entrenamiento para cada escala, desde la resolución más alta hasta la más baja')
parser.add_argument('--sparse', action='store_true',
                    help='Considerar valores NaN en el flujo objetivo al calcular EPE, evitar si el flujo es denso')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='Frecuencia con la que se imprimen métricas durante el entrenamiento')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='Ruta del modelo preentrenado (si se desea continuar entrenamiento)')
parser.add_argument('--div-flow', default=2,
                    help='Valor por el cual se divide el flujo. Valor original es 2')
parser.add_argument('--milestones', default=[40, 80, 120, 160, 200, 240], nargs='*',
                    help='Épocas en las que la tasa de aprendizaje se reduce a la mitad')

# Inicialización de métricas globales
best_EPE = -1  # Mejor error registrado (inicialmente indefinido)
n_iter = 0  # Contador global de iteraciones

# Definición de la clase del conjunto de datos de moteado
class SpecklesDataset(Dataset):
    """
    Clase para manejar la carga de datos del conjunto de moteado
    """
    def __init__(self, csv_file, root_dir, transform=None):
        self.Speckles_frame = pd.read_csv(csv_file)  # Leer el archivo CSV con las rutas de las imágenes
        self.root_dir = root_dir  # Directorio donde están almacenadas las imágenes
        self.transform = transform  # Transformaciones opcionales

    def __len__(self):
        return len(self.Speckles_frame)  # Retorna el número de muestras en el dataset

    def __getitem__(self, idx):
        """
        Carga y retorna una muestra del dataset
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Construcción de rutas para las imágenes de referencia y desplazamientos
        Ref_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 0])
        Def_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 1])
        Dispx_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 2])
        Dispy_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 3])
       
        # Carga de datos desde archivos CSV
        Ref = np.genfromtxt(Ref_name, delimiter=',')
        Def = np.genfromtxt(Def_name, delimiter=',')
        Dispx = np.genfromtxt(Dispx_name, delimiter=',')
        Dispy = np.genfromtxt(Dispy_name, delimiter=',')

        # Agregar dimensión de canal a las imágenes y desplazamientos
        Ref = Ref[np.newaxis, ...]  
        Def = Def[np.newaxis, ...]
        Dispx = Dispx[np.newaxis, ...]
        Dispy = Dispy[np.newaxis, ...]

        sample = {'Ref': Ref, 'Def': Def, 'Dispx': Dispx, 'Dispy': Dispy}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Clase para normalizar los datos de entrada
class Normalization(object):
    """
    Normaliza las imágenes y los desplazamientos antes de pasarlos al modelo.
    
    - Las imágenes de referencia y deformadas se normalizan dividiendo por 255.0.
    - Los desplazamientos se normalizan con una media de -1.0 y una desviación estándar de 2.0.
    """
    def __call__(self, sample):
        Ref, Def, Dispx, Dispy = sample['Ref'], sample['Def'], sample['Dispx'], sample['Dispy']

        self.mean = 0.0
        self.std = 255.0        
        self.mean1 = -1.0
        self.std1 = 2.0
        
        return {'Ref': torch.from_numpy((Ref - self.mean) / self.std).float(),
                'Def': torch.from_numpy((Def - self.mean) / self.std).float(),
                'Dispx': torch.from_numpy((Dispx - self.mean1) / self.std1).float(),
                'Dispy': torch.from_numpy((Dispy - self.mean1) / self.std1).float()}

# Función principal del script
def main():
    """
    Función principal que ejecuta el entrenamiento de la red neuronal.
    
    - Configura los parámetros de entrenamiento a partir de los argumentos de línea de comandos.
    - Carga los conjuntos de datos de entrenamiento y validación.
    - Inicializa el modelo y define su optimizador.
    - Configura el aprendizaje y la programación del decaimiento del aprendizaje.
    """
    global args, best_EPE
    args = parser.parse_args()
    
    # Construir el nombre del directorio de guardado de modelos
    save_path = '{},{},{}epochs{},b{},lr{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.lr)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_writer = SummaryWriter(os.path.join(save_path, 'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))

    # Cargar datos con transformaciones
    transform = transforms.Compose([Normalization()])
        
    train_set = SpecklesDataset(csv_file=os.path.join(BASE_PATH, 'Dataset', 'Train_annotations.csv'), root_dir=os.path.join(BASE_PATH, 'Dataset', 'Train_Data'), transform=transform)
    test_set = SpecklesDataset(csv_file=os.path.join(BASE_PATH, 'Dataset', 'Test_annotations.csv'), root_dir=os.path.join(BASE_PATH, 'Dataset', 'Test_Data'), transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    print(f'{len(test_set) + len(train_set)} muestras encontradas, {len(train_set)} para entrenamiento y {len(test_set)} para validación.')

    # Inicializar modelo seleccionado
    model = models.__dict__[args.arch](data=None).to(device)  # Cargar el modelo en el dispositivo
    model = torch.nn.DataParallel(model).to(device)  # Permitir paralelización en múltiples GPUs
    cudnn.benchmark = True  # Optimizar rendimiento en GPU

    # Definir el optimizador antes de cargar un posible checkpoint
    assert args.solver in ['adam', 'sgd']
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr, betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.momentum)

    # Configurar la reducción progresiva de la tasa de aprendizaje
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    # Configurar parámetros para early stopping
    early_stopping_patience = 10  # Número de épocas sin mejora antes de detener el entrenamiento
    no_improve_epochs = 0  # Contador de épocas sin mejora


    for epoch in range(args.start_epoch, args.epochs):
        # Entrenar para una época
        train_loss, train_EPE = train(train_loader, model, optimizer, epoch, train_writer, scheduler)
        train_writer.add_scalar('mean EPE', train_EPE, epoch)

        # Evaluar en el conjunto de prueba
        with torch.no_grad():
            EPE = validate(val_loader, model, epoch)
        test_writer.add_scalar('mean EPE', EPE, epoch)

        # Actualizar mejor EPE y guardar el modelo
        if best_EPE < 0:
            best_EPE = EPE

        is_best = EPE < best_EPE
        if is_best:
            best_EPE = EPE
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_EPE': best_EPE,
            'div_flow': args.div_flow,
            'optimizer': optimizer.state_dict()  # Guarda el estado del optimizador
        }, is_best, save_path)

        # Mover scheduler.step() aquí
        scheduler.step()

        # Early Stopping
        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

def train(train_loader, model, optimizer, epoch, train_writer, scheduler):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    flow2_EPEs = AverageMeter()

    epoch_size = len(train_loader) if args.epoch_size == 0 else min(len(train_loader), args.epoch_size)

    model.train()  # Modo entrenamiento

    end = time.time()

    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        # Extraer los datos del batch
        target_x = batch['Dispx'].to(device)
        target_y = batch['Dispy'].to(device)
        target = torch.cat([target_x, target_y], 1).to(device)

        in_ref = batch['Ref'].float().to(device)
        in_def = batch['Def'].float().to(device)

        # Replicar las imágenes de 1 canal a 3 canales
        in_ref = in_ref.repeat(1, 3, 1, 1)  # Repetir a lo largo de la dimensión de canales
        in_def = in_def.repeat(1, 3, 1, 1)  # Repetir a lo largo de la dimensión de canales

        # Concatenar la imagen de referencia y la deformada (deben tener 6 canales en total)
        input = torch.cat([in_ref, in_def], dim=1).to(device)  # Concatenar en la dimensión de canales (dim=1)

        # Calcular la salida del modelo
        output = model(input)

        # Calcular la pérdida
        loss = multiscaleEPE(output, target, weights=args.multiscale_weights, sparse=args.sparse)
        flow2_EPE = args.div_flow * realEPE(output[0], target, sparse=args.sparse)

        # Actualizar métricas y retropropagar
        losses.update(loss.item(), target.size(0))
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        optimizer.zero_grad()
        loss.backward()

        # Agregar clipping de gradientes
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{epoch_size}] Time {batch_time.val:.3f} Data {data_time.val:.3f} Loss {losses.val:.4f} EPE {flow2_EPEs.val:.4f}')
        
        n_iter += 1

    return losses.avg, flow2_EPEs.avg

def validate(val_loader, model, epoch):
    global args

    batch_time = AverageMeter()  # Para medir el tiempo por batch
    flow2_EPEs = AverageMeter()  # Para almacenar el error EPE (End Point Error)

    # Cambiar a modo evaluación (desactiva dropout, batch normalization, etc.)
    model.eval()

    end = time.time()
    for i, batch in enumerate(val_loader):
        # Extraer los desplazamientos objetivo (ground truth)
        target_x = batch['Dispx'].to(device)       
        target_y = batch['Dispy'].to(device) 
        target = torch.cat([target_x, target_y], 1).to(device)  # Concatenar en la dimensión de canales
              
        # Repetir las imágenes de entrada 3 veces, como en el entrenamiento
        in_ref = batch['Ref'].float().to(device) 
        in_ref = torch.cat([in_ref, in_ref, in_ref], 1).to(device)  # Convertir a 3 canales
        
        in_def = batch['Def'].float().to(device) 
        in_def = torch.cat([in_def, in_def, in_def], 1).to(device)  # Convertir a 3 canales
        
        # Concatenar las imágenes de referencia y deformada para formar la entrada
        input = torch.cat([in_ref, in_def], 1).to(device)

        # Calcular la salida del modelo
        output = model(input)

        # Calcular el error EPE (End Point Error)
        flow2_EPE = args.div_flow * realEPE(output, target, sparse=args.sparse)

        # Actualizar métricas
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # Medir el tiempo transcurrido
        batch_time.update(time.time() - end)
        end = time.time()

        # Imprimir métricas cada cierto número de iteraciones
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                  .format(i, len(val_loader), batch_time, flow2_EPEs))
                  
    # Imprimir el error EPE promedio para la época actual
    print(' * EPE {:.3f}'.format(flow2_EPEs.avg))

    return flow2_EPEs.avg


if __name__ == '__main__':
    main()