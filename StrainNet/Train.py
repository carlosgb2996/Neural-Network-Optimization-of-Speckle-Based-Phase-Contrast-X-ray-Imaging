# -*- coding: utf-8 -*-
import argparse
import os
import time
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import models
import pandas as pd
import numpy as np
from multiscaleloss import multiscaleEPE, realEPE
import datetime
from tensorboardX import SummaryWriter
from util import AverageMeter, save_checkpoint

BASE_PATH = '/hpcfs/home/fisica/s.naranjob/StrainNet' # Cluster
#BASE_PATH = os.path.join('content') # Colab

if torch.cuda.is_available():
    device = torch.device('cuda')  # Usar la GPU
    print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')  # Usar la CPU
    print("CUDA no está disponible, usando CPU.")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='StrainNet Training on speckle dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--arch', default='StrainNet_f', choices=['StrainNet_f','StrainNet_h'],
                    help='network f or h')                    
parser.add_argument('--solver', default='adam', choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--multiscale-weights', '-w', default=[0.005,0.01,0.02,0.08,0.32], type=float, nargs=5,
                    help='training weight for each scale, from highest resolution (flow2) to lowest (flow6)',
                    metavar=('W2', 'W3', 'W4', 'W5', 'W6'))
parser.add_argument('--sparse', action='store_true',
                    help='look for NaNs in target flow when computing EPE, avoid if flow is guaranteed to be dense')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')
parser.add_argument('--div-flow', default=2,
                    help='value by which flow will be divided. Original value is 2')
parser.add_argument('--milestones', default=[40,80,120,160,200,240], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')

best_EPE = -1
n_iter = 0

class SpecklesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.Speckles_frame = pd.read_csv(csv_file) 
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.Speckles_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        Ref_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 0])
        Def_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 1])
        Dispx_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 2])
        Dispy_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 3])
       
        Ref = np.genfromtxt(Ref_name, delimiter=',')
        Def = np.genfromtxt(Def_name, delimiter=',')
        Dispx = np.genfromtxt(Dispx_name, delimiter=',')
        Dispy = np.genfromtxt(Dispy_name, delimiter=',')

        Ref = Ref[np.newaxis, ...]  # Añadir una dimensión de canal      
        Def = Def[np.newaxis, ...]
        Dispx = Dispx[np.newaxis, ...]
        Dispy = Dispy[np.newaxis, ...]

        sample = {'Ref': Ref, 'Def': Def, 'Dispx': Dispx, 'Dispy': Dispy}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Normalization(object):
    """Normaliza las imágenes y los desplazamientos."""
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

def main():
    global args, best_EPE
    args = parser.parse_args()
    
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

    # Data loading
    transform = transforms.Compose([Normalization()])
        
    train_set = SpecklesDataset(csv_file=os.path.join(BASE_PATH, 'Dataset', 'Train_annotations.csv'), root_dir=os.path.join(BASE_PATH, 'Dataset', 'Train_Data'), transform=transform)
    test_set = SpecklesDataset(csv_file=os.path.join(BASE_PATH, 'Dataset', 'Test_annotations.csv'), root_dir=os.path.join(BASE_PATH, 'Dataset', 'Test_Data'), transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    print('{} samples found, {} train samples and {} test samples '.format(len(test_set)+len(train_set), len(train_set), len(test_set)))

    # Crear el modelo
    model = models.__dict__[args.arch](data=None).to(device)  # Cambiado network_data a data
    model = torch.nn.DataParallel(model).to(device)
    cudnn.benchmark = True

    # Definir el optimizador antes de cargar el checkpoint
    assert args.solver in ['adam', 'sgd']
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr, betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.momentum)

    if args.pretrained:
        # Cargar el checkpoint
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        
        # Restaurar la época si está disponible
        if 'epoch' in checkpoint:
            args.start_epoch = checkpoint['epoch']
        
        # Restaurar el optimizador si está disponible
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        print(f"=> loaded checkpoint '{args.pretrained}' (epoch {checkpoint['epoch']}))")
    else:
        print('Creating model from scratch')

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    # Parámetros para early stopping
    early_stopping_patience = 10
    no_improve_epochs = 0

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

    batch_time = AverageMeter()
    flow2_EPEs = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, batch in enumerate(val_loader):
        target_x = batch['Dispx'].to(device)       
        target_y = batch['Dispy'].to(device) 
        target = torch.cat([target_x, target_y], 1).to(device)
              
        # Repetir las imágenes de entrada 3 veces, como en el entrenamiento
        in_ref = batch['Ref'].float().to(device) 
        in_ref = torch.cat([in_ref, in_ref, in_ref], 1).to(device)
        
        in_def = batch['Def'].float().to(device) 
        in_def = torch.cat([in_def, in_def, in_def], 1).to(device)
        
        input = torch.cat([in_ref, in_def], 1).to(device)

        # compute output
        output = model(input)
        flow2_EPE = args.div_flow * realEPE(output, target, sparse=args.sparse)
        # record EPE
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                  .format(i, len(val_loader), batch_time, flow2_EPEs))
                  
    print(' * EPE {:.3f}'.format(flow2_EPEs.avg))

    return flow2_EPEs.avg

if __name__ == '__main__':
    main()