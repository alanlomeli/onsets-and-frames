import os
from datetime import datetime

import numpy as np
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from onsets_and_frames import *
# from onsets_and_frames import constants 
import onsets_and_frames.constants as constants

#tpu
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

        
os.environ['XRT_TPU_CONFIG'] = 'tpu_worker;0;10.0.0.2:8470'

ex = Experiment('train_transcriber')


iterations = 1000000
resume_iteration = None

device = constants.DEFAULT_DEVICE
checkpoint_interval = 1000
train_on = constants.DATASET
batch_size = 8

sequence_length = 327680
model_complexity = 48
learning_rate = 0.0006
learning_rate_decay_steps = 10000
learning_rate_decay_rate = 0.98
leave_one_out = None
clip_gradient_norm = 3
validation_length = sequence_length
validation_interval = 500


logdir = constants.LOG_DIR + '/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')

def train():
    global resume_iteration
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_groups, validation_groups = ['train'], ['validation']

    all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017', '2018'}
    train_groups = list(all_years)
    validation_groups = [str(leave_one_out)]

    dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length)
    validation_dataset = MAESTRO(groups=validation_groups, sequence_length=sequence_length)

    device = xmp.xla_device()
    loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    loader = pl.ParallelLoader(loader, [device])

    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'../model-{resume_iteration}.pt')
        model = torch.load(model_path, map_location=device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    summary(model)
    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    #spawn thingy
    xmp.spawn(train_fn, args=(resume_iteration, iterations, loader, model, optimizer, scheduler, writer, validation_dataset), nprocs=8, start_method='fork')
    train_fn(resume_iteration, iterations, loader, model, optimizer, scheduler, writer, validation_dataset)

def train_fn(resume_iteration, iterations, loader, model, optimizer, scheduler, writer, validation_dataset):
    loop = tqdm(range(resume_iteration + 1, iterations + 1))
    for i in loop:
        para_loader = pl.ParallelLoader(loader, [device])
        for batch in enumerate(para_loader.per_device_loader(device)):
            predictions, losses = model.run_on_batch(batch)

            loss = sum(losses.values())
            optimizer.zero_grad()

            loss.backward()
            xmp.optimizer_step(optimizer)  # Use this for TPU optimization

            if clip_gradient_norm:
                clip_grad_norm_(model.parameters(), clip_gradient_norm)

            optimizer.step()
            scheduler.step()


            for key, value in {'loss': loss, **losses}.items():
                writer.add_scalar(key, value.item(), global_step=i)

            if i % validation_interval == 0:
                model.eval()
                with torch.no_grad():
                    for key, value in evaluate(validation_dataset, model).items():
                        writer.add_scalar('validation/' + key.replace(' ', '_'), np.mean(value), global_step=i)
                model.train()

            if i % checkpoint_interval == 0:
                xmp.save(model, os.path.join(logdir, f'model-{i}.pt'))
                xmp.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))

if __name__ == '__main__':
    train()