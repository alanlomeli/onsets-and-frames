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
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

        
os.environ['XRT_TPU_CONFIG'] = 'tpu_worker;0;10.0.0.2:8470'

ex = Experiment('train_transcriber')




def _run():

    logdir = constants.LOG_DIR + '/transcriber-' + datetime.now().strftime('%y%m%d-%H%M%S')
    writer = SummaryWriter(logdir)
    TRAIN_BATCH_SIZE = 128
    EPOCHS = 1000000
    resume_iteration = None

    checkpoint_interval = 1000
    
    sequence_length = 327680
    model_complexity = 48
    learning_rate = 0.0006 * xm.xrt_world_size()
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98
    leave_one_out = None
    clip_gradient_norm = 3
    validation_length = sequence_length
    validation_interval = 500

    def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
        model.train()
        loop = tqdm(range(resume_iteration + 1, EPOCHS + 1))
        for i, batch in zip(loop, cycle(data_loader)):

            optimizer.zero_grad()
            predictions, losses = model.run_on_batch(batch)

            loss = sum(losses.values())

            loss.backward()
            xm.optimizer_step(optimizer)  # Use this for TPU optimization

            if clip_gradient_norm:
                clip_grad_norm_(model.parameters(), clip_gradient_norm)

            if scheduler is not None:
                scheduler.step()

            for key, value in {'loss': loss, **losses}.items():
                writer.add_scalar(key, value.item(), global_step=i)

            if i % checkpoint_interval == 0:
                xm.save(model, os.path.join(logdir, f'model-{i}.pt'))
                xm.save(optimizer.state_dict(), os.path.join(logdir, 'last-optimizer-state.pt'))



    train_groups, validation_groups = ['train'], ['validation']

    all_years = {'2004', '2006', '2008', '2009', '2011', '2013', '2014', '2015', '2017', '2018'}
    train_groups = list(all_years)
    validation_groups = [str(leave_one_out)]
    dataset = MAESTRO(groups=train_groups, sequence_length=sequence_length)

    train_data_loader = DataLoader(dataset, TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)


    device = xm.xla_device()

    if resume_iteration is None:
        model = OnsetsAndFrames(N_MELS, MAX_MIDI - MIN_MIDI + 1, model_complexity).to(device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        resume_iteration = 0
    else:
        model_path = os.path.join(logdir, f'../model-{resume_iteration}.pt')
        model = torch.load(model_path, map_location=device)
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        optimizer.load_state_dict(torch.load(os.path.join(logdir, 'last-optimizer-state.pt')))

    model = model.to(device)



    summary(model)

    scheduler = StepLR(optimizer, step_size=learning_rate_decay_steps, gamma=learning_rate_decay_rate)

    for epoch in range(EPOCHS):
        para_loader = pl.ParallelLoader(train_data_loader, [device])
        train_loop_fn(para_loader.per_device_loader(device), model, optimizer, device, scheduler=scheduler)


def _mp_fn(rank, flags):
    a = _run()

FLAGS={}
xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')