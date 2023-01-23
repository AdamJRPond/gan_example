import os
import random

import numpy
import torch

import gan_trainer.config.train_config as conf

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader(data, batch_size):
    g = torch.Generator()
    g.manual_seed(conf.seed)

    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g
    )


def load_checkpoint(generator, discriminator, gen_optimizer, disc_optimizer, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print(f"=> loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)

        start_epoch = checkpoint['epoch']

        generator.load_state_dict(checkpoint['gen_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])

        discriminator.load_state_dict(checkpoint['disc_state_dict'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])

    else:
        print(f"=> no checkpoint found at '{filename}'")

    return generator, discriminator, gen_optimizer, disc_optimizer, start_epoch
