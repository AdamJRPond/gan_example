batch_size = 128
b1 = 0.5
b2 = 0.999
betas = (0.5, 0.999)
lr = 0.0002
z_size = 100  # Size of latent variable (z)

n_channels = 1  # Grayscale
n_gen_feats = 32

n_epochs = 40
log_gen_imgs_every = 2
ckpt_model_every = 10
seed = 123

params = {
    "batch_size": batch_size,
    "b1": b1,
    "b2": b2,
    "lr": lr,
    "z_size": z_size,
    "n_channels": n_channels,
    "seed": seed
}
