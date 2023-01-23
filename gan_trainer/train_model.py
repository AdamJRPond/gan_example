import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.nn import functional as F
from torch.utils import data
from torchvision import datasets, transforms
import torchvision.utils as vutils

import gan_trainer.config.train_config as conf
import gan_trainer.utils as ut
from gan_trainer.classes import Generator, Discriminator


def training_procedure():
    print('Training started...')

    global_step = 0
    for epoch in range(conf.n_epochs):
        for i, (image_batch, _) in enumerate(train_loader):
            # Set to training mode
            generator.train()
            discriminator.train()

            image_batch = image_batch.to(device)
            # Assign 1 for real label; 0 for fake label
            label_real = torch.ones(image_batch.size(0)).to(device)
            label_fake = torch.zeros(image_batch.size(0)).to(device)

            # Generate a batch of samples from the latent prior
            latent = torch.randn(image_batch.size(0), 100, 1, 1).to(device)
            fake_image_batch = generator(latent).to(device)

            real_pred = discriminator(image_batch).squeeze().to(device)
            fake_pred = discriminator(fake_image_batch.detach()).squeeze().to(device)

            disc_loss = 0.5 * (
                F.binary_cross_entropy(real_pred, label_real) +
                F.binary_cross_entropy(fake_pred, label_fake)
            )

            disc_optimizer.zero_grad()

            # Discriminator backpropogation
            disc_loss.backward()
            disc_optimizer.step()

            fake_pred = discriminator(fake_image_batch).squeeze().to(device)
            gen_loss = F.binary_cross_entropy(fake_pred, label_real)

            gen_optimizer.zero_grad()

            # Generator backpropogation
            gen_loss.backward()
            gen_optimizer.step()

            # Output training stats
            print(f'Epoch: [{epoch}/{conf.n_epochs}] | Step: [{i}/{len(train_loader)}] | Loss_D: {disc_loss.item():.4f} | Loss_G: {gen_loss.item():.4f}|')

            mlflow.log_metric("Train Gen. Loss", gen_loss, step=global_step)
            mlflow.log_metric("Train Disc. Loss", disc_loss, step=global_step)

            global_step += 1

        # Traditional validation procedure code would go here...
        for i, (image_batch, _) in enumerate(val_loader):
            # Set to eval mode
            generator.eval()
            discriminator.eval()

            image_batch = image_batch.to(device)
            # Assign 1 for real label; 0 for fake label
            label_real = torch.ones(image_batch.size(0)).to(device)
            label_fake = torch.zeros(image_batch.size(0)).to(device)

            # Generate a batch of samples from the latent prior
            latent = torch.randn(image_batch.size(0), 100, 1, 1).to(device)
            fake_image_batch = generator(latent).to(device)

            real_pred = discriminator(image_batch).squeeze().to(device)
            fake_pred = discriminator(fake_image_batch.detach()).squeeze().to(device)

            disc_loss = 0.5 * (
                F.binary_cross_entropy(real_pred, label_real) +
                F.binary_cross_entropy(fake_pred, label_fake)
            )

            mlflow.log_metric("Validation Disc. Loss", disc_loss, step=global_step)

        if epoch % conf.log_gen_imgs_every == 0:

            # Create and save fake image generated from random noise
            fixed_noise = torch.randn(conf.n_gen_feats, conf.z_size, 1, 1).to(device)
            fake = generator(fixed_noise)

            img_path = f'artifacts/images/gen_examples_epoch_{epoch}.png'
            with open(img_path, 'wb') as f:
                vutils.save_image(
                    fake.detach(),
                    f,
                    normalize=True
                )

            mlflow.log_artifact(img_path)

        if epoch % conf.ckpt_model_every == 0:
            ckpt_path = f'artifacts/models/checkpoints/gan_model_ckpt_{epoch}.pt'
            torch.save(
                {
                    'epoch': epoch,
                    'disc_state_dict': discriminator.state_dict(),
                    'gen_state_dict': generator.state_dict(),
                    'disc_optimizer_state_dict': disc_optimizer.state_dict(),
                    'gen_optimizer_state_dict': gen_optimizer.state_dict()
                },
                ckpt_path
            )
            mlflow.log_artifact(ckpt_path, "models/checkpoint/")

    # Save real image samples
    real_img_path = 'artifacts/images/real_samples.png'
    with open(real_img_path, 'wb') as f:
        vutils.save_image(
            image_batch,
            f,
            normalize=True
        )
    mlflow.log_artifact(real_img_path)

    print("========= Training finished! =========")

def testing_procedure():
    print('Testing started...')
    with torch.inference_mode():
        preds = []
        labels = []
        for image_batch, _ in test_loader:
            image_batch = image_batch.to(device)
            # Assign 1 for real label; 0 for fake label
            label_real = torch.ones(image_batch.size(0)).to(device)
            label_fake = torch.zeros(image_batch.size(0)).to(device)
            labels.extend(label_real.data.cpu().numpy())
            labels.extend(label_fake.data.cpu().numpy())

            # Generate a batch of samples from the latent prior
            latent = torch.randn(image_batch.size(0), 100, 1, 1).to(device)
            fake_image_batch = generator(latent).to(device)
            real_pred = discriminator(image_batch).squeeze().to(device)
            fake_pred = discriminator(fake_image_batch.detach()).squeeze().to(device)
            preds.extend(real_pred.data.cpu().numpy())
            preds.extend(fake_pred.data.cpu().numpy())

        preds = [1 if p > 0.5 else 0 for p in preds]

        # constant for classes
        classes = ('Real', 'Fake')

        # Build confusion matrix
        cf_matrix = confusion_matrix(labels, preds)

        acc_score = accuracy_score(labels, preds)
        mlflow.log_metric("Disc. Accuracy", acc_score)

        df_cm = pd.DataFrame(
            cf_matrix / np.sum(cf_matrix) * 100,
            index=[i for i in classes],
            columns=[i for i in classes]
        )

        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        cm_path = 'artifacts/images/test_cm.png'
        plt.savefig(cm_path)

        mlflow.log_artifact(cm_path)

        print("========= Testing finished! =========")


if __name__ == "__main__":
    import argparse

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")

    opt = parser.parse_args()

    with mlflow.start_run():
        mlflow.log_params(conf.params)

        # Set torch flags to maintain reproducibility
        torch.manual_seed(conf.seed)
        torch.use_deterministic_algorithms(True)

        # CUDA for PyTorch
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.backends.cudnn.benchmark = True
            print("CUDA is available")

        else:
            device = torch.device("cpu")
            print("No GPU found.")

        # Define a transform to resize the data
        transform = transforms.Compose(
            [transforms.Resize(64),
             transforms.ToTensor()]
        )

        mnist_full = datasets.FashionMNIST(
            "./data",
            train=True,
            download=True,
            transform=transform
        )
        mnist_train, mnist_val, _ = data.random_split(
            mnist_full,
            [1024, 256, 58720],
            # [256, 64, 59680],
            # [128, 32, 59840],
            generator=torch.Generator().manual_seed(conf.seed)
        )

        # Batch loaders
        train_loader = ut.build_dataloader(mnist_train, conf.batch_size)
        val_loader = ut.build_dataloader(mnist_val, conf.batch_size)

        # Instantiate model classes and initialise network weights
        generator = Generator(conf.z_size).to(device)
        generator.apply(ut.weights_init)

        discriminator = Discriminator().to(device)
        discriminator.apply(ut.weights_init)

        # Network optimizers
        gen_optimizer = torch.optim.Adam(
            params=generator.parameters(),
            lr=conf.lr,
            betas=conf.betas
        )
        disc_optimizer = torch.optim.Adam(
            params=discriminator.parameters(),
            lr=conf.lr,
            betas=conf.betas
        )

        if opt.checkpoint:
            generator, discriminator, gen_optimizer, disc_optimizer, start_epoch = \
                ut.load_checkpoint(generator, discriminator, gen_optimizer, disc_optimizer, opt.checkpoint)

        # Run training loop
        training_procedure()

        torch.save(
            {
                'gen_state_dict': generator.state_dict(),
            },
            'artifacts/models/finished/generator_model_finished.pt'
        )

        torch.save(
            {
                'gen_state_dict': generator.state_dict(),
            },
            'artifacts/models/finished/discriminator_model_finished.pt'
        )

        mlflow.pytorch.log_model(discriminator, "models", code_paths=["./gan_trainer/"])
        mlflow.pytorch.log_model(generator, "models", code_paths=["./gan_trainer/"])

        # Traditional testing procedure code would go here...
        mnist_full_test = datasets.FashionMNIST(
            "./data",
            train=False,
            download=True,
            transform=transform
        )

        # Splitting data here just to reduce dataset size
        mnist_test, _ = data.random_split(
            mnist_full_test,
            [0.05, 0.95],
            generator=torch.Generator().manual_seed(conf.seed)
        )

        test_loader = ut.build_dataloader(mnist_test, conf.batch_size)

        testing_procedure()
