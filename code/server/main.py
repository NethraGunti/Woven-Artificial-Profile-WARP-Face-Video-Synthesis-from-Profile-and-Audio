# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# -*- coding: utf-8 -*-
"""train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13Lcuu_MLdu9ZXziFCLjH28pHrzshzHDS
"""

import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
)
from model import Discriminator, StyleGenerator
from math import log2
from tqdm import tqdm
import pandas as pd
import config
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmarks = True


def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.OUT_CHANNLES)],
                [0.5 for _ in range(config.OUT_CHANNLES)]
            ),
        ]
    )

    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(root=config.DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return loader, dataset


def train_fn(
        critic,
        gen,
        loader,
        dataset,
        step,
        alpha,
        opt_critic,
        opt_gen,
        # tensorboard_step,
        # writer,
        # scaler_gen,
        # scaler_critic,
        epoch,
):
    loop = tqdm(loader, leave=True)
    avg_gen_loss, avg_disc_loss = [], []
    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        noise = torch.randn(cur_batch_size, config.L_DIM).to(config.DEVICE)

        fake = gen(noise, step, alpha)
        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake.detach(), alpha, step)
        gp = gradient_penalty(critic, real, fake, alpha, train_step=step, device=config.DEVICE)
        loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMDA_GP * gp
                + (0.001 * torch.mean(critic_real ** 2))
        )

        opt_critic.zero_grad()
        # scaler_critic.scale(loss_critic).backward()
        loss_critic.backward()
        # scaler_critic.step(opt_critic)
        opt_critic.step()
        # scaler_critic.update()

        # with torch.cuda.amp.autocast():
        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        # scaler_gen.scale(loss_gen).backward()
        # scaler_gen.step(opt_critic)
        # scaler_gen.update()
        loss_gen.backward()
        opt_gen.step()

        alpha += cur_batch_size / (len(dataset) * config.PROGRESSIVE_EPOCHS[step] * 0.5)
        alpha = min(alpha, 1)

        # if batch_idx % 500 == 0:
        #     with torch.no_grad():
        #         fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 * 0.5
        #     # plot_to_tensorboard(
        #     #     writer,
        #     #     loss_critic.item(),
        #     #     loss_gen.item(),
        #     #     real.detach(),
        #     #     fixed_fakes.detach(),
        #     #     tensorboard_step
        #     # )
        #     # tensorboard_step += 1
        avg_gen_loss.append(loss_gen.item())
        avg_disc_loss.append(loss_critic.item())
        loop.set_postfix(disc_loss=loss_critic.item(), gen_loss=loss_gen.item())
    generate_examples(gen, 7, epoch, n=1)

    return alpha, avg_disc_loss, avg_gen_loss


def main():
    gen = StyleGenerator()
    critic = Discriminator(config.IN_CHANNELS)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    # scaler_critic = torch.cuda.amp.GradScaler()
    # scaler_gen = torch.cuda.amp.GradScaler()

    # writer = SummaryWriter(f"logs/gen1")

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,
        )

    gen.train()
    critic.train()
    # tensorboard_step = 0
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))
    for num_epochs in config.PROGRESSIVE_EPOCHS[step:6]:
        alpha = 1e-5
        loader, dataset = get_loader(4 * 2 ** step)
        print(f"Image size: {4 * 2 ** step}")

        for epoch in range(num_epochs):
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            alpha, avg_disc_loss, avg_gen_loss = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                # tensorboard_step,
                # writer,
                # scaler_gen,
                # scaler_critic,
                epoch,
            )
            plt.clf()
            plt.plot(avg_gen_loss, label="Generator Loss")
            plt.plot(avg_disc_loss, label="Discrimator Loss")
            plt.legend()
            plt.savefig(f"/content/drive/MyDrive/BTP/Graphs/Loss_Graph_epoch{epoch+17}")

            values = {"disc":avg_disc_loss, "gen":avg_gen_loss}
            df = pd.DataFrame(values)
            df.to_csv(f"/content/drive/MyDrive/BTP/Graphs/Loss_epoch{epoch+17}.csv")
            

            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC)
        step += 1


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
