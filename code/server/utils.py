# -*- coding: utf-8 -*-
"""utils.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Rnkq2VXUAyNv_FW7tnuIkG61r0jCuCc7
"""

import torch
import random
import numpy as np
import os
import torchvision
import config
from torchvision.utils import save_image


def plot_to_tensorboard(writer, loss_critic, loss_gen, real, fake, tensorboard_step):
  writer.add_scalar("Loss Critic", loss_critic, global_step=tensorboard_step)

  with torch.no_grad():
    img_grid_real = torchvision.utils.make_grid(real[:8], normalize=True)
    img_grid_fake = torchvision.utils.make_grid(fake[:8], normalize=True)
    writer.add_image("Real", img_grid_real, global_step=tensorboard_step)
    writer.add_image("Fake", img_grid_fake, global_step=tensorboard_step)


def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
  BATCH_SIZE, C, H, W = real.shape
  beta = torch.randn((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)

  interpolated_images = real * beta + fake.detach() * (1 - beta)
  interpolated_images.requires_grad_(True)

  mixed_scores = critic(interpolated_images, alpha, train_step)

  gradient = torch.autograd.grad(
      inputs=interpolated_images,
      outputs=mixed_scores,
      grad_outputs=torch.ones_like(mixed_scores),
      create_graph=True,
      retain_graph=True,
  )[0]
  gradient = gradient.view(gradient.shape[0], -1)
  gradient_norm = gradient.norm(2, dim=1)
  gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
  return gradient_penalty


def save_checkpoint(model, optimizer, filename="StyleGAN_checkpoint.pth.tar"):
  print("=> Saving checkpoint")
  checkpoint = {
      "state_dict": model.state_dict(),
      "optimizer": optimizer.state_dict(),
  }
  torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer=None, lr=None):
  print("=> Loading checkpoint")
  checkpoint = torch.load(checkpoint_file, map_location="cpu")
  model.load_state_dict(checkpoint["state_dict"])
  if optimizer:
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
      param_group["lr"] = lr


def generate_examples(gen, steps, epoch, n=100):
  gen.eval()
  alpha = 0.3
  for i in range(n):
    with torch.no_grad():
      noise = torch.zeros(1, config.L_DIM).to(config.DEVICE)
      img = gen(noise, steps, alpha)
      save_image(img * 0.5 * 0.5, f"/content/drive/MyDrive/BTP/savedExamples/saved_examples_{epoch+16}_{i}.png")

  gen.train()