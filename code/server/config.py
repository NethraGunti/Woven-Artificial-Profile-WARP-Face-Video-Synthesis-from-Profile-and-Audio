import torch
from math import log2

START_TRAIN_AT_IMG_SIZE = 128
DATASET = "F:/StyleGANN/celebaHQ/celeba_hq/val"
CHECKPOINT_GEN = "/content/drive/MyDrive/BTP/generator.pth"
CHECKPOINT_CRITIC = "/content/drive/MyDrive/BTP/critic.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_MODEL = True
LOAD_MODEL = False
LEARNING_RATE = 1e-3
BATCH_SIZES = [16, 16, 16, 16, 16, 8, 8, 4, 4]
IMAGE_SIZE = 512
OUT_CHANNLES = 3
L_DIM = 512
IN_CHANNELS = 512
LAMDA_GP = 10
NUM_STEPS = int(log2(IMAGE_SIZE / 4)) + 1

PROGRESSIVE_EPOCHS = [2] * len(BATCH_SIZES)
FIXED_NOISE = torch.randn(8, L_DIM).to(DEVICE)
NUM_WORKERS = 4