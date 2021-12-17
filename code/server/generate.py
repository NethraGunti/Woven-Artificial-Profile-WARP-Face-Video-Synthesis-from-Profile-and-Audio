import torch
from modelCopy import StyleGenerator
from utils import load_checkpoint
from scipy.stats import truncnorm
from torchvision.utils import save_image



def truncate_noise(n_samples, z_dim, truncation):
    truncated_noise = truncnorm.rvs(-1 * truncation,
                                    truncation, size=(n_samples, z_dim))
    return torch.Tensor(truncated_noise)


def sample_image(generator: StyleGenerator, noise: torch.Tensor):
    with torch.no_grad():
        image = generator(noise, 5, 0.8).detach().squeeze().cpu()
    assert image.ndim == 3
    return image


if __name__ == "__main__":
    generator = StyleGenerator()
    load_checkpoint("generator.pth", generator)
    generator.eval()
    with torch.no_grad():
        noise = truncate_noise(1, 512, 0.5)
        # noise = torch.zeros([1, 512])
        images = generator(noise, 5, 0.4).detach().squeeze().cpu()
        save_image(images, "sample.png")