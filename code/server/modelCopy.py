import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2, sqrt

factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]


def init_linear(linear):
    torch.nn.init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    torch.nn.init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=0, keepdim=True) + 1e-8)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv).to("cpu")

    def forward(self, input):
        return self.conv(input)


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear).to("cpu")

    def forward(self, input):
        return self.linear(input)


class AdaIN(nn.Module):
    def __init__(self, in_channels, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channels)
        self.style = Linear(style_dim, in_channels * 2)

        self.style.linear.bias.data[:in_channels] = 1
        self.style.linear.bias.data[in_channels:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class Noise(nn.Module):
    def __init__(self, n_channels):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, n_channels, 1, 1).to("cpu"))

    def forward(self, image):
        shape = image.shape
        noise = torch.randn(shape[0], 1, shape[2], shape[3], device="cpu")
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, n_channels, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, n_channels, size, size).to("cpu"))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


class GenBlock(nn.Module):
    def __init__(self, style_dim, in_channels, out_channels, kernel_size=3, padding=1, stride=1, initial=False):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channels)
        else:
            self.conv1 = Conv(in_channels, out_channels, kernel_size, padding, stride)
        self.noise1 = equal_lr(Noise(out_channels))
        self.adain1 = AdaIN(out_channels, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = Conv(out_channels, out_channels, kernel_size, padding, stride)
        self.noise2 = equal_lr(Noise(out_channels))
        self.adain2 = AdaIN(out_channels, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style):
        out = self.conv1(input)
        out = self.noise1(out)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out


class Generator(nn.Module):
    def __init__(self, style_dim, in_channels, out_channels=3):
        super().__init__()
        self.initial = GenBlock(style_dim, in_channels, in_channels, initial=True)
        self.progression_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList([Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)])
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        for i in range(len(factors) - 1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i+1])
            self.progression_blocks.append(GenBlock(style_dim, in_channels=conv_in_c, out_channels=conv_out_c))
            self.rgb_layers.append(Conv(conv_out_c, out_channels, kernel_size=1, stride=1, padding=0))

    def fade_in(self, alpha, upsampled, generated):
        return torch.tanh(alpha * generated + (1 - alpha) * upsampled)

    def forward(self, alpha, steps, style):
        out = self.initial(torch.randn((1, 1)), style)

        if steps == 0:
            return self.rgb_layers[steps](out)

        for step in range(steps):
            upsampled = self.upsample(out)
            out = self.progression_blocks[step](upsampled, style)

        final_upsampled = self.rgb_layers[steps-1](upsampled)
        final_out = self.rgb_layers[steps](out)

        return self.fade_in(alpha, final_upsampled, final_out)


class StyleGenerator(nn.Module):
    def __init__(self, initial_dim=512, n_mlp=8):
        super().__init__()

        self.generator = Generator(512, 512)

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(Linear(initial_dim, initial_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers).to("cpu")

    def forward(self, input, steps, alpha):
        styles = self.style(input)
        return self.generator(alpha, steps, styles)


class DiscConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels)
        self.conv2 = Conv(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.progression_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in_c = int(in_channels * factors[i])
            conv_out_c = int(in_channels * factors[i-1])
            self.progression_blocks.append(DiscConv(conv_in_c, conv_out_c))
            self.rgb_layers.append(Conv(out_channels, conv_in_c, kernel_size=1, stride=1, padding=0))

        self.initial_rgb = Conv(in_channels=out_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            Conv(in_channels+1, in_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            Conv(in_channels, in_channels, kernel_size=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            Conv(in_channels, 1, kernel_size=1, padding=0, stride=1)
        )


    def fade_in(self, alpha, downsampled, out):
        return alpha * out + (1 - alpha) * downsampled


    def minibatch_std(self, x):
        batch_stats = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_stats], dim=1)


    def forward(self, x, alpha, steps):
        cur_step = len(self.progression_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.progression_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step+1, len(self.progression_blocks)):
            out = self.progression_blocks[step](out)
            out = self.avg_pool(out)

        out =self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)


# if __name__ == "__main__":
#     style_dim = 512
#     in_channels = 512
#     gen = StyleGenerator()
#     critic = Discriminator(in_channels)
#
#     for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
#         num_steps = int(log2(img_size / 4))
#         x = torch.randn((1, 512))
#         z = gen(x, steps=num_steps, alpha=0.5)
#         assert z.shape == (1, 3, img_size, img_size)
#         out = critic(z, alpha=0.5, steps=num_steps)
#         assert out.shape == (1, 1)
#         print(f"Success at img size: {img_size}")