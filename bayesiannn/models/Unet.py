import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
from pyro.nn import PyroModule, PyroSample

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class ConvBlock(PyroModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bayesian: bool = False,
        distributions: dict = None,
    ):
        super().__init__()
        self.block = PyroModule[nn.Sequential](
            PyroModule[nn.Conv2d](
                in_features, out_features, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            PyroModule[nn.Conv2d](
                out_features, out_features, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
        )

        if bayesian:
            self.add_bayesian_weights(distributions)

    def forward(self, x):
        return self.block(x)

    def add_bayesian_weights(self, distributions: dict):
        """
        Заменяет веса модели вероятностыми распределениями.
        distributions: заданные пользователем распределения
        (иначе используются нормальные)
        """
        if "conv1" not in distributions:
            distributions["conv1"] = dist.Normal(
                torch.tensor(0.0, device=DEVICE),
                torch.tensor(1.0, device=DEVICE),
            )
        if "conv2" not in distributions:
            distributions["conv2"] = dist.Normal(
                torch.tensor(0.0, device=DEVICE),
                torch.tensor(1.0, device=DEVICE),
            )

        conv1_dim = self.block[0].weight.size()
        conv2_dim = self.block[3].weight.size()
        self.block[0].weight = PyroSample(
            distributions["conv1"].expand(conv1_dim).to_event(4)
        )
        self.block[3].weight = PyroSample(
            distributions["conv2"].expand(conv2_dim).to_event(4)
        )


class UpConvBlock(PyroModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bayesian: bool = False,
        distributions: dict = None,
    ):
        super().__init__()
        if distributions is None:
            distributions = {}

        self.upscale = PyroModule[nn.ConvTranspose2d](
            in_features, out_features, 2, stride=2
        )
        self.block = ConvBlock(
            out_features * 2, out_features, bayesian, distributions
        )

        if bayesian:
            self.add_bayesian_weights(distributions)

    def forward(self, x, encoder_map):
        x = self.upscale(x)
        x = torch.cat([x, encoder_map], dim=1)
        return self.block(x)

    def add_bayesian_weights(self, distributions: dict):
        """
        Заменяет веса модели вероятностыми распределениями.
        distributions: заданные пользователем распределения
        (иначе используются нормальные)
        """
        if "upconv" not in distributions:
            distributions["upconv"] = dist.Normal(
                torch.tensor(0.0, device=DEVICE),
                torch.tensor(1.0, device=DEVICE),
            )

        upscale_dim = self.upscale.weight.size()
        self.upscale.weight = PyroSample(
            distributions["conv1"].expand(upscale_dim).to_event(4)
        )


class Unet(PyroModule):
    def __init__(self, bayesian: bool = False, distributions: dict = None):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)
        if distributions is None:
            distributions = {}

        for key in [f"block{i}" for i in range(1, 5 + 1)]:
            if key not in distributions:
                distributions[key] = {}
        for key in [f"up_block{i}" for i in range(1, 4 + 1)]:
            if key not in distributions:
                distributions[key] = {}

        self.block1 = PyroModule[ConvBlock](
            3, 32, bayesian, distributions["block1"]
        )
        self.block2 = PyroModule[ConvBlock](
            32, 64, bayesian, distributions["block2"]
        )
        self.block3 = PyroModule[ConvBlock](
            64, 128, bayesian, distributions["block3"]
        )
        self.block4 = PyroModule[ConvBlock](
            128, 256, bayesian, distributions["block4"]
        )
        self.block5 = PyroModule[ConvBlock](
            256, 512, bayesian, distributions["block5"]
        )

        self.up_block1 = PyroModule[UpConvBlock](
            512, 256, bayesian, distributions["up_block1"]
        )
        self.up_block2 = PyroModule[UpConvBlock](
            256, 128, bayesian, distributions["up_block2"]
        )
        self.up_block3 = PyroModule[UpConvBlock](
            128, 64, bayesian, distributions["up_block3"]
        )
        self.up_block4 = PyroModule[UpConvBlock](
            64, 32, bayesian, distributions["up_block4"]
        )

        if bayesian:
            self.segment_conv = PyroModule[nn.Sequential](
                PyroModule[nn.Conv2d](32, 1, 3, stride=1, padding=1),
                nn.Sigmoid(),
            )
            self.add_bayesian_weights(distributions)

        else:
            self.segment_conv = nn.Conv2d(32, 1, 3, stride=1, padding=1)

        self.bayesian = bayesian

    def forward(self, x, y=None):
        xe1 = self.block1(x)
        xe1m = self.max_pool(xe1)

        xe2 = self.block2(xe1m)
        xe2m = self.max_pool(xe2)

        xe3 = self.block3(xe2m)
        xe3m = self.max_pool(xe3)

        xe4 = self.block4(xe3m)
        xe4m = self.max_pool(xe4)

        xe5 = self.block5(xe4m)
        xd1 = self.up_block1(xe5, xe4)
        xd2 = self.up_block2(xd1, xe3)
        xd3 = self.up_block3(xd2, xe2)
        xd4 = self.up_block4(xd3, xe1)
        xd5 = self.segment_conv(xd4)

        # xd5 - вероятность для каждого пикселя в распределении Бернули
        # поэтому каждый пиксель просэмплируем, чтобы получить
        # оценку неопределенности
        # .plate - указываем, что у нас батч независимых наблюдений

        # пиксели - зависимые случайные величины, поэтому укажем
        # это в .to_event(3) - последние 3 координаты из одного распределения
        if self.bayesian:
            with pyro.plate("data", x.size(0)):
                obs = pyro.sample(
                    "obs", dist.Bernoulli(probs=xd5).to_event(3), obs=y
                )
                return obs
        else:
            return xd5

    def add_bayesian_weights(self, distributions: dict):
        """
        Заменяет веса модели вероятностыми распределениями.
        distributions: заданные пользователем распределения
        (иначе используются нормальные)
        """
        if "segment_conv" not in distributions:
            distributions["segment_conv"] = dist.Normal(
                torch.tensor(0.0, device=DEVICE),
                torch.tensor(1.0, device=DEVICE),
            )

        segment_conv_dim = self.segment_conv[0].weight.size()

        self.segment_conv[0].weight = PyroSample(
            (
                distributions["segment_conv"]
                .expand(segment_conv_dim)
                .to_event(4)
            )
        )
