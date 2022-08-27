import numpy as np
import os
import torch

from collections.abc import Iterable
from skimage import color, io, transform
from sklearn.datasets import make_moons
from torch.utils.data import Dataset

import urllib.request
import zipfile
import json
from survae.data import TrainValidTestLoader, DATA_PATH

class Text8(TrainValidTestLoader):
    def __init__(self, root=DATA_PATH, seq_len=256, download=True):
        self.train = Text8Dataset(root, seq_len=seq_len, split='train', download=download)
        self.valid = Text8Dataset(root, seq_len=seq_len, split='valid')
        self.test = Text8Dataset(root, seq_len=seq_len, split='test')


class Text8Dataset(Dataset):
    """
    The text8 dataset consisting of 100M characters (with vocab size 27).
    We here split the dataset into (90M, 5M, 5M) characters for
    (train, val, test) as in [1,2,3].
    The sets are then split into chunks of equal length as specified by `seq_len`.
    The default is 256, corresponding to what was used in [1]. Other choices
    include 180, as [2] reports using.
    [1] Discrete Flows: Invertible Generative Models of Discrete Data
        Tran et al., 2019, https://arxiv.org/abs/1905.10347
    [2] Architectural Complexity Measures of Recurrent Neural Networks
        Zhang et al., 2016, https://arxiv.org/abs/1602.08210
    [3] Subword Language Modeling with Neural Networks
        Mikolov et al., 2013, http://www.fit.vutbr.cz/~imikolov/rnnlm/char.pdf
    """

    def __init__(self, root=DATA_PATH, seq_len=256, split='train', download=False):
        assert split in {'train', 'valid', 'test'}
        self.root = os.path.join(os.path.expanduser(root), 'text8')
        self.seq_len = seq_len
        self.split = split

        if not self._check_raw_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found. You can use download=True to download it.')

        if not self._check_processed_exists(split):
            self._preprocess_data(split)

        # Load data
        self.data = torch.load(self.processed_filename(split))

        # Load lookup tables
        char2idx_file = os.path.join(self.root, 'char2idx.json')
        idx2char_file = os.path.join(self.root, 'idx2char.json')
        with open(char2idx_file) as f:
            self.char2idx = json.load(f)
        with open(idx2char_file) as f:
            self.idx2char = json.load(f)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def s2t(self, s):
        assert len(s) == self.seq_len, 'String not of length {}'.format(self.seq_len)
        return torch.tensor([self.char2idx[char] for char in s])

    def t2s(self, t):
        return ''.join([self.idx2char[t[i]] if t[i] < len(self.idx2char) else ' ' for i in range(self.seq_len)])

    def text2tensor(self, text):
        if isinstance(text, str):
            tensor = self.s2t(text).unsqueeze(0)
        else:
            tensor = torch.stack([self.s2t(s) for s in text], dim=0)
        return tensor 

    def tensor2text(self, tensor):
        assert tensor.dim() == 2, 'Tensor should have shape (batch_size, {})'.format(self.seq_len)
        assert tensor.shape[1] == self.seq_len, 'Tensor should have shape (batch_size, {})'.format(self.seq_len)
        bsize = tensor.shape[0]
        text = [self.t2s(tensor[b]) for b in range(bsize)]
        return text

    def _preprocess_data(self, split):
        # Read raw data
        rawdata = zipfile.ZipFile(self.local_filename).read('text8').decode('utf-8')

        # Extract vocab
        vocab = sorted(list(set(rawdata)))
        char2idx, idx2char = {}, []
        for i, char in enumerate(vocab):
            char2idx[char] = i
            idx2char.append(char)

        # Extract subset
        if split == 'train':
            rawdata = rawdata[:90000000]
        elif split == 'valid':
            rawdata = rawdata[90000000:95000000]
        elif split == 'test':
            rawdata = rawdata[95000000:]

        # Encode characters
        data = torch.tensor([char2idx[char] for char in rawdata])

        # Split into chunks
        data = data[:self.seq_len*(len(data)//self.seq_len)]
        data = data.reshape(-1, self.seq_len)

        # Save processed data
        torch.save(data, self.processed_filename(split))

        # Save lookup tables
        char2idx_file = os.path.join(self.root, 'char2idx.json')
        idx2char_file = os.path.join(self.root, 'idx2char.json')
        with open(char2idx_file, 'w') as f:
            json.dump(char2idx, f)
        with open(idx2char_file, 'w') as f:
            json.dump(idx2char, f)

    @property
    def local_filename(self):
        return os.path.join(self.root, 'text8.zip')

    def processed_filename(self, split):
        return os.path.join(self.root, '{}.pt'.format(split))

    def download(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading text8...')

        url = 'http://mattmahoney.net/dc/text8.zip'
        print('Downloading from {}...'.format(url))
        urllib.request.urlretrieve(url, self.local_filename)
        print('Saved to {}'.format(self.local_filename))

    def _check_raw_exists(self):
        return os.path.exists(self.local_filename)

    def _check_processed_exists(self, split):
        return os.path.exists(self.processed_filename(split))



# Taken from https://github.com/bayesiains/nsf/blob/master/data/plane.py

class PlaneDataset(Dataset):
    def __init__(self, num_points, num_bits, flip_axes=False):
        self.num_points = num_points
        self.num_bits = num_bits
        self.flip_axes = flip_axes
        self.data = None
        self.reset()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    def reset(self):
        self._create_data()
        self.data = (self.data + 4.0) / 8.0 # [-4,4] -> [0,1]
        if self.num_bits is not None:
            self.data = self.data * (2**self.num_bits) # [0, 1] -> [0, 256]
            self.data = self.data.floor().clamp(min=0, max=2**self.num_bits-1).byte() # [0, 256] -> {0,1,...,255}
        else:
            self.data = self.data * 2 - 1 # [0, 1] -> [-1, 1]
        if self.flip_axes:
            x1 = self.data[:, 0]
            x2 = self.data[:, 1]
            self.data = torch.stack([x2, x1]).t()

    def _create_data(self):
        raise NotImplementedError


class GaussianDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2 = 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x1, x2)).t()


class CrescentDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = 0.5 * x1 ** 2 - 1
        x2_var = torch.exp(torch.Tensor([-2]))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x2, x1)).t()


class CrescentCubedDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = 0.2 * x1 ** 3
        x2_var = torch.ones(x1.shape)
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x2, x1)).t()


class SineWaveDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = torch.sin(5 * x1)
        x2_var = torch.exp(-2 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x1, x2)).t()


class AbsDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = torch.abs(x1) - 1.
        x2_var = torch.exp(-3 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x1, x2)).t()


class SignDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = torch.sign(x1) + x1
        x2_var = torch.exp(-3 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x1, x2)).t()


class FourCirclesDataset(PlaneDataset):
    def __init__(self, num_points, num_bits, flip_axes=False):
        if num_points % 4 != 0:
            raise ValueError('Number of data points must be a multiple of four')
        super().__init__(num_points, num_bits, flip_axes)

    @staticmethod
    def create_circle(num_per_circle, std=0.1):
        u = torch.rand(num_per_circle)
        x1 = torch.cos(2 * np.pi * u)
        x2 = torch.sin(2 * np.pi * u)
        data = 2 * torch.stack((x1, x2)).t()
        data += std * torch.randn(data.shape)
        return data

    def _create_data(self):
        num_per_circle = self.num_points // 4
        centers = [
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1]
        ]
        self.data = torch.cat(
            [self.create_circle(num_per_circle) - torch.Tensor(center)
             for center in centers]
        )


class DiamondDataset(PlaneDataset):
    def __init__(self, num_points, num_bits, flip_axes=False, width=20, bound=2.5, std=0.04):
        # original values: width=15, bound=2, std=0.05
        self.width = width
        self.bound = bound
        self.std = std
        super().__init__(num_points, num_bits, flip_axes)

    def _create_data(self, rotate=True):
        means = np.array([
            (x + 1e-3 * np.random.rand(), y + 1e-3 * np.random.rand())
            for x in np.linspace(-self.bound, self.bound, self.width)
            for y in np.linspace(-self.bound, self.bound, self.width)
        ])

        covariance_factor = self.std * np.eye(2)

        index = np.random.choice(range(self.width ** 2), size=self.num_points, replace=True)
        noise = np.random.randn(self.num_points, 2)
        self.data = means[index] + noise @ covariance_factor
        if rotate:
            rotation_matrix = np.array([
                [1 / np.sqrt(2), -1 / np.sqrt(2)],
                [1 / np.sqrt(2), 1 / np.sqrt(2)]
            ])
            self.data = self.data @ rotation_matrix
        self.data = self.data.astype(np.float32)
        self.data = torch.Tensor(self.data)


class TwoSpiralsDataset(PlaneDataset):
    def _create_data(self):
        n = torch.sqrt(torch.rand(self.num_points // 2)) * 540 * (2 * np.pi) / 360
        d1x = -torch.cos(n) * n + torch.rand(self.num_points // 2) * 0.5
        d1y = torch.sin(n) * n + torch.rand(self.num_points // 2) * 0.5
        x = torch.cat([torch.stack([d1x, d1y]).t(), torch.stack([-d1x, -d1y]).t()])
        self.data = x / 3 + torch.randn_like(x) * 0.1


class TestGridDataset(PlaneDataset):
    def __init__(self, num_points_per_axis, num_bits, bounds):
        self.num_points_per_axis = num_points_per_axis
        self.bounds = bounds
        self.shape = [num_points_per_axis] * 2
        self.X = None
        self.Y = None
        super().__init__(num_points=num_points_per_axis ** 2, num_bits=num_bits)

    def _create_data(self):
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], self.num_points_per_axis)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], self.num_points_per_axis)
        self.X, self.Y = np.meshgrid(x, y)
        data_ = np.vstack([self.X.flatten(), self.Y.flatten()]).T
        self.data = torch.tensor(data_).float()


class CheckerboardDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.rand(self.num_points) * 4 - 2
        x2_ = torch.rand(self.num_points) - torch.randint(0, 2, [self.num_points]).float() * 2
        x2 = x2_ + torch.floor(x1) % 2
        self.data = torch.stack([x1, x2]).t() * 2


class TwoMoonsDataset(PlaneDataset):
    '''Adapted from https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py'''
    def _create_data(self):
        data = make_moons(n_samples=self.num_points, noise=0.1, random_state=0)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        self.data = torch.from_numpy(data).float()
