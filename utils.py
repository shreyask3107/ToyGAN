import os
import json
import matplotlib.pyplot as plt


def save_json(dict, path, name):
    with open(os.path.join(path, name), 'w') as f:
        f.write(json.dumps(dict, indent=4))


def plot_toy(tensor, num_bits, save_path=None, name='samples.png', pixels=1000, dpi=96):

    if num_bits is None:
        bounds = [[-1, 1], [-1, 1]]
        bins = 500
    else:
        bounds = [[0, 2**num_bits], [0, 2**num_bits]]
        bins = 2**num_bits

    tensor = tensor.detach().cpu().numpy()
    plt.figure(figsize=(pixels/dpi, pixels/dpi), dpi=dpi)
    plt.hist2d(tensor[...,0], tensor[...,1], bins=bins, range=bounds)
    plt.xlim(bounds[0])
    plt.ylim(bounds[1])
    plt.axis('off')
    if save_path is not None:
        plt.savefig(os.path.join(save_path, name), bbox_inches = 'tight', pad_inches = 0)
    plt.show()


class MetricMeter():

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = None
        self.count = 0

    def log(self, metrics):
        if self.metrics is None:
            self.metrics = {k: v.detach().cpu().item() for k, v in metrics.items()}
        else:
            for k in self.metrics.keys():
                self.metrics[k] += metrics[k].detach().cpu().item()
        self.count += 1

    def compute(self):
        return {k: v / self.count for k,v in self.metrics.items()}

    def __iter__(self):
        self.iterator = iter(self.compute().items())
        return self.iterator

    def __next__(self):
        return next(self.iterator)

    def __str__(self):
        loss_strs = ['{}: {:.3f}'.format(k, v) for k,v in self]
        return ', '.join(loss_strs)
