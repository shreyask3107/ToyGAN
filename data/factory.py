from torch.utils.data import DataLoader
from .datasets import GaussianDataset, CrescentDataset, CrescentCubedDataset, \
                      SineWaveDataset, AbsDataset, SignDataset, \
                      FourCirclesDataset, DiamondDataset, TwoSpiralsDataset, \
                      CheckerboardDataset, TwoMoonsDataset, Text8Dataset
from survae.data import TrainValidTestLoader, DATA_PATH


def get_data(args):
    if args.dataset == 'gaussian':
        train = GaussianDataset(num_points=args.train_samples, num_bits=args.num_bits)
        test = GaussianDataset(num_points=args.test_samples, num_bits=args.num_bits)
    elif args.dataset == 'crescent':
        train = CrescentDataset(num_points=args.train_samples, num_bits=args.num_bits)
        test = CrescentDataset(num_points=args.test_samples, num_bits=args.num_bits)
    elif args.dataset == 'crescent_cubed':
        train = CrescentCubedDataset(num_points=args.train_samples, num_bits=args.num_bits)
        test = CrescentCubedDataset(num_points=args.test_samples, num_bits=args.num_bits)
    elif args.dataset == 'sinewave':
        train = SineWaveDataset(num_points=args.train_samples, num_bits=args.num_bits)
        test = SineWaveDataset(num_points=args.test_samples, num_bits=args.num_bits)
    elif args.dataset == 'abs':
        train = AbsDataset(num_points=args.train_samples, num_bits=args.num_bits)
        test = AbsDataset(num_points=args.test_samples, num_bits=args.num_bits)
    elif args.dataset == 'sign':
        train = SignDataset(num_points=args.train_samples, num_bits=args.num_bits)
        test = SignDataset(num_points=args.test_samples, num_bits=args.num_bits)
    elif args.dataset == 'diamond':
        train = DiamondDataset(num_points=args.train_samples, num_bits=args.num_bits)
        test = DiamondDataset(num_points=args.test_samples, num_bits=args.num_bits)
    elif args.dataset == 'four_circles':
        train = FourCirclesDataset(num_points=args.train_samples, num_bits=args.num_bits)
        test = FourCirclesDataset(num_points=args.test_samples, num_bits=args.num_bits)
    elif args.dataset == 'two_spirals':
        train = TwoSpiralsDataset(num_points=args.train_samples, num_bits=args.num_bits)
        test = TwoSpiralsDataset(num_points=args.test_samples, num_bits=args.num_bits)
    elif args.dataset == 'checkerboard':
        train = CheckerboardDataset(num_points=args.train_samples, num_bits=args.num_bits)
        test = CheckerboardDataset(num_points=args.test_samples, num_bits=args.num_bits)
    elif args.dataset == 'two_moons':
        train = TwoMoonsDataset(num_points=args.train_samples, num_bits=args.num_bits)
        test = TwoMoonsDataset(num_points=args.test_samples, num_bits=args.num_bits)
    elif args.dataset == 'text8':
        train = Text8Dataset(DATA_PATH, seq_len=256, split='train', download=True)
        test = Text8Dataset(DATA_PATH, seq_len=256, split='test', download=True)    
    else:
        raise ValueError(f'dataset {args.dataset} unknown')

    # Data Loader
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=args.test_batch_size, shuffle=False)

    return train_loader, test_loader
