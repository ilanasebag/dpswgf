import argparse


def create_args() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=int, metavar='DEVICE', default=None)
    p.add_argument('--config', type=str, metavar='PATH', required=True)
    p.add_argument('--data_path', type=str, metavar='PATH', required=True)
    p.add_argument('--exp_path', type=str, metavar='PATH', required=True)
    p.add_argument('--num_workers', type=int, metavar='WORKERS', default=4)
    p.add_argument('--load_only', action='store_true')
    return p
