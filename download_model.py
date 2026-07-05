"""Downloads the pretrained final model into runs/final_model/fold0/best.pt.

The weights are not versioned in the repository (too large); they are published as a
GitHub Release asset. Just run:

    python download_model.py

(or pass a different URL as the first argument). Then solve any maze without training:

    python predict.py --input maze.npz --ckpt_dir runs/final_model --k 1

The file is a float16 checkpoint of the final single model (5-level U-Net,
goal-conditioning). It is cast back to float32 automatically when loaded.
"""

import os
import sys
import argparse
import urllib.request

DEFAULT_URL = ('https://github.com/CarlosPoLopez/Navigation_CNN/releases/'
               'download/v1.0/final_model_fp16.pt')
DEFAULT_OUT = os.path.join('runs', 'final_model', 'fold0', 'best.pt')


def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(100, 100 * done / total_size)
        sys.stdout.write(f'\r  downloading... {pct:5.1f}%  ({done/1048576:.1f} MB)')
    else:
        sys.stdout.write(f'\r  downloading... {done/1048576:.1f} MB')
    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser(description='Download the pretrained final model.')
    p.add_argument('url', nargs='?', default=DEFAULT_URL,
                   help='URL of the best.pt asset (default: the v1.0 GitHub Release).')
    p.add_argument('--out', default=DEFAULT_OUT,
                   help=f'destination path (default: {DEFAULT_OUT}).')
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f'Fetching {args.url}')
    urllib.request.urlretrieve(args.url, args.out, _progress)
    size = os.path.getsize(args.out) / 1048576
    print(f'\n  Saved -> {args.out}  ({size:.1f} MB)')
    print('  Now run:  python predict.py --input maze.npz --ckpt_dir runs/final_model --k 1')


if __name__ == '__main__':
    main()

