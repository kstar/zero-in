#!/usr/bin/env python3
import sys
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('solve-cli')
import argparse
import subprocess
import timing
import json
import glob

import numpy as np
from PIL import Image
import sep
from astropy.io import fits

try:
    from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                     denoise_wavelet, estimate_sigma)
    skimage_found = True
except (ImportError, ModuleNotFoundError):
    logger.error(f'skimage (scikit-image) not found. The --denoise option will not work.')
    skimage_found = False
    

from platesolve import PlateSolution, extract_sources_sep, solve_field_xylist
import platesolve


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(g):
        return g

def main(argv):

    parser = argparse.ArgumentParser(description='Command-line interface to SEP and astrometry.net', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    group = parser.add_argument_group(title='Preprocessing', description='Preprocessing of image before SEP')
    group.add_argument(
        '--debayer', '-D',
        choices=['RGGB', 'GBRG', 'BGGR', 'GRBG'],
        default=None,
        required=False,
        help='Debayer the image before processing'
    )
    group.add_argument(
        '--denoise', '-d',
        choices=['tv', 'bilateral', 'wavelet', 'wavelet_YCbCr'],
        default=None,
        required=False,
        help='Denoise the image using one of the sklearn algorithms.'
    )
    group.add_argument(
        '--crop', nargs=2, action='append', metavar=('side', 'fraction'),
        help='Crop fraction length of the image from the specified side (top|bottom|left|right)',
    )
    group.add_argument(
        '--binning', '-b',
        type=int, default=1, required=False, help='Binning'
    )

    group = parser.add_argument_group(title='Extraction', description='Source Extraction parameters for SEP')
    group.add_argument(
        '--sep-threshold', '-t',
        type=float, required=True, help='Threshold for SEP'
    )
    group.add_argument(
        '--bg-block', '-k',
        type=int, default=None, required=False, help='Background extraction block',
    )
    group.add_argument(
        '--bg-filter', '-f',
        type=int, default=None, required=False, help='Size of the convolution filter for background extraction',
    )
    group = parser.add_argument_group(title='Filtering', description='Short-listing candidates')
    group.add_argument(
        '--max-aspect-ratio', '-ta',
        type=float, default=4.0, required=False, help='Reject detections with aspect ratios more skewed than this',
    )
    group.add_argument(
        '--noise-patch-threshold', '-tM',
        type=float, default=0.2, required=False, help='Reject detections which have semi-major axis larger than this fraction of image width'
    )
    group.add_argument(
        '--hot-pixel-threshold', '-tm',
        type=float, default=1, required=False, help='Reject detections which have semi-minor axis smaller than this number of pixels'
    )

    group = parser.add_argument_group(title='Extraction Output', description='Choose where to write extraction outputs')
    group.add_argument(
        '--xylist', '-o',
        type=argparse.FileType('a'), required=False, default=sys.stdout, help='Write xylist to this file. If omitted, we write to stdout'
    )
    group.add_argument(
        '--source-plot', '-p',
        type=str, required=False, default=None, help='Write the plot of sources in PNG format to this file'
    )
    group.add_argument(
        '--background', '-B',
        type=str, required=False, default=None, help='Write the extracted background to this file'
    )
    group.add_argument(
        '--input-image', '-i',
        type=str, required=False, default=None, help='Write the input image to this file'
    )
    group.add_argument(
        '--binned-image', '-I',
        type=str, required=False, default=None, help='Write the binned and channel-averaged image to this file'
    )
    group.add_argument(
        '--image-stretching', '-S',
        choices=['asinh', 'log', 'linear'],
        required=False, default='asinh', help='Stretching to apply to input and binned image before writing them'
    )
    group.add_argument(
        '--display', action='store_true',
        help='Invoke the display command to show the image results requested'
    )
    group.add_argument(
        '--display-timeout', type=int,
        help='Timeout for display windows in seconds. (Supply 0 for no timeout)',
        default=60,
        required=False
    )

    group = parser.add_argument_group(title='Solve', description='Solve after extraction')
    group.add_argument(
        '--solve', action='store_true',
        help='Invoke astrometry.net to solve the field after extraction'
    )
    group.add_argument(
        '--solve-timeout', type=int,
        help='Timeout solve-field after this much wall-clock time. `0` means no timeout',
        default=60,
        required=False,
    )
    group.add_argument(
        '--top-k', type=int,
        help='Preserve only the top-k entries in the xylist for solving. `0` means use all entries.',
        default=0,
        required=False,
    )
    group.add_argument(
        '--scale-units',
        choices=['arcsecperpix', 'degwidth', 'dw', 'degw', 'arcminwidth', 'amw', 'aw', 'focalmm',],
        required=False,
        default='arcsecperpix',
        help='Unit of optional scale hint supplied'
    )
    group.add_argument(
        '--scale-low',
        type=float, required=False, default=None,
        help='Scale estimate, lower bound'
    )
    group.add_argument(
        '--scale-high',
        type=float, required=False, default=None,
        help='Scale estimate, upper bound'
    )
    group.add_argument(
        '--ra',
        type=float, required=False, default=None,
        help='RA Estimate in degrees'
    )
    group.add_argument(
        '--dec',
        type=float, required=False, default=None,
        help='Dec Estimate in degrees'
    )
    group.add_argument(
        '--radius',
        type=float, required=False, default=None,
        help='Search radius around (--ra, --dec) in degrees'
    )
    group.add_argument(
        '--parity', choices=['positive', 'negative', 'any'],
        default='any', required=False,
        help='Image parity hint'
    )
    group.add_argument(
        '--uniformize', default=0, type=int, required=False,
        help='Same as --uniformize argument to `solve-field`, but the default is 0 instead of 10 (solve-field default)'
    )
    group.add_argument(
        '--remove-line-sources', action='store_true',
        help='By default, we supply --no-remove-lines to `solve-field` to skip the Pythonic step of removing line sources (filters out satellite tracks etc). This reverts the behavior to not supply --no-remove-lines'
    )
    group.add_argument(
        '--solve-extra-args', '-a', type=str,
        help='Extra arguments to solve-field. Eg: -a \'--odds-to-reject 1e8\'',
        default=None,
        required=False,
    )
    group = parser.add_argument_group(title='Timing Output', description='Timing outputs')
    group.add_argument(
        '--json', '-j',
        default=None, required=False,
        help='Write a JSON with the timing and other information to this file'
    )


    group = parser.add_argument_group(title='Input', description='Input options')
    parser.add_argument('input', type=str, help='Input image in FITS format, or directory of images to run on, or a list of FITS file paths (see --list-input). Use `-` for stdin.')
    parser.add_argument('--glob', type=str, help='Glob to use if the input is a directory.', default='**/*.fits', required=False)
    parser.add_argument('--list-input', action='store_true', help='Input file contains a list of image paths to run on')

    args = parser.parse_args(sys.argv[1:])
    if args.denoise is not None and (not skimage_found):
        raise ImportError(f'scikit-image is not installed. Cannot use `--denoise`')


    file_list = []
    if args.list_input:
        if args.input == '-':
            f = sys.stdin
        elif os.path.isfile(args.stdin):
            f = open(args.stdin, 'r')
        elif os.path.isdir(args.stdin):
            raise ValueError(f'Cannot use directory {args.input} as input with `--list-input`')
        else:
            raise ValueError(f'Invalid input path {args.input}')

        file_list = [q.rstrip('\n').strip(' ') for q in f.readlines()]
        if args.input != '-':
            f.close()

    else:
        if args.input == '-' or os.path.isfile(args.input):
            file_list = [args.input]
        elif os.path.isdir(args.input):
            file_list = list(glob.glob(os.path.join(args.input, args.glob), recursive=True))
            if len(file_list) == 0:
                raise ValueError(f'Empty glob {args.glob} in directory {args.input}')
        else:
            raise ValueError(f'Invalid input path {args.input}')


    assert len(file_list) > 0

    if len(file_list) > 1:
        if args.source_plot or args.background or args.input_image or args.binned_image or args.display:
            raise ValueError(f'Cannot use image outputs with multiple input files')

    def write_image(data: np.ndarray, path: str):
        if data.dtype not in (np.float32, np.float64): # NOTE: Does not handle signed integer images
            data = data.astype(np.float32) / float(np.iinfo(data.dtype).max()) # pylint: disable=E1102
        data = (data - data.min())/(data.max() - data.min())
        Image.fromarray((255 * data).astype(np.uint8)).save(path)

    def stretch(data: np.ndarray):
        if args.image_stretching == 'log':
            data = np.log(1.0 + data)/0.693147
        elif args.image_stretching == 'asinh':
            data = np.arcsinh(1.17520 * data)
        elif args.image_stretching == 'linear':
            pass
        else:
            raise NotImplementedError(f'Unimplemented scaling algorithm {args.image_stretching}')
        return data


    stats = {}
    for filepath in tqdm(file_list):
        try:
            logger.info(f'Processing file {filepath}')
            Timer = timing.makeOrGetTimingClass('solve_cli')

            input_is_fits = None
            with Timer('read_image'):
                if filepath == '-':
                    f = fits.open(sys.stdin)
                    image = f[0].data[::-1, ...] # pylint: disable=E1101
                    input_is_fits = True
                else:
                    if filepath.lower().endswith('.fit') or filepath.lower().endswith('.fits'):
                        f = fits.open(filepath)
                        image = f[0].data[::-1, ...] # pylint: disable=E1101
                        input_is_fits = True
                    else:
                        image = np.asarray(Image.open(filepath))
                        input_is_fits = False

            debayer_order = {
                # From https://indilib.org/forum/ccds-dslrs/5761-saved-fits-not-debayred.html
                # (ry, rx), (gy, gx), (Gy, Gx), (by, bx)
                'RGGB': [(0,0), (1,0), (0,1), (1,1)],
                'GBRG': [(1,0), (0,0), (1,1), (0,1)],
                'BGGR': [(1,1), (0,1), (1,0), (0,0)],
                'GRBG': [(0,1), (1,1), (0,0), (1,0)],
            }

            if args.debayer:
                if not input_is_fits:
                    logger.warning('Debayering a non-FITS image {filepath}?')
                with Timer('debayer'):
                    mask = debayer_order[args.debayer]
                    h, w = image.shape
                    assert h % 2 == 0, h
                    assert w % 2 == 0, w
                    i = image.reshape(h//2, 2, w//2, 2).transpose([0, 2, 1, 3])
                    R = i[..., mask[0][0], mask[0][1]]
                    G = (
                        i[..., mask[1][0], mask[1][1]] + i[..., mask[2][0], mask[2][1]]
                    )/2.0
                    B = i[..., mask[3][0], mask[3][1]]
                    image = np.stack([R, G, B], axis=-1) # debayered image

            if args.crop:
                with Timer('crop'):
                    for side, fraction in args.crop:
                        H, W, _ = image.shape
                        if side not in {'top', 'bottom', 'left', 'right'}:
                            raise ValueError(f'Invalid side for --crop: {side}')
                        try:
                            fraction = float(fraction)
                        except ValueError:
                            raise ValueError(f'Invalid fraction for --crop: {fraction}')

                        if side == 'top':
                            image = image[int(fraction * H):, ...]
                        elif side == 'bottom':
                            image = image[:(H - int(fraction * H)), ...]
                        elif side == 'left':
                            image = image[:, int(fraction * W):, ...]
                        elif side == 'right':
                            image = image[:, :(W - int(fraction * W)), ...]
                        else:
                            raise NotImplementedError(side)

            if args.denoise:
                with Timer('denoise'):
                    if args.denoise == 'tv':
                        image = denoise_tv_chambolle(image, weight=0.2, multichannel=True)
                    elif args.denoise == 'wavelet':
                        image = denoise_wavelet(image, rescale_sigma=True, multichannel=True)
                    elif args.denoise == 'wavelet_YCbCr':
                        image = denoise_wavelet(image, rescale_sigma=True, convert2ycbcr=True, multichannel=True)
                    elif args.denoise == 'bilateral':
                        image = denoise_bilateral(image, sigma_color=0.1, sigma_spatial=15, multichannel=True)
                    else:
                        raise NotImplementedError(f'Unhandled denoising choice: {args.denoise}')

            with Timer('extract_sources'):
                (xylist, h, w), (binned_image, background, full_xylist) = extract_sources_sep(
                    image, args.sep_threshold, binning=args.binning,
                    bg_block=args.bg_block, bg_filter=args.bg_filter,
                    max_aspect_ratio=args.max_aspect_ratio,
                    noise_patch_threshold=args.noise_patch_threshold,
                    hot_pixel_threshold=args.hot_pixel_threshold,
                    plot_file=args.source_plot, display_plot=False,
                    output_intermediates=True,
                )

            print('# x, y, flux', file=args.xylist) # CSV header
            for x, y, flux in xylist:
                print(f'{x}, {y}, {flux}', file=args.xylist)

            image_outputs = []
            if args.source_plot:
                image_outputs.append(args.source_plot)

            if args.input_image:
                write_image(stretch(image), args.input_image)
                image_outputs.append(args.input_image)

            if args.background:
                write_image(background.back(np.float32), args.background)
                image_outputs.append(args.background)

            if args.binned_image:
                write_image(stretch(binned_image), args.binned_image)
                image_outputs.append(args.binned_image)

            if args.display:
                for out in image_outputs:
                    plot_process = subprocess.Popen(['timeout', str(args.display_timeout), 'display', out])

            if args.solve:
                logger.info('Invoking solve-field!')
                with Timer('solve'):
                    try:
                        solve_field_xylist(
                            xylist, h, w,
                            top_k=args.top_k,
                            scale_units=args.scale_units,
                            scale_high=args.scale_high,
                            scale_low=args.scale_low,
                            ra=args.ra,
                            dec=args.dec,
                            radius=args.radius,
                            parity=args.parity,
                            timeout=args.solve_timeout,
                            extra_args=(args.solve_extra_args.split(' ') if args.solve_extra_args else None),
                            uniformize=args.uniformize,
                            no_remove_lines=(not args.remove_line_sources),
                        )
                        solve_status = 'success'
                    except RuntimeError as e:
                        logger.error(f'Failed to plate-solve. The exception was:\n{e}')
                        solve_status = 'failure'
            else:
                solve_status = 'not_requested'

            logger.info('Overall timing table: =========')
            total = 0
            for key, value in Timer.get().items():
                print(f'{key.ljust(30, " ")}: {value:0.04f}', file=sys.stderr)
                total += value
            print('-' * 40, file=sys.stderr)
            print('Total'.ljust(30, " ") + f': {total:0.04f}', file=sys.stderr)
            print('\n', file=sys.stderr)

            logger.info('Fine-grained timing table: ====')
            for key, value in platesolve.timing.items():
                print(f'{key.ljust(30, " ")}: {value:0.04f}', file=sys.stderr)

            stats.setdefault('data', []).append({
                'filename': filepath,
                'timing': {
                    'general': dict(Timer.get()),
                    'fine-grained': dict(platesolve.timing),
                    'total': total,
                },
                'solve': {
                    'solve_status': solve_status,
                },
                'result': True,
                'exception': None,
            })
        except Exception as e:
            logger.error(f'Failed to process file {filepath} due to exception {e}')
            stats.setdefault('data', []).append({
                'filename': filepath,
                'result': False,
                'exception': str(e),
            })

    stats.update({
        'args': {
            k: v
            for k, v in args.__dict__.items() if type(v) in (type(None), int, str, float, bool, list, tuple)
        },
        'meta': {
            'info': 'solve-cli json output',
            'version': '0.0.1',
        },
    })
    if args.json:
        if not os.path.isdir(os.path.dirname(args.json)):
            os.makedirs(os.path.dirname(args.json))
        with open(args.json, 'w') as f:
                json.dump(stats, f)


if __name__ == "__main__":
    main(sys.argv)
