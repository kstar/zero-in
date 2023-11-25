# Wrapper around solve-field
import math
import collections
from io import BytesIO
from astropy.wcs import WCS
from astropy.io import fits
import time
import os
import subprocess
import multiprocessing
import sys
import sep
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from dms import *
from PIL import Image, ImageDraw
import logging

from coordinates import ICRS

logger = logging.getLogger('PlateSolve')

global timing
timing = {}

ImageWithGnomonicSolution = collections.namedtuple(
    "ImageWithGnomonicSolution", [
        "image", # image, type varies
        "center", # ICRS object
        "scale", # scale in arcsecperpix
        "north_angle", # North angle in degrees
     ]
)

class PlateSolution:
    """
    A class to hold an image with a WCS-based solution
    """

    def __init__(self, img, wcs_fits=None):
        """
        img: Image in any of the following formats:
           * (str) A path to a FITS file
           * (np.ndarray) FITS image data
           * (bytes) FITS file contents as bytes

        wcs_fits: A FITS file containing a WCS header, or a WCS header
        object. None is allowed only if img is not of type np.ndarray,
        whereby the WCS header is taken to be the first FITS HDU of the
        image file
        """

        if isinstance(img, np.ndarray) and wcs_fits is None:
            raise ValueError(
                'Must supply WCS header separately if the image is supplied as a numpy array!'
            )

        # N.B. It appears that astropy reads the FITS data with the Y-axis pointing the wrong way
        if type(img) is str:
            # Path to a FITS file
            self._fits = fits.open(img)
            self._img_data = self._fits[0].data[::-1, ...]
        elif isinstance(img, np.ndarray):
            # Array
            self._fits = None
            self._img_data = np.ascontiguousarray(img[::-1, ...]).copy()
        elif type(img) is bytes:
            # FITS data in bytes
            self._fits = fits.open(BytesIO(img))
            self._img_data = self._fits[0].data[::-1, ...]
        else:
            raise TypeError(
                f'Unsupported image type {type(img)}'
            )

        if wcs_fits:
            self._w = WCS(wcs_fits)
        else:
            # Attempt to get WCS info from the image file itself
            self._w = WCS(self._fits[0])

        self._img_size = self._img_data.shape[:2]

        self._gnomonic_solution = None

        # FIXME: Needs refactoring: Transition to performing alignment
        # point annotation in QGraphicsView instead of using PIL
        if self._img_data.dtype in (np.float32, np.float64):
            logger.warning(f'PlateSolution: Converting image of dtype {self._img_data.dtype.str} into uint8 for PIL')
            self._img_data = (np.clip(self._img_data, 0, 1) * 255).astype(np.uint8)
        self._orig_img = Image.fromarray(self._img_data)
        self._plot_img = None
        self._draw = None
        self._qimage = None

    def to_pixels(self, ra, dec, relative_to='top_left'):
        """
        Given (ra, dec),
        Returns: (x, y)
        """
        if type(ra) is str:
            ra = convert_ra(ra)
        if type(dec) is str:
            dec = convert_dec(dec)

        h, w = self.image_size
        x, y = self._w.all_world2pix(ra, dec, 0)
        y = h - y # Needed for the Y-flip. See __init__

        if relative_to == 'top_left':
            return float(x), float(y)
        elif relative_to == 'center':
            return (float(x) - w/2.0, h/2.0 - float(y))
        elif relative_to == 'center_ydown':
            return (float(x) - w/2.0, float(y) - h/2.0)
        else:
            raise NotImplementedError(f'Unhandled coordinate system: relative_to={relative_to}')

    def compute_scale(self):
        """
        Hackish solution to compute the scale (arcsec / pixel)
        """
        cx, cy = self._img_size[1]/2.0, self._img_size[0]/2.0
        ra1_, dec1_ = self.to_radec(cx, cy)
        ra2_, dec2_ = self.to_radec(cx + 10, cy) # 10 px shift
        ra1 = math.radians(ra1_)
        ra2 = math.radians(ra2_)
        dec1 = math.radians(dec1_)
        dec2 = math.radians(dec2_)

        logger.warning('FIXME: Arccosine is not accurate for small angles; use Haversine')
        theta = math.degrees(
            math.acos(
                math.sin(dec1) * math.sin(dec2) +
                math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
            )
        ) * 3600.0 # arcsec
        logger.debug('Plate scale in arcsecperpix {}'.format(theta/10.0))

        return (theta / 10)

    def compute_north_angle(self):
        """
        Hackish solution to compute the north angle on the plate

        Returns: North angle from the upward axis if the plate is not
                 centered at a pole, NaN otherwise
        """
        ra0, dec0 = self.to_radec(0, 0, relative_to='center')

        if abs(dec0) >= 89.95:
            # Almost centered on the pole, return NaN
            return math.nan

        # Estimate north angle by moving a small amount along
        # declination axis from the plate center
        ra1, dec1 = (ra0, dec0 + 2.99/60.0)
        dx, dy = self.to_pixels(ra1, dec1, relative_to='center')
        angle = math.degrees(math.atan2(dy, dx)) - 90

        if self._fits:
            crota1 = self._fits[0].header['CROTA1']
            crota2 = self._fits[0].header['CROTA2']
            logger.debug(f'Computed north angle {angle}° (CCW), CROTA1 = {crota1}, CROTA2 = {crota2}')
        else:
            logger.debug(f'Computed north angle {angle}° (CCW)')

        return angle

    def as_qimage(self, dst_dtype=np.uint8, subsamp=1, tonemapping='log', **params):
        """
        dst_dtype: None for auto-detection, or can be np.uint8 or np.uint16
        subsamp: Subsample the image by this factor (NotImplementedError)
        tonemapping: Choose a way to rescale the output
        """

        # FIXME: May want to move the implementation to C++

        if not self._qimage: # FIXME: Cache is not conditioned on parameters
            raw_pixels = self._img_data
            u, l = float(raw_pixels.max()), float(raw_pixels.min())
            h, w = raw_pixels.shape[:2]
            c = raw_pixels.shape[2] if raw_pixels.ndim == 3 else 1
            if raw_pixels.dtype in (np.int8, np.uint8, '>i1', '<i1', '>u1', '<u1'):
                dst_dtype = np.uint8
            elif raw_pixels.dtype in (np.int16, np.uint16, '>i2', '<i2', '>u2', '<u2'):
                dst_dtype = np.uint16
            else:
                from IPython import embed
                embed(header=f'Unhandled data format {raw_pixels.dtype}')
                raise NotImplementedError(f'Unhandled data format {raw_pixels.dtype}')

            assert np.issubdtype(dst_dtype, np.integer)
            scale = np.iinfo(dst_dtype).max
            bits = np.iinfo(dst_dtype).bits

            pixels = (raw_pixels.astype(np.float32) - l)/(u - l)
            if tonemapping == 'log':
                pixels = np.log(1.0 + pixels)/.693147 # magic number is ln(2)
            elif tonemapping == 'linear':
                pass
            elif tonemapping == 'asinh':
                beta = params.get('beta', 1.17520) # magic number is sinh(1)
                pixels = np.arcsinh(beta * pixels)/np.arcsinh(beta)
            elif tonemapping == 'histogram':
                raise NotImplementedError('Histogram Equalization Not Implemented')
            else:
                raise NotImplementedError('Invalid tonemapping algorithm!')
            pixels = (scale * pixels).astype(dst_dtype)

            # Determine QImage format
            from PyQt5.QtGui import QImage
            fmt = None
            if pixels.ndim == 2:
                if bits == 16:
                    fmt = QImage.Format_Grayscale16
                elif bits == 8:
                    fmt = QImage.Format_Grayscale8
            elif pixels.ndim == 3:
                logger.warning('Warning: Untested code-path (converting color FITS image)')
                if bits == 8 and pixels.shape[2] == 3:
                    fmt = QImage.Format_RGB888

            if fmt is None:
                raise NotImplementedError(
                    f'Unhandled data format: pixel array shape={pixels.shape}, type={pixels.dtype}, bps={bits}'
                )


            assert bits % 8 == 0, ('Eh?', bits)

            self._qimage = QImage(
                np.ascontiguousarray(pixels).tobytes(),
                w, h,
                w * c * (bits//8),
                fmt
            )

        return self._qimage


    @property
    def image_with_solution(self):
        if self._gnomonic_solution is None:
            self._gnomonic_solution = ImageWithGnomonicSolution(
                image=self.image, # np.ndarray
                center=ICRS(*self.to_radec(0, 0)),
                scale=self.compute_scale(),
                north_angle=self.compute_north_angle(),
            )
        return self._gnomonic_solution

    @property
    def image_size(self):
        return self._img_size

    @property
    def image(self):
        return self._orig_img

    def to_radec(self, x, y, relative_to='top_left'):
        """
        Returns: (ra, dec)
        """
        h, w = self.image_size
        if relative_to == 'top_left':
            x_q, y_q = x, (h - y) # h - y instead of y because of the y-flip. See __init__ for more
        elif relative_to == 'center':
            x_q, y_q = w/2.0 + x, h - (h/2.0 - y)
        elif relative_to == 'center_ydown':
            x_q, y_q = w/2.0 + x, h - (h/2.0 + y)
        else:
            raise NotImplementedError(f'Unhandled coordinate system: relative_to={relative_to}')

        return tuple(map(float, self._w.all_pix2world(x_q, y_q, 0)))

    def plot(self, ra, dec, linewidth=3, crosshair_size=5):
        self.plot_xy(*self.to_pixels(ra, dec), linewidth=linewidth, crosshair_size=crosshair_size) # pylint: disable=no-value-for-parameter

    def _ensure_plot_img(self):
        if self._plot_img is None:
            self._plot_img = self._orig_img.copy().convert("RGB")

    def plot_xy(self, x, y, linewidth=3, crosshair_size=5):
        """
        Marks the position (x, y) on a copy of the image
        """

        self._ensure_plot_img()

        if self._draw is None:
            self._draw = ImageDraw.Draw(self._plot_img)

        line = lambda *xy: self._draw.line(xy, width=int(linewidth * (self._plot_img.width / 512)))
        S = crosshair_size * self._plot_img.width / 128
        line(x - 2 * S, y, x - S, y)
        line(x + S, y, x + 2 * S, y)
        line(x, y - S, x, y - 2 * S)
        line(x, y + S, x, y + 2 * S)

    def save(self, path):

        self._ensure_plot_img()

        self._plot_img.save(path)

    def save_thumb(self, path, size=512):

        self._ensure_plot_img()

        img = self._plot_img.copy()
        img.thumbnail((size, size))
        img.save(path)

    def show(self):

        self._ensure_plot_img()

        self._plot_img.show()

    def show_thumb(self, size=512):
        self._ensure_plot_img()
        img = self._plot_img.copy()
        img.thumbnail((size, size))
        img.show()

def make_plot(data_sub, objects, candidate_set, target_file='/tmp/sep_plot.png'): # Useful for debugging
    fig, ax = plt.subplots()
    im = ax.imshow(np.maximum(data_sub,0)**0.1, cmap='gray', origin='upper')

    for i in range(len(objects)):
        e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                    width=6*objects['a'][i],
                    height=6*objects['b'][i],
                    angle=objects['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        if i not in candidate_set:
            e.set_linestyle(':')
            e.set_edgecolor('gold')
        else:
            e.set_edgecolor('red')
        ax.add_artist(e)
    plt.savefig(target_file)


def extract_sources_sep(
    np_data: np.ndarray, sep_threshold: float, binning=1,
    bg_block=None, bg_filter=None,
    max_aspect_ratio=4.0, noise_patch_threshold=0.20, hot_pixel_threshold=1,
    plot_file='/tmp/sep_plot.png', display_plot=False,
    output_intermediates=False,
):
    """
    Use SEP to perform source extraction

    1. Pre-process the image to apply `binning` and average the channels
    2. Perform background extraction using SEP routine
    3. Subtract the background
    4. Extract sources
    5. Plot the sources if requested
    6. Filter the list of sources to remove large detections and weird aspect ratios

    Return: if output_intermediates is False: a tuple:
               (flux-sorted xylist of detections, height, width)
            Otherwise a nestedtuple structure:
               ((flux-sorted xylist, h, w), (binned grayscale image, background, full list of detections))

    np_data: is an np.ndarray of any type, will be cast to np.float32
    sep_threshold: threshold for SEP (See SEP documentation)
    binning: bin the image by this factor before performing background extraction etc
    bg_block (int): block size for background extraction (See SEP documentation). Default value is 128/binning
    bg_filter (int): Convolution filter size for bkg extraction (See SEP docs). Default is 7/binning
    max_aspect_ratio: Discard candidate sources with aspect ratios more skewed than this
    noise_patch_threshold: Reject detections which are larger in size than this fraction of the image size
    hot_pixel_threshold: Reject detections that are smaller or equal to this number of pixels in FWHM
    output_intermediates: Also return the intermediate outputs (see "Return")
    plot_file: If not None, write a plot to this file
    display_plot: Also display the plot generated using `display` with a 30s timeout
    """

    global timing

    t0 = time.time()
    data = np_data.astype(np.float32)
    if binning != 1:
        if data.ndim == 2:
            data = data.reshape(data.shape[0], data.shape[1], 1)
        h, w, c = data.shape
        H, W = int(h/binning), int(w/binning)
        logger.info(f'Binning data to {H}x{W}x{c}')
        data = data[
            0:H * binning,
            0:W * binning,
            :
        ]
        data = data.reshape(H, binning, W, binning, c).mean(axis=(1, 3))
        # if scale_units in ('arcsecperpix', 'app'):
        #     if scale_low:
        #         scale_low *= binning
        #     if scale_high:
        #         scale_high *= binning
        # elif scale_units == 'focalmm':
        #     if scale_low:
        #         scale_low /= binning
        #     if scale_high:
        #         scale_high /= binning
        # elif scale_units in ('degwidth', 'degw', 'dw', 'arcminwidth', 'amw', 'aw'):
        #     pass
        # else:
        #     logger.warning(f'Do not know how to apply binning to unhandled scale units {scale_units}! Ignoring!!!')

    if data.ndim != 2:
        data = data.mean(axis=2) # Average the channels if present

    timing['03.2:data_conditioning'] = time.time() - t0
    t0 = time.time()

    bg_block = math.ceil(128/binning)
    bg_filter = math.ceil(7/binning)
    bkg = sep.Background(data, bw=bg_block, bh=bg_block, fw=bg_filter, fh=bg_filter)
    data_sub = data - bkg
    scale = bkg.globalrms

    timing['03.3:bkg_sub'] = time.time() - t0
    t0 = time.time()

    objects = sep.extract(data_sub, sep_threshold, err=bkg.rms())

    timing['03.4:extraction'] = time.time() - t0
    t0 = time.time()

    # Create XY List from objects that have acceptable aspect ratios
    candidates = []
    for i in range(len(objects)):
        a_ = objects['a'][i]
        b_ = objects['b'][i]
        a = max(a_, b_)
        b = min(a_, b_)
        if (
                ( # Aspect ratio filter
                    a <= max_aspect_ratio * b
                )
                and ( # Ignore large "LSB galaxy" extractions from SExtractor (noise patches)
                    a <= noise_patch_threshold * max(data.shape)
                )
                and ( # Remove hot-pixel detections which have tiny FWHM
                    b >= hot_pixel_threshold
                )
        ):
            candidates.append(i)

    logger.info(
        'Out of {} detected objects, {} met the aspect-ratio and size criteria'.format(
            len(objects), len(candidates)
        )
    )

    timing['03.5:filtering'] = time.time() - t0
    t0 = time.time()
    if plot_file is not None:
        candidate_set = set(candidates)
        make_plot(data_sub, objects, candidate_set, target_file=plot_file)
        if display_plot:
            plot_process = subprocess.Popen(['timeout', '30', 'display', plot_file])
        else:
            plot_process = None
    else:
        plot_process = None
    timing['03.6:plotting'] = time.time() - t0
    t0 = time.time()

    # Sort detected objects by brightness
    xylist_data = objects[['x', 'y', 'flux']][candidates]
    xylist_data = np.sort(xylist_data, order='flux')[::-1]

    # Apply binning: Don't care about flux
    height, width = data.shape[0] * binning, data.shape[1] * binning
    xylist_data['x'] = xylist_data['x'] * binning
    xylist_data['y'] = xylist_data['y'] * binning

    timing['03.7:sort'] = time.time() - t0
    t0 = time.time()


    if not output_intermediates:
        return (xylist_data, height, width)
    else:
        return (xylist_data, height, width), (data, bkg, objects)


def solve_field_xylist(
   xylist_data, height, width,
   top_k=0,
   scale_units='arcsecperpix', scale_high=None, scale_low=None,
   ra=None, dec=None, radius=None, parity=None,
   tmp_path_prefix='/tmp/solveme',
   binary='/usr/local/astrometry/bin/solve-field', timeout=None,
   extra_args=None,
   uniformize=0, no_remove_lines=True,
):
    """ Invoke the solve-field binary (`binary`) to solve a given XY-list (`xylist_data`)

    xylist_data, height, width: Tuple returned by `extract_sources_sep`
    top_k: If not zero, only take the top k sources in the xylist
    scale_units, scale_high, scale_low: Optional scale hint to astrometry.net (see solve-field manpages)
    ra, dec, radius: Optional position hint to astrometry.net (see solve-field manpages)
    parity: Optional parity hint. Possible values are `positive`, `negative`, None (=`any`, `both`)
    tmp_path_prefix: Prefix path for temporary files (.xyls, .axy...) including output WCS
    timeout: Wall-clock time timeout for solve-field
    extra_args: String containing extra arguments to solve-field to be appended to the command line as-is

    Return: Path to the WCS file
    """

    global timing

    t0 = time.time()
    tmp_files = ['{}.{}'.format(tmp_path_prefix, ext) for ext in ('fits', 'wcs', 'solved', 'xyls')]

    _, wcs_file, solved_file, xyls_file = tmp_files
    for tmp_file in tmp_files:
        if os.path.isfile(tmp_file):
            os.unlink(tmp_file)

    timing['03.1:cleanup'] = time.time() - t0
    t0 = time.time()

    if top_k != 0:
        if len(xylist_data) > top_k:
            logger.warning(f'Truncating {len(xylist_data)} detections to {top_k} as requested')
        xylist_data = xylist_data[:top_k]

    # Write to XYlist file
    hdu = fits.BinTableHDU(data=xylist_data)
    xyls_file = tmp_path_prefix + '.xyls'
    wcs_file = tmp_path_prefix + '.wcs'
    hdu.writeto(xyls_file)

    timing['03.8:bin_xylist_and_hdu'] = time.time() - t0
    t0 = time.time()

    command = [
        binary,
        xyls_file,
        '--height',
        str(height),
        '--width',
        str(width),
        '--x-column',
        'x',
        '--y-column',
        'y',
        '--sort-column',
        'flux',
        '--overwrite',
        '--no-plots',
        '--new-fits',
        'none',
        '--wcs',
        wcs_file,
        '--corr',
        'none',
        '--pnm',
        'none',
        '--rdls',
        'none',
        '--solved',
        solved_file,
    ]
    # if top_k != 0:
    #     command += ['-d', top_k]
    if scale_units:
        command += ['--scale-units', scale_units]
    if scale_low:
        command += ['--scale-low', str(scale_low)]
    if scale_high:
        command += ['--scale-high', str(scale_high)]

    if ra:
        command += ['--ra', str(convert_ra(ra))]
    if dec:
        command += ['--dec', str(convert_dec(dec))]
    if radius:
        command += ['--radius', str(radius)]

    if parity:
        if parity.startswith('pos'):
            command += ['--parity', 'pos']
        elif parity.startswith('neg'):
            command += ['--parity', 'neg']
        elif parity in ('any', 'both'):
            pass
        else:
            raise ValueError(f'Invalid parity choice: {parity}')

    if uniformize is not None:
        command += ['--uniformize', str(int(uniformize))]

    if no_remove_lines:
        command += ['--no-remove-lines']

    if extra_args and extra_args != []:
        command += extra_args

    if timeout:
        # command += ['--cpulimit', str(timeout)]
        command = ['timeout', str(timeout)] + command


    print('Executing: `{}`'.format(' '.join(command)), file=sys.stderr)
    result = subprocess.call(command)
    if result != 0 or not os.path.isfile(solved_file):
        raise RuntimeError('Plate solving command `{}` failed with error code: {}'.format(' '.join(command), result))
    timing['03.9:solve'] = time.time() - t0

    return wcs_file


def solve_field_sep(np_data, # Debayered if needed (channels will be averaged in `extract_sources_sep`)
                    sep_threshold, binning=1, top_k=0,
                    plot_detections=True,
                    binary='/usr/local/astrometry/bin/solve-field', tmp_path_prefix='/tmp/solveme',
                    timeout=15,
                    scale_units='arcsecperpix', scale_high=None, scale_low=None,
                    ra=None, dec=None, radius=None, parity=None,
):
    logger.info('Extracting sources with SEP')

    global timing
    timing = {}

    if np_data.dtype in (np.uint8, np.int8):
        logger.warning('Received 8-bit data. Please check your CCD settings!')

    xylist_data, height, width = extract_sources_sep(
        np_data, sep_threshold, binning=binning,
        bg_block=None, bg_filter=None, max_aspect_ratio=4.0, output_intermediates=False,
        plot_file=('/tmp/sep_plot.png' if plot_detections == True else None), display_plot=plot_detections,
    )
    # DEBUG
    print('Candidate objects [Check sort order!]:', file=sys.stderr)
    print(xylist_data, file=sys.stderr)

    wcs_file = solve_field_xylist(
       xylist_data, height, width,
       top_k=top_k,
       scale_units=scale_units, scale_high=scale_high, scale_low=scale_low,
       ra=ra, dec=dec, radius=radius, parity=parity,
       tmp_path_prefix=tmp_path_prefix,
       binary=binary, timeout=timeout,
    )

    t0 = time.time()
    ps = PlateSolution(np_data, wcs_file)
    timing['03.a:result'] = time.time() - t0
    return ps

def solve_field(fits_data: bytes, # N.B. Must be debayered if needed
                binary='/usr/local/astrometry/bin/solve-field',
                tmp_path_prefix='/tmp/solveme',
                binning=1, top_k=0,
                timeout=15,
                scale_units='arcsecperpix', scale_high=None, scale_low=None,
                ra=None, dec=None, radius=None,
):

    print('Entered solve_field', file=sys.stderr, flush=True)

    tmp_files = ['{}.{}'.format(tmp_path_prefix, ext) for ext in ('fits', 'wcs', 'solved', 'xyls')]

    for tmp_file in tmp_files:
        if os.path.isfile(tmp_file):
            os.unlink(tmp_file)

    fits_file, wcs_file, solved_file, _ = tmp_files

    with open(fits_file, 'wb') as f:
        f.write(fits_data)

    command = [
        binary,
        '--fits-image',
        fits_file,
        '--overwrite',
        '--no-plots',
        '--new-fits',
        'none',
        '--wcs',
        wcs_file,
        '--corr',
        'none',
        '--pnm',
        'none',
        '--rdls',
        'none',
        '--solved',
        solved_file,
    ]

    # Downsampling and top-k stars
    if binning != 1:
        command += ['-z', binning]
    if top_k != 0:
        command += ['-d', top_k]

    if scale_units:
        command += ['--scale-units', scale_units]
    if scale_low:
        command += ['--scale-low', str(scale_low)]
    if scale_high:
        command += ['--scale-high', str(scale_high)]

    if ra:
        command += ['--ra', str(convert_ra(ra))]
    if dec:
        command += ['--dec', str(convert_dec(dec))]
    if radius:
        command += ['--radius', str(radius)]

    if timeout:
        # command += ['--cpulimit', str(timeout)]
        command = ['timeout', str(timeout)] + command

    print('Executing: `{}`'.format(' '.join(command)), file=sys.stderr)
    result = subprocess.call(command)
    if result != 0 or not os.path.isfile(solved_file):
        raise RuntimeError('Plate solving command `{}` failed with error code: {}'.format(' '.join(command), result))

    return PlateSolution(fits_file, wcs_file)
