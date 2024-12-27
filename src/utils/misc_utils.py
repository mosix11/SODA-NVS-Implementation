import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
from PIL import Image
import io
import PIL
import collections
import inspect
import os
from tqdm import tqdm
import requests

import hashlib
import tarfile
import zipfile

import zipfile
import shutil
from multiprocessing import cpu_count



def img_is_color(img):

    if len(img.shape) == 3:
        # Check the color channels to see if they're all the same.
        c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
        if (c1 == c2).all() and (c2 == c3).all():
            return True

    return False


def download_file_fast(url, output_path):
    """Downloads a file with a progress bar and handles retries."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, stream=True, headers=headers)
    total_size = int(response.headers.get('content-length', 0))
    chunk_size = 1024 * 1024  # 1MB chunks

    with open(output_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(output_path)}",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            file.write(data)
            bar.update(len(data))
            
def extract_zip_multithreaded(zip_path, extract_to):
    """Extracts a zip file using the maximum number of available threads."""
    # Check if p7zip is installed
    if shutil.which("7z") is None:
        raise EnvironmentError("7z command-line utility is required for multi-threaded extraction. Install p7zip.")

    # Run the extraction with maximum threads
    command = f"7z x -o{extract_to} {zip_path} -mmt{cpu_count()}"
    os.system(command)


def download_file(url, saving_path, name):
    response = requests.get(url, stream=True)
    
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    with open(os.path.join(saving_path, name), "wb") as handle:
        for data in tqdm(response.iter_content()):
            handle.write(data)

def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=True, num_cols=2, figsize=(20, 10), title_fontsize=30):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.
    Parameters:
    ----------
    images: list
        List of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    '''

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)

    fig.tight_layout()
    _ = plt.show()


def show_image_categories(img_list, categories):
    assert isinstance(img_list, list)
    assert len(img_list) > 0
    assert isinstance(img_list[0], np.ndarray)
    
    NUM_SAMPLES = len(img_list)
    NUM_COLUMNS = len(categories)
    NUM_ROWS = int(NUM_SAMPLES/NUM_COLUMNS)

    fig = plt.figure(figsize=(10,10))
    grid = ImageGrid(fig, 111, nrows_ncols=(NUM_ROWS, NUM_COLUMNS), axes_pad=0.05)

    for category_id, category in enumerate(categories):
        i = category_id
        for r in range(NUM_ROWS):
            ax = grid[r*NUM_COLUMNS + i]
            # print(f'image {} at grid {r*NUM_COLUMNS + i}')
            img = img_list[(i*NUM_ROWS) + r]
            ax.imshow(img / 255.)
            ax.axis('off')
            if r==0:
                print(categories[i])
                ax.text(0.5, 0.5, categories[i], fontsize='medium')
    plt.show()

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot_graph(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.show()


def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices."""
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.show()

class ProgressBoard():
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.xscale = xscale
        self.yscale = yscale
        self.ls = ls
        self.colors = colors
        self.fig = fig
        self.axes = axes
        self.figsize = figsize
        self.display = display
        plt.ion()


    def draw(self, x, y, label, every_n=1):
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]),
                        mean([p.y for p in points])))
        points.clear()
        if not self.display:
            return
        if self.fig is None:
            self.fig = plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):
            plt_lines.append(plt.plot([p.x for p in v], [p.y for p in v],
                                        linestyle=ls, color=color)[0])
            labels.append(k)
        axes = self.axes if self.axes else plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        
    def save_fig(self, name='temp', dir='./'):
        self.fig.savefig(name+'.png')





def download(url, folder='./data', sha1_hash=None): 
    """Download a file to folder and return the local filepath."""
    if not url.startswith('http'):
        # For back compatability
        # url, sha1_hash = DATA_HUB[url]
        pass
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    # Download
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname



def extract(filename, folder=None):  #@save
    """Extract a zip/tar file into folder."""
    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)
    assert ext in ('.zip', '.tar', '.gz'), 'Only support zip/tar files.'
    if ext == '.zip':
        fp = zipfile.ZipFile(filename, 'r')
    else:
        fp = tarfile.open(filename, 'r')
    if folder is None:
        folder = base_dir
    fp.extractall(folder)


# def visualize_samples(data_loader, num_category, num_sample_each_cat=5):
#     temp_batch = next(iter(data_loader))
#     X_batch, y_batch = temp_batch
#     for i in range(int(num_category*num_sample_each_cat/5)):
#         new_batch = next(iter(data_loader))
#         new_X_batch, new_y_batch = new_batch
#         X_batch = torch.cat((X_batch, new_X_batch), dim=0)
#         y_batch = torch.cat((y_batch, new_y_batch), dim=0)
    
#     X, y = [], []
#     for i in range(num_category):
#         X_i = X_batch[y_batch==i]
#         for j in range(num_sample_each_cat):
#             X.append(X_i[j].squeeze().numpy())
#             y.append(i)
    
#     y = dataset.text_labels(y)
#     categories = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     utils.show_image_categories(X, categories)
        

def check_len(a, n):  
    """Check the length of a list."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'

def check_shape(a, shape): 
    """Check the shape of a tensor."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'
            
            
            
class Accumulator:  
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    

class AverageMeter(object):
    """Computes and stores the average and current value (moving average)"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)
    
def plot_to_tensorboard(writer, fig, tag, step):
    """
    Log a Matplotlib figure to TensorBoard.

    Parameters:
    - writer: Instance of SummaryWriter.
    - fig: Matplotlib figure.
    - tag: Name of the image in TensorBoard.
    - step: Training step or epoch.
    """
    # Convert Matplotlib figure to a numpy array
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    
    # Convert numpy array to tensor
    img_tensor = torch.tensor(img_array).permute(2, 0, 1)
    
    # Add image to TensorBoard
    writer.add_image(tag, img_tensor, step)