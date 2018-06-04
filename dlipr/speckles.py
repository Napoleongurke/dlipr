"""
Module to load the dataset of speckled images from the course.
"""

from dlipr.utils import get_datapath, Dataset, maybe_savefig
import numpy as np
import h5py
import matplotlib.pyplot as plt


def load_data():
    """Load the dataset of images of simulated GISAXS measurements
    (Grazing Incidence Small-Angle X-ray Scattering).
    The dataset contains the speckled (noisy) images along with the underlying
    unspeckled images for training a denoising autoencoder.

    Returns:
        Dataset: Speckled and unspeckled images (20000 train, 5500 test)
    """
    data = Dataset()

    # monkey-patch the plot_examples function
    def monkeypatch_method(cls):
        def decorator(func):
            setattr(cls, func.__name__, func)
            return func
        return decorator

    @monkeypatch_method(Dataset)
    def plot_examples(self, num_examples=10, fname=None):
        """Plot the first examples of speckled and unspeckled images.

        Args:
            num_examples (int, optional): number of examples to plot for each class
            fname (str, optional): filename for saving the plot
        """
        fig, axes = plt.subplots(2, num_examples, figsize=(num_examples, 2))
        for i, X in enumerate((self.X_train, self.Y_train)):
            for j in range(num_examples):
                ax = axes[i, j]
                ax.imshow(X[j])
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0, 0].set_ylabel('speckled')
        axes[1, 0].set_ylabel('unspeckled')
        maybe_savefig(fig, fname)

    fname = get_datapath('AutoEncoder/data.h5')
    fin = h5py.File(fname)['data']

    def format(X):
        return np.swapaxes(X, 0, 1).reshape((-1, 64, 64))

    speckle = format(fin['speckle_images'])
    normal = format(fin['normal_images'])

    data.X_train, data.X_test = np.split(speckle, [20000])
    data.Y_train, data.Y_test = np.split(normal, [20000])
    return data
