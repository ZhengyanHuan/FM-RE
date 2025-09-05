# plot_func.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from dataset import distance_data_to_plane
from ot.sliced import sliced_wasserstein_distance as SWD

def plot_scatter3D(xyz, center = 0, range = 4, elev_angle = 15, azim_angle = 60,  abcd = None, ref = None):
    xyz[-1] += np.random.randn(*xyz[-1].shape)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    xyz_density = gaussian_kde(xyz.T)(xyz.T)

    # Interpolate density values to get color for each point


    # Plot with color mapping based on density
    p = ax.scatter(xyz[:-1, 0], xyz[:-1, 1], xyz[:-1, 2], c=xyz_density[:-1], cmap='viridis', marker='o', s=5)

    # xx, zz = np.meshgrid(np.linspace(center-range, center+range, 10), np.linspace(center-range, center+range, 10))
    # yy = -zz-xx +1 # Since x = y
    # ax.plot_surface(xx, yy, zz, alpha=0.3, color='red')
    cb = fig.colorbar(p, ax=ax, shrink=0.5, aspect=5)
    cb.set_label('3D Density')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_ylim(10, 0)
    ax.set_xlim([(center-range)*np.sqrt(2), (center+range)*np.sqrt(2)])
    ax.set_ylim([(center-range)*np.sqrt(2), (center+range)*np.sqrt(2)])  # Inverted Y-axis for left-handed system
    ax.set_zlim([center-range, center+range])
    title = '3D Scatter'
    if abcd is not None:
        distance, avgdistance = distance_data_to_plane(xyz, abcd)
        title += '  Avg_distance to plane = %.5f' % avgdistance

    if ref is not None:
        distance_SWD = compute_SWD(ref, xyz)
        title += '  SWD distance = %.5f' % distance_SWD
    plt.title(title)

    ax.view_init(elev=elev_angle, azim=azim_angle)

    plt.show()



def compute_SWD(ref, pred, sample_size=None):
    sample_size = min(pred.shape[0], ref.shape[0]) if sample_size is None else sample_size
    pred = shuffle(pred, sample_size=sample_size)
    ref = shuffle(ref, sample_size=sample_size)

    return SWD(pred, ref)

def shuffle(x, sample_size):
    """
        x: (B, D)
        ===
        return: (sample_size, D)
    """
    idx = np.random.choice(x.shape[0], sample_size, replace=False)
    return x[idx]
