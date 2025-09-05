import numpy as np
from sklearn.decomposition import PCA
from ot.sliced import sliced_wasserstein_distance as SWD


class distribution_evaluator:
    def __init__(self, dataset, n_components = 40):
        self.dataset = dataset.cpu().numpy()
        self.MNIST_PCA = self.compute_pca(self.dataset, num_components = n_components)
        self.n_components = n_components

    def compute_pca(self, images, num_components = 40) :
        flattened = images.reshape(images.shape[0], -1)
        pca = PCA(n_components=num_components)
        pca.fit(flattened)

        total_variance = np.sum(pca.explained_variance_ratio_[:num_components])
        print("Number of components: %d" % num_components)
        print("Total variance ratio:", total_variance)

        return pca

    def pca_transform(self, images) :
        images = images.cpu().numpy()
        flattened = images.reshape(images.shape[0], -1)
        transformed = self.MNIST_PCA.transform(flattened)
        return transformed

    def compute_SWD(self, ref, pred, sample_size=None):
        sample_size = min(pred.shape[0], ref.shape[0]) if sample_size is None else sample_size
        pred = self.shuffle(pred, sample_size=sample_size)
        ref = self.shuffle(ref, sample_size=sample_size)

        return SWD(pred, ref)

    def shuffle(self, x, sample_size):
        """
            x: (B, D)
            ===
            return: (sample_size, D)
        """
        idx = np.random.choice(x.shape[0], sample_size, replace=False)
        return x[idx]


    def SWD_after_PCA(self, ref, pred, sample_size=None):
        ref_pca = self.pca_transform(ref)
        pred_pca = self.pca_transform(pred)

        return self.compute_SWD(ref_pca, pred_pca, sample_size = sample_size)

