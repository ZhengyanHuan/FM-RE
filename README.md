# FM-RE: Constraint-Aware Flow Matching via Randomized Exploration

Research code for **Constraint-Aware Flow Matching via Randomized Exploration**, by Zhengyan Huan, Jacob Boerma, Li-Ping Liu, and Shuchin Aeron, published in *Transactions on Machine Learning Research* (April 2026).

[Paper (arXiv v2)](https://arxiv.org/abs/2508.13316v2) | [OpenReview](https://openreview.net/forum?id=OR4h9WPJhV)

The repository contains experiment-specific PyTorch implementations and Jupyter notebooks for constrained synthetic data generation, MNIST generation with image attribute constraints, and adversarial example generation for hard-label image classifiers.

## Method

Conventional flow matching (FM) can generate samples outside a constraint set even when the training data satisfy the constraint. The paper studies two ways to incorporate constraint information during training:

- **FM-DD** uses a differentiable distance to the constraint set as an additional penalty.
- **FM-RE** uses randomized velocity exploration and a membership oracle to learn a flow with improved constraint satisfaction. Its two-stage formulation uses ordinary flow matching before a split time `t0`, then jointly optimizes flow matching and constraint satisfaction after `t0`. Sampling uses the learned mean velocity.


## Repository layout

Folder names refer to the experiment sections in the paper.

| Folder | Paper section | Experiment description |
| --- | --- | --- |
| [`6p1_box_2box`](6p1_box_2box/) | 6.1 | Two-dimensional box and disconnected two-box distributions. |
| [`6p1_l2ball`](6p1_l2ball/) | 6.1 | Gaussian mixtures restricted to the unit Euclidean ball in 8 or 20 dimensions. |
| [`6p1_subspace`](6p1_subspace/) | 6.1 | A Gaussian projected onto a 9-dimensional hyperplane in 10 dimensions. |
| [`6p2_MNIST_brightness_thickness`](6p2_MNIST_brightness_thickness/) | 6.2 | MNIST generation constrained by brightness or maximum stroke thickness. |
| [`6p3_MNIST_LeNet`](6p3_MNIST_LeNet/) | 6.3 | Adversarial MNIST generation against LeNet-5. |
| [`6p3_CIFAR_ResNet`](6p3_CIFAR_ResNet/) | 6.3 | Adversarial CIFAR-10 generation against ResNet-50. |

Each folder has its own `configs.py`. These are independent experiments, not an installable Python package or a single command-line application. The `test.ipynb` files are evaluation notebooks, not automated unit tests.


## Data and checkpoint preparation

The checkout does not include generated synthetic datasets or trained flow-model checkpoints. Paths in the notebooks refer to local experiment artifacts that must be generated, supplied, or updated.


### Synthetic datasets

| Experiment | Preparation |
| --- | --- |
| Box / two boxes | Set `configs.dataset` and dataset.py's `dataset_type` consistently, then run python dataset.py. The pairs are `uniform_box` / `uniform`, `two_uniform_box` / `uniform2`, and `cropped_Gaussian` / `cropped_Gaussian`. Outputs are written under `data/`. The checked-in defaults select different cases in these two files. |
| Unit ball | Set `xdim` in dataset_MDM.py to match `configs.d_model`, then run python dataset_MDM.py. Outputs are `data/MDMl2ball_dim8.npy` or `data/MDMl2ball_dim20.npy`. The current generator defaults to 20 dimensions, while `configs.py` defaults to 8. |
| Subspace | Run python dataset.py to create `data/projected_10DGaussian.npy`. The default constraint is `sum(x) + 10 = 0`; constraint evaluation allows a distance tolerance of `5e-4`. |

The generators default to millions of samples. Reduce their sample counts when checking the setup. The box generator executes when imported, so run it deliberately as a script.

### MNIST and CIFAR-10

Image experiments download datasets through `torchvision` with `download=True` into their local `data/` directories.

- **MNIST attributes:** `load.myMMIST().get_data_brightness()` filters by the number of pixels above `0.5`; the current bounds require more than 100 and fewer than 784 such pixels. `get_data()` filters by maximum thickness strictly between 2 and 3, measured using an OpenCV distance transform. Choose one dataset-loading cell and use the matching `constraint_type='brightness'` or `'thickness'` in `PGFM.train2_2stage(...)`. Running both loading cells leaves the brightness dataset in `train_set`.
- **MNIST adversarial examples:** the notebooks resize images to `32 x 32` and split the MNIST test set 80:20 for generator training and evaluation. The classifier checkpoint is included at `6p3_MNIST_LeNet/externel/lenet_epoch=12_test_acc=0.991.pth`.
- **CIFAR-10 adversarial examples:** the notebooks normalize images and split the CIFAR-10 test set 80:20. Supply `6p3_CIFAR_ResNet/externel/resnet50_cifar10_lr01.pth` before constructing the classifier. The loader expects a dictionary whose `net` entry is a state dictionary compatible with `externel/resnet_models.py`'s `ResNet50()`. 

## Training and sampling

For synthetic and attribute-constrained generation, the general workflow is:

1. Configure the dataset, device, training iterations, batch sizes, learning rate, and checkpoint names in the experiment's `configs.py`.
2. Prepare the data and train regular FM with `FMfuncs.OTFlowMatching().train(...)`, or `FM_funcs.OTFlowMatching().train(...)` for MNIST attributes.
3. (Use regular FM model as a starting point) Save the regular FM model and pass its path to the selected constraint-aware training routine.
4. Load the matching model architecture and checkpoint for sampling; use the notebook's evaluation cells for the selected experiment.


## Evaluation

- **Synthetic data:** sliced Wasserstein distance (SWD), constraint violation counts/rates, and distance to the constraint set for the subspace case.
- **MNIST attributes:** attribute satisfaction and feature-based FID through `fid/fid_compute.py`; this uses LeNet features rather than an Inception network. `evaluate.py` also provides PCA-based SWD utilities.
- **Adversarial examples:** classifier accuracy on generated images, perturbation Euclidean norms, and visual comparisons. Correct classification corresponds to a constraint violation in this experiment's formulation.


## Citation

```bibtex
@article{huan2026constraintaware,
  title   = {Constraint-Aware Flow Matching via Randomized Exploration},
  author  = {Huan, Zhengyan and Boerma, Jacob and Liu, Li-Ping and Aeron, Shuchin},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {https://openreview.net/forum?id=OR4h9WPJhV}
}
```
