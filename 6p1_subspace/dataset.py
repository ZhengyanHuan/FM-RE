import numpy as np
import configs


def Generate_3D_Gaussian(n=int(1e6), dim = configs.d_model):
    center = np.zeros(dim)
    sigma = np.eye(dim)
    out_Nd = np.random.multivariate_normal(center, sigma, n)
    return out_Nd

def project_points_to_plane(points_Nd, coeffs = None):

    if coeffs is None:
        coeffs = np.ones(points_Nd.shape[1]+1)
    normal = coeffs[:-1]
    numerator = np.dot(points_Nd, normal) + coeffs[-1]
    denominator = np.dot(normal, normal)

    # Compute projection using the formula
    projected_points = points_Nd - (numerator[:, np.newaxis] / denominator) * normal

    return projected_points

def distance_data_to_plane(points_Nd, coeffs = None):
    if coeffs is None:
        coeffs = np.ones(points_Nd.shape[1]+1)
    normal = coeffs[:-1]
    numerator = np.abs(np.dot(points_Nd, normal) + coeffs[-1])
    denominator = np.linalg.norm(normal)
    distance = numerator / denominator
    return distance, np.mean(distance)

if __name__ == '__main__':
    x = Generate_3D_Gaussian()
    mycoeff = np.ones(configs.d_model+1)
    mycoeff[-1] = 10
    xp = project_points_to_plane(x, coeffs= mycoeff)
    np.save("./data/projected_10DGaussian.npy", xp)