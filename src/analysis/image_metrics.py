from wpca import WPCA
import numpy as np

def image_mean(img): return img.squeeze().numpy().mean()

def image_sigma(img): return img.squeeze().numpy().std()

def active_pixels(img, thr=0): return np.sum(img.squeeze().numpy() > thr)

def active_mean(img, thr=0):
    img = img.squeeze().numpy()
    return img[img > thr].mean()

def active_sigma(img, thr=0):
    img = img.squeeze().numpy()
    return img[img > thr].std()

def wpca_angle_and_ratio(img):
    img_np = img.squeeze().numpy()
    active_coords = np.array(img_np.nonzero())
    weights = img_np[active_coords[0], active_coords[1]]

    wpca = WPCA(n_components=2)
    wpca.fit(active_coords.T, weights=np.tile(weights, (2, 1)).T)

    y, x = wpca.components_[0]  # rows, columns --> y, x
    v_max = np.max(wpca.explained_variance_ratio_)
    return (np.arctan2(y, x) / np.pi * 180) % 180, v_max

def center_of_mass(img):
    img = img.squeeze().numpy()
    coordinates = np.mgrid[0:img.shape[0], 0:img.shape[1]].reshape(2, -1).T
    weights = img.reshape(-1)
    COM = np.average(coordinates, weights=weights, axis=0)
    rho = np.sqrt(((COM - 40)**2).sum())
    theta = np.angle(COM[0] - 40 + 1j * (COM[1] - 40), deg=True)%360
    return COM, rho, theta

def signal_scatter(img):
    img = img.squeeze().numpy()
    coordinates = np.mgrid[0:img.shape[0], 0:img.shape[1]].reshape(2, -1).T
    weights = img.reshape(-1)
    COM = np.average(coordinates, weights=weights, axis=0)
    sigma = np.average(
        np.sqrt(((coordinates - COM)**2).sum(axis=1)),
        weights=weights,
        axis=0)
    return sigma

def batch_metric(batch, metric):
    return np.array([metric(img) for img in batch])

