import numpy as np

from thesis.analysis.gradcam import gradcam, gradcam_raw, gradcam_weights


def test_weights_are_spatial_mean():
    g = np.zeros((2, 2, 3))
    g[..., 0] = [[1, 3], [5, 7]]
    assert np.allclose(gradcam_weights(g), [4, 0, 0])


def test_raw_is_relu_of_weighted_sum():
    rng = np.random.default_rng(0)
    a = rng.random((7, 7, 4))
    g = np.broadcast_to(np.array([1.0, -2.0, 0.5, 0.0]), (7, 7, 4))
    expected = np.maximum(a @ np.array([1.0, -2.0, 0.5, 0.0]), 0)
    assert np.allclose(gradcam_raw(a, g), expected)


def test_zero_gradient_gives_empty_map():
    cam, empty = gradcam(np.ones((7, 7, 4)), np.zeros((7, 7, 4)), size=32)
    assert empty and cam.shape == (32, 32) and not cam.any()


def test_hot_channel_region():
    a = np.zeros((7, 7, 2))
    a[1:3, 1:3, 0] = 1.0
    g = np.zeros((7, 7, 2))
    g[..., 0] = 1.0
    cam, empty = gradcam(a, g, size=7)
    assert not empty and cam[2, 2] > cam[5, 5]
