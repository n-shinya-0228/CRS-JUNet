"""Internal checks; synthetic examples are never dataset accuracy results."""
import numpy as np
import torch
from .model import SphericalBEVFusion
from .projection import build_dual_projection
from .trainer import Trainer


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = SphericalBEVFusion(20)
    bev = torch.randn(2, 7, 32, 48)
    sphere = torch.randn(2, 6, 16, 48)
    grid = torch.zeros(2, 32, 48, 2)
    valid = torch.ones(2, 32, 48)
    features = {}
    handles = [model.bev_decoder.register_forward_hook(
        lambda module, inputs, output: features.update(bev=output)),
        model.sphere_decoder.register_forward_hook(
        lambda module, inputs, output: features.update(sphere=output))]
    output = model(bev, sphere, grid, valid)
    assert features['bev'].shape == (2, 128, 32, 48)
    assert features['sphere'].shape == (2, 128, 16, 48)
    expected = model.head(features['bev'] + torch.nn.functional.grid_sample(
        features['sphere'], grid, align_corners=True))
    torch.testing.assert_close(output['bev_logits'], expected)
    loss = torch.nn.functional.cross_entropy(output['bev_logits'],
        torch.randint(1, 20, (2, 32, 48)))
    loss.backward()
    for encoder in (model.bev_encoder, model.sphere_encoder):
        grad = encoder.stem.conv.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    for handle in handles:
        handle.remove()
    model.eval()
    with torch.no_grad():
        zero = torch.zeros_like(valid)
        a = model(bev, sphere, grid, zero)['bev_logits']
        b = model(bev, sphere * 10, grid, zero)['bev_logits']
        torch.testing.assert_close(a, b)
    # First point is higher, although the lower point comes last in input order.
    points = np.array([[10, 0, 0.1], [10, 0, 0.0]], dtype=np.float32)
    sensor = dict(height=64, width=512, fov_up=3., fov_down=-23.,
                  means=[0]*5, stds=[1]*5)
    sample = build_dual_projection(points, np.array([0.8, 0.4]), np.array([1, 2]), sensor)
    assert sample['bev_labels'][100, 256] == 1
    assert sample['bev_to_sphere_valid'][100, 256] == 1
    empty = build_dual_projection(np.empty((0, 3)), np.empty(0), np.empty(0), sensor)
    assert empty['bev_to_sphere_valid'].sum() == 0
    for key in ('bev', 'sphere', 'bev_to_sphere'):
        assert np.isfinite(empty[key]).all()
    trainer = Trainer.__new__(Trainer)
    trainer.nclasses = 3
    metrics = trainer._metrics(torch.tensor([[0, 0, 0], [0, 3, 1], [0, 1, 3]]))
    assert abs(metrics['miou'] - 0.6) < 1e-6
    assert abs(metrics['accuracy'] - 0.75) < 1e-6
    print('PASS: feature addition, both branch gradients, validity mask, Polar3 representative, empty scan, metrics')


if __name__ == '__main__':
    main()
