import pytest

torch = pytest.importorskip("torch")

from ml_course.cnn import TinyCNN, make_bar_dataset


def test_cnn_data_and_forward_shapes() -> None:
    dataset = make_bar_dataset(samples=12, image_size=16)
    images, labels = dataset.tensors
    assert images.shape == (12, 1, 16, 16)
    assert labels.shape == (12,)
    logits = TinyCNN()(images)
    assert logits.shape == (12, 2)
    assert torch.isfinite(logits).all()

