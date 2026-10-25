"""SCNet checkpoint compatibility and supported architecture checks."""
import torch
import pytest
from sep.models import build_model
from sep.inference import load_model


def test_scnet_checkpoint_round_trip(tmp_path):
    args = {"arch": "scnet", "scnet_width": 32, "layers": 2}
    model = build_model(args).eval()
    checkpoint = tmp_path / "best.pt"
    torch.save({"args": args, "model": model.state_dict()}, checkpoint)
    loaded = load_model(checkpoint)
    audio = torch.randn(1, 2, 8192)
    query = torch.tensor([0])
    with torch.no_grad():
        expected = model(audio, query)
        actual = loaded(audio, query)
    assert actual.shape == audio.shape
    torch.testing.assert_close(actual, expected)


def test_unsupported_checkpoint_architecture():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        build_model({"arch": "unsupported"})
