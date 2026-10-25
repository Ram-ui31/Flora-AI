"""Keep small CPU tests predictable on development machines and CI runners."""
import torch


def pytest_sessionstart(session):
    torch.set_num_threads(1)
