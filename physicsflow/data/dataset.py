import torch


def get_dataset(config: dict, split: str="train") -> torch.utils.data.Dataset:
    # Make sure to split your data somehow, either return two datasets here or use the split option
    return torch.utils.data.Dataset()  # Placeholder for actual dataset implementation
