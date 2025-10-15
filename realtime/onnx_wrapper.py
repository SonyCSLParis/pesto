"""
ONNX wrapper for PESTO model that manages cache state explicitly.
ONNX Runtime is stateless, so we need to externalize the cache state.
"""

import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
from pesto.utils.cached_conv import CachedConv1d


class StatelessPESTO(nn.Module):
    """
    Wrapper for PESTO model to work with ONNX Runtime by managing cache state explicitly.

    This wrapper extracts the cache state from the model's cached convolutions,
    returns it as an output, and accepts it as an input for the next inference.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.cache_size_dict = self.get_cache_names()
        self.cache_size = sum([math.prod(v) for v in self.cache_size_dict.values()])

    def get_cache_names(self) -> Dict[str, Tuple[int]]:
        """Get information about all cached modules and their cache shapes."""
        cache_size_dict: Dict[str, Tuple[int]] = {}

        for name, m in self.model.named_modules():
            if isinstance(m, CachedConv1d):
                # CachedConv1d has a CachedPadding1d as its cache attribute
                if hasattr(m.cache, "pad"):
                    # Store the full shape for proper reconstruction
                    cache_size_dict[name + "-cache"] = tuple(m.cache.pad.shape)

        return cache_size_dict

    def gather_caches(self, batch_size: int = 1) -> torch.Tensor:
        """Gather all cache tensors into a single flattened tensor."""
        cache_tensors = []

        for name, m in self.model.named_modules():
            if isinstance(m, CachedConv1d):
                if hasattr(m.cache, "pad"):
                    # Flatten the entire cache tensor
                    c = m.cache.pad.flatten()
                    cache_tensors.append(c)

        if cache_tensors:
            cache_tensor = torch.cat(cache_tensors, dim=0)
            # Repeat to match batch size - separate cache state for each sample
            return cache_tensor.unsqueeze(0).expand(batch_size, -1)
        else:
            # Return empty tensor if no caches
            return torch.empty(batch_size, 0)

    def set_caches(self, cache: torch.Tensor) -> None:
        """Set all cache tensors from a single flattened tensor."""
        if cache.numel() == 0:
            return

        # Use the first sample's cache state (batch dimension 0)
        # All samples in a batch share the same cache state during inference
        cache_flat = cache[0]  # Shape: (cache_size,)
        cache_pointer = 0

        for name, m in self.model.named_modules():
            if isinstance(m, CachedConv1d):
                if hasattr(m.cache, "pad"):
                    shape = self.cache_size_dict[name + "-cache"]
                    c_numel = math.prod(shape)

                    # Extract the cache slice and reshape to original shape
                    cache_slice = cache_flat[cache_pointer : cache_pointer + c_numel]
                    m.cache.pad = cache_slice.view(shape)

                    cache_pointer = cache_pointer + c_numel

    def forward(
        self, audio: torch.Tensor, cache: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that manages cache state explicitly.

        Args:
            audio: Input audio tensor
            cache: Flattened cache state from previous inference

        Returns:
            Tuple of (prediction, confidence, volume, activations, cache_out)
        """
        # Set the cache state
        self.set_caches(cache)

        # Run the model
        model_output = self.model(audio)
        preds, confidence, vol, activations = model_output

        # Gather the updated cache state
        batch_size = audio.shape[0]
        cache_out = self.gather_caches(batch_size)

        return preds, confidence, vol, activations, cache_out

    def init_cache(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """Initialize cache state with zeros."""
        if self.cache_size == 0:
            return torch.empty(batch_size, 0, device=device)
        return torch.zeros(batch_size, self.cache_size, device=device)
