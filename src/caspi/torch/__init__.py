"""PyTorch integration for Spark DataFrames.

This module provides tools for integrating PyTorch with Apache Spark DataFrames.

Main Components:
    - loader: Main function for converting Spark DataFrames to PyTorch DataLoaders
    - SparkArrowBatchDataset: Dataset that streams Arrow RecordBatch objects from Spark
    - RebatchingDataset: Dataset that ensures consistent batch sizes
"""

from caspi.torch.helpers import (
    TensorDict,
    TensorBatch,
)
from caspi.torch.loader import (
    loader,
    SparkArrowBatchDataset,
    RebatchingDataset,
    BatchPrefetchDataset,
)

__all__ = [
    # Main interface
    "loader",
    "SparkArrowBatchDataset",
    "RebatchingDataset",
    "BatchPrefetchDataset",
    "TensorDict",
    "TensorBatch",
]
