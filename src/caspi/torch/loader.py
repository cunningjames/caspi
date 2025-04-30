"""PyTorch loader interface for Spark DataFrames."""
from collections import deque
from collections.abc import Iterator
from typing import Callable

import pyarrow as pa

from pyspark.sql import DataFrame
from torch.utils.data import DataLoader, IterableDataset

from caspi.torch.helpers import (
    TensorDict,
    _arrow_batch_to_tensor_dict,
    _concatenate_tensor_dicts,
    _get_tensor_dict_rows,
    _serialise_batches,
    _slice_tensor_dict,
    _validate_df_schema,
    _ARROW_BATCH_SCHEMA,
)


class SparkArrowBatchDataset(IterableDataset):
    """Streams Arrow `RecordBatch` objects from Spark to the driver via only
    public PySpark APIs, then converts them to `TensorDict`s.
    """

    def __init__(self, df: DataFrame, tokenizer: Callable | None = None):
        """Initialize a SparkArrowBatchDataset.
        
        Args:
            df: Source PySpark `DataFrame`.
            tokenizer: Optional tokenizer for StringType columns.
        """

        _validate_df_schema(df, timestamp_to="int64", tokenizer=tokenizer)
        self._df = df
        self._spark_schema = df.schema
        self._tokenizer = tokenizer

    def __iter__(self) -> Iterator[TensorDict]:
        # Convert PyArrow schema to string representation for Spark
        schema_str = str(_ARROW_BATCH_SCHEMA)
        df_serialised = self._df.mapInArrow(
            _serialise_batches, schema_str
        )

        for row in df_serialised.toLocalIterator():
            payload = row["batch"]
            stream = pa.ipc.open_stream(payload)
            for record_batch in stream:
                yield _arrow_batch_to_tensor_dict(
                    record_batch, self._spark_schema, self._tokenizer
                )


class RebatchingDataset(IterableDataset):
    """Wraps an IterableDataset yielding TensorDicts to enforce a fixed batch size.

    Pulls data from the source dataset, concatenates it, and yields
    batches of the desired size. Handles the final partial batch.
    """

    def __init__(self, source_dataset: IterableDataset[TensorDict], batch_size: int):
        """Initialize the RebatchingDataset.

        Args:
            source_dataset: The underlying dataset yielding TensorDict batches
                (e.g., SparkArrowBatchDataset).
            batch_size: The desired fixed batch size to yield.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.source_dataset = source_dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[TensorDict]:
        """Iterate through the source dataset and yield fixed‑size batches.

        The old implementation concatenated the entire buffer every time a new
        batch arrived, causing O(N²) data movement.  This version keeps a deque
        of incoming TensorDicts (`buffer_batches`) and splices only the rows
        required for the next output batch, concatenating *once per yield*.
        """
        source_iter = iter(self.source_dataset)
        buffer_batches: deque[TensorDict] = deque()
        rows_in_buffer = 0
        source_exhausted = False

        while True:
            # -------------------------------------------------------------- #
            # Stage 1: TOP‑UP — keep pulling from the source until we have
            #           enough rows to satisfy one full batch or the source
            #           is exhausted.
            # -------------------------------------------------------------- #
            while rows_in_buffer < self.batch_size and not source_exhausted:
                try:
                    next_batch = next(source_iter)
                    if not next_batch:
                        continue  # skip empty batches
                    buffer_batches.append(next_batch)
                    rows_in_buffer += _get_tensor_dict_rows(next_batch)
                except StopIteration:
                    source_exhausted = True
                    break

            # If we have no data left, exit
            if rows_in_buffer == 0 and source_exhausted:
                break

            # Stage 2: BUILD one output batch of size `yield_size`
            yield_size = min(self.batch_size, rows_in_buffer)
            rows_needed = yield_size
            to_concat: list[TensorDict] = []

            # Consume from the front of buffer_batches until we've collected
            # the requested number of rows.
            while rows_needed > 0 and buffer_batches:
                head = buffer_batches[0]
                head_rows = _get_tensor_dict_rows(head)

                if head_rows <= rows_needed:
                    # Use the whole head batch
                    to_concat.append(head)
                    buffer_batches.popleft()
                    rows_in_buffer -= head_rows
                    rows_needed -= head_rows
                else:
                    # Use a slice of the head batch
                    to_concat.append(_slice_tensor_dict(head, 0, rows_needed))
                    # Replace head with its remainder (in‑place update)
                    buffer_batches[0] = _slice_tensor_dict(head, rows_needed, head_rows)
                    rows_in_buffer -= rows_needed
                    rows_needed = 0

            # Concatenate the pieces *once* and yield
            output_batch: TensorDict = {}
            for piece in to_concat:
                output_batch = _concatenate_tensor_dicts(output_batch, piece)

            yield output_batch

            # Terminate when no more data remains
            if source_exhausted and rows_in_buffer == 0:
                break


def loader(
    df: DataFrame,
    batch_size: int,
    pin_memory_device: str = "",
    tokenizer: Callable | None = None,
) -> DataLoader[TensorDict]:
    """Create a Torch DataLoader from a PySpark DataFrame using `mapInArrow`.

    This version ensures that the DataLoader yields batches of the specified
    `batch_size`, handling partial batches at the end.

    Args:
        df: PySpark DataFrame to load.
        batch_size: The desired size for batches yielded by the DataLoader.
        pin_memory_device: Device string for DataLoader's pin_memory.
        tokenizer: Optional tokenizer for StringType columns.

    Returns:
        DataLoader yielding dicts mapping column names to torch.Tensor or
        list[torch.Tensor] batches processed distributively and then re-batched.
    """

    spark_dataset = SparkArrowBatchDataset(df, tokenizer=tokenizer)
    rebatched_dataset = RebatchingDataset(spark_dataset, batch_size)

    # batch_size=None for the DataLoader because RebatchingDataset handles batching.
    # num_workers=0 is essential, parallelism is handled by Spark.
    return DataLoader(
        rebatched_dataset,
        batch_size=None,  # Batching is done by RebatchingDataset
        shuffle=False,  # Shuffling is complex with distributed source
        num_workers=0,  # Parallelism handled by Spark
        pin_memory_device=pin_memory_device,
    )
