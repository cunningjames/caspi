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
    """Streams Arrow `RecordBatch` objects from Spark to the driver.

    Uses only public PySpark APIs (`mapInArrow`) to stream data efficiently
    and then converts the received Arrow batches into `TensorDict`s suitable
    for PyTorch models.

    Attributes:
        _df: The source PySpark DataFrame.
        _spark_schema: The schema of the source DataFrame.
        _tokenizer: An optional callable used to tokenize string columns.
    """

    def __init__(self, df: DataFrame, tokenizer: Callable | None = None) -> None:
        """Initializes the SparkArrowBatchDataset.

        Args:
            df (DataFrame): The source PySpark `DataFrame`.
            tokenizer (Callable | None): An optional tokenizer for StringType
                columns. If provided, string columns will be tokenized and
                represented as tensors based on the tokenizer's output.
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
    """Creates a PyTorch DataLoader from a PySpark DataFrame.

    This function sets up a data loading pipeline that:
    1. Uses `SparkArrowBatchDataset` to efficiently stream data from Spark
       executors to the driver using `mapInArrow`.
    2. Wraps the stream with `RebatchingDataset` to ensure that the data is
       yielded in batches of the specified `batch_size`.
    3. Configures a `torch.utils.data.DataLoader` to manage the final data
       stream, ready for consumption by a PyTorch model.

    Args:
        df (DataFrame): The PySpark DataFrame to load data from.
        batch_size (int): The desired number of rows in each batch yielded by
            the DataLoader.
        pin_memory_device (str): The device string (e.g., "cuda:0") to use for
            `pin_memory` in the DataLoader. Defaults to "" (no pinning).
        tokenizer (Callable | None): An optional tokenizer function to process
            StringType columns in the DataFrame. If provided, string columns
            will be tokenized.

    Returns:
        DataLoader[TensorDict]: A PyTorch DataLoader instance. It yields
            dictionaries (`TensorDict`) where keys are column names (or derived
            names for tokenized strings) and values are `torch.Tensor` or
            `list[torch.Tensor]`. The batches produced will have `batch_size`
            rows, except possibly the last one. Parallelism is handled by Spark,
            so `num_workers` is set to 0. Shuffling is disabled due to the
            distributed nature of the data source.
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
