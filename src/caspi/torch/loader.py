"""PyTorch loader interface for Spark DataFrames."""

from collections import deque
from collections.abc import Iterator
from typing import Callable

import pyarrow as pa
import torch

from pyspark.sql import DataFrame
from pyspark.sql.functions import rand
from torch.utils.data import DataLoader, IterableDataset

from caspi.torch.helpers import (
    TensorDict,
    _arrow_batch_to_tensor_dict,
    _concatenate_tensor_dicts,
    _get_tensor_dict_rows,
    _serialise_batches,
    _slice_tensor_dict,
    _to_device,
    _validate_df_schema,
    _ARROW_BATCH_SCHEMA,
)


class SparkArrowBatchDataset(IterableDataset):
    """Streams Arrow `RecordBatch` objects from Spark to the driver.

    Uses only public PySpark APIs (`mapInArrow`) to stream data efficiently
    and then converts the received Arrow batches into `TensorDict`s suitable
    for PyTorch models. It can optionally shuffle and/or repartition the
    DataFrame at the beginning of each epoch.

    Attributes:
        _df: The source PySpark DataFrame.
        _spark_schema: The schema of the source DataFrame.
        _tokenizer: An optional callable used to tokenize string columns.
        shuffle_per_epoch (bool): Whether to shuffle the DataFrame before each epoch.
        repartition_num (int | None): Number of partitions for repartitioning
            before each epoch.
        repartition_cols (list[str] | None): Columns to use for repartitioning
            before each epoch.
        random_seed (int | None): Seed for shuffling for reproducibility. If set,
            the shuffle order will be the same for every epoch. If None and
            shuffle_per_epoch is True, shuffle order will be random each epoch.
    """

    def __init__(
        self, 
        df: DataFrame, 
        tokenizer: Callable | None = None,
        shuffle_per_epoch: bool = False,
        repartition_num: int | None = None,
        repartition_cols: list[str] | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Initializes the SparkArrowBatchDataset.

        Args:
            df (DataFrame): The source PySpark `DataFrame`.
            tokenizer (Callable | None): An optional tokenizer for StringType
                columns.
            shuffle_per_epoch (bool): If True, the DataFrame will be globally
                shuffled before data is streamed for each epoch. Default is False.
            repartition_num (int | None): If provided, the DataFrame will be
                repartitioned to this many partitions before each epoch.
            repartition_cols (list[str] | None): If provided, the DataFrame will
                be repartitioned using these columns before each epoch. Can be
                used with or without `repartition_num`.
            random_seed (int | None): Seed for the random number generator used
                in shuffling. If `shuffle_per_epoch` is True and `random_seed` is
                set, the shuffle order will be identical across epochs. If None,
                the shuffle will be different each epoch.
        """

        _validate_df_schema(df, timestamp_to="int64", tokenizer=tokenizer)
        self._df = df  # Store the original DataFrame
        self._spark_schema = df.schema
        self._tokenizer = tokenizer
        
        self.shuffle_per_epoch = shuffle_per_epoch
        self.repartition_num = repartition_num
        self.repartition_cols = repartition_cols
        self.random_seed = random_seed
        # self._epoch_count = 0 # Optional: for more complex seed logic if needed

    def __iter__(self) -> Iterator[TensorDict]:
        # This method is called by DataLoader at the start of each epoch.
        # self._epoch_count += 1 # If varying seed based on epoch number

        # Start with the original DataFrame for this epoch's operations
        df_for_this_epoch = self._df

        # 1. Apply per-epoch shuffling if configured
        if self.shuffle_per_epoch:
            # seed_for_epoch = self.random_seed
            # if self.random_seed is not None and self._epoch_count > 1:
            #     # Example of varying seed: makes shuffle different but reproducible
            #     seed_for_epoch = self.random_seed + self._epoch_count -1 
            
            if self.random_seed is not None:
                df_for_this_epoch = df_for_this_epoch.orderBy(rand(seed=self.random_seed))
            else:
                df_for_this_epoch = df_for_this_epoch.orderBy(rand())

        # 2. Apply per-epoch repartitioning if configured
        if self.repartition_num is not None and self.repartition_cols and len(self.repartition_cols) > 0:
            df_for_this_epoch = df_for_this_epoch.repartition(self.repartition_num, *self.repartition_cols)
        elif self.repartition_num is not None:
            df_for_this_epoch = df_for_this_epoch.repartition(self.repartition_num)
        elif self.repartition_cols and len(self.repartition_cols) > 0:
            df_for_this_epoch = df_for_this_epoch.repartition(*self.repartition_cols)
        
        # Convert PyArrow schema to string representation for Spark
        schema_str = str(_ARROW_BATCH_SCHEMA)
        # Use the potentially shuffled/repartitioned DataFrame for mapInArrow
        df_serialised = df_for_this_epoch.mapInArrow(_serialise_batches, schema_str)

        for row in df_serialised.toLocalIterator():
            payload = row["batch"]
            stream = pa.ipc.open_stream(payload)
            for record_batch in stream:
                yield _arrow_batch_to_tensor_dict(
                    record_batch, self._spark_schema, self._tokenizer
                )


class RebatchingDataset(IterableDataset):
    """Wraps an IterableDataset yielding TensorDicts to enforce a fixed batch size.
    (No changes needed in this class for enhancement 1)
    """

    def __init__(
        self, 
        source_dataset: IterableDataset[TensorDict], 
        batch_size: int,
        device: str | torch.device | None = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.source_dataset = source_dataset
        self.batch_size = batch_size
        self.device = device

    def __iter__(self) -> Iterator[TensorDict]:
        source_iter = iter(self.source_dataset)
        buffer_batches: deque[TensorDict] = deque()
        rows_in_buffer = 0
        source_exhausted = False

        while True:
            while rows_in_buffer < self.batch_size and not source_exhausted:
                try:
                    next_batch = next(source_iter)
                    if not next_batch:
                        continue
                    buffer_batches.append(next_batch)
                    rows_in_buffer += _get_tensor_dict_rows(next_batch)
                except StopIteration:
                    source_exhausted = True
                    break

            if rows_in_buffer == 0 and source_exhausted:
                break

            yield_size = min(self.batch_size, rows_in_buffer)
            rows_needed = yield_size
            to_concat: list[TensorDict] = []

            while rows_needed > 0 and buffer_batches:
                head = buffer_batches[0]
                head_rows = _get_tensor_dict_rows(head)

                if head_rows <= rows_needed:
                    to_concat.append(head)
                    buffer_batches.popleft()
                    rows_in_buffer -= head_rows
                    rows_needed -= head_rows
                else:
                    to_concat.append(_slice_tensor_dict(head, 0, rows_needed))
                    buffer_batches[0] = _slice_tensor_dict(head, rows_needed, head_rows)
                    rows_in_buffer -= rows_needed
                    rows_needed = 0
            
            output_batch: TensorDict = {}
            # Ensure to_concat is not empty before trying to concatenate
            if to_concat:
                # Initialize output_batch with the first item to ensure correct type handling
                output_batch = to_concat[0]
                for piece in to_concat[1:]:
                    output_batch = _concatenate_tensor_dicts(output_batch, piece)
            elif source_exhausted and rows_in_buffer == 0 : # No data to form a batch, and source is done
                break


            if self.device is not None and output_batch: # Check output_batch is not empty
                output_batch = _to_device(output_batch, self.device)
            
            if output_batch: # Only yield if the batch is not empty
                yield output_batch

            if source_exhausted and rows_in_buffer == 0:
                break


def loader(
    df: DataFrame,
    batch_size: int,
    tokenizer: Callable | None = None,
    device: str | torch.device | None = None,
    pin_memory: bool = False,
    # --- New parameters for epoch management ---
    shuffle_per_epoch: bool = False,
    repartition_num: int | None = None,
    repartition_cols: list[str] | None = None,
    random_seed: int | None = None,
) -> DataLoader[TensorDict]:
    """Creates a PyTorch DataLoader from a PySpark DataFrame.

    This function sets up a data loading pipeline that:
    1. Uses `SparkArrowBatchDataset` to efficiently stream data from Spark
       executors to the driver using `mapInArrow`. This dataset can now
       handle per-epoch shuffling and repartitioning of the Spark DataFrame.
    2. Wraps the stream with `RebatchingDataset` to ensure that the data is
       yielded in batches of the specified `batch_size`.
    3. Configures a `torch.utils.data.DataLoader` to manage the final data
       stream.

    Args:
        df (DataFrame): The PySpark DataFrame to load data from.
        batch_size (int): The desired number of rows in each batch yielded by
            the DataLoader.
        tokenizer (Callable | None): An optional tokenizer for StringType columns.
        device (str | torch.device | None): Optional device for output tensors.
        pin_memory (bool): If True, enables pin_memory for the DataLoader.
        shuffle_per_epoch (bool): If True, the Spark DataFrame will be globally
            shuffled before each epoch. Default is False.
        repartition_num (Optional[int]): Number of partitions for repartitioning
            the Spark DataFrame before each epoch.
        repartition_cols (Optional[List[str]]): Columns to use for repartitioning
            the Spark DataFrame before each epoch.
        random_seed (Optional[int]): Seed for shuffling. If set with
            `shuffle_per_epoch=True`, the shuffle order is the same each epoch.
            If None, shuffle is random per epoch.

    Returns:
        DataLoader[TensorDict]: A PyTorch DataLoader instance.
    """
    
    spark_dataset = SparkArrowBatchDataset(
        df, 
        tokenizer=tokenizer,
        shuffle_per_epoch=shuffle_per_epoch,
        repartition_num=repartition_num,
        repartition_cols=repartition_cols,
        random_seed=random_seed,
    )
    rebatched_dataset = RebatchingDataset(spark_dataset, batch_size, device=device)
    
    current_device = torch.device("cpu") # Default device if not specified
    if device is not None:
        if isinstance(device, str): 
            current_device = torch.device(device)
        else:
            current_device = device


    return DataLoader(
        rebatched_dataset,
        batch_size=None,
        shuffle=False,  # Shuffling is handled by SparkArrowBatchDataset at the Spark level
        num_workers=0,
        pin_memory=pin_memory if current_device.type != "cpu" else False,
    )