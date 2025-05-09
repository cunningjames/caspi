"""PyTorch loader interface for Spark DataFrames."""

from __future__ import annotations

import queue
import threading
import warnings
import weakref
from collections import deque
from collections.abc import Iterator
from typing import Any, Callable

import pyarrow as pa
import torch
from pyspark.sql import DataFrame
from pyspark.sql.functions import rand
from torch.utils.data import DataLoader, IterableDataset

from caspi.torch.helpers import (
    _ARROW_BATCH_SCHEMA,
    TensorDict,
    _arrow_batch_to_tensor_dict,
    _concatenate_tensor_dicts,
    _get_tensor_dict_rows,
    _serialise_batches,
    _slice_tensor_dict,
    _to_device,
    _validate_df_schema,
)

_PREFETCH_SENTINEL = object()


def _worker_fn(self_proxy: BatchPrefetchDataset) -> None:
    """Runs in the background thread. `self_proxy` is a weakref.proxy.
    Any attempt to use it *after* the dataset is GCed raises ReferenceError,
    which we treat as our signal to shut down.
    """
    
    try:
        source_iter = iter(self_proxy.source_dataset)
        while True:
            if self_proxy._stop_event is None or self_proxy._queue is None:
                return
            
            if self_proxy._stop_event.is_set():
                return

            try:
                batch = next(source_iter)
            except StopIteration:
                self_proxy._queue.put(_PREFETCH_SENTINEL)
                return

            while not self_proxy._stop_event.is_set():
                try:
                    self_proxy._queue.put(batch, timeout=1)
                    break
                except queue.Full:
                    continue
    except ReferenceError:
        return
    except Exception as e:
        try:
            if self_proxy._queue is None:
                return

            self_proxy._queue.put(e)
        except ReferenceError:
            pass


class BatchPrefetchDataset(IterableDataset):
    """An IterableDataset that prefetches batches from a source dataset.

    This dataset uses a worker thread to load batches from the `source_dataset`
    into a queue, allowing the main thread to consume batches without waiting
    for I/O operations.

    Attributes:
        source_dataset (IterableDataset[TensorDict]): The dataset from which
            to prefetch batches.
        max_queue_size (int): The maximum number of batches to store in the
            prefetch queue.
        device (str | torch.device | None): The target device for the prefetched
            batches. If None, batches are not moved to a specific device by
            this dataset.
        timeout (int): The maximum time to wait for a batch from the queue.
    """

    def __init__(
        self,
        source_dataset: IterableDataset[TensorDict],
        max_queue_size: int,
        device: str | torch.device | None = None,
        timeout: int = 120,
    ):
        """Initializes the BatchPrefetchDataset.

        Args:
            source_dataset (IterableDataset[TensorDict]): The dataset from which
                to prefetch batches.
            max_queue_size (int): The maximum number of batches to store in the
                prefetch queue. Must be a positive integer.
            device (str | torch.device | None): The target device for the
                prefetched batches. If None, batches are not moved.
            timeout (int): The maximum time to wait for a batch from the queue.
                Default is 120 seconds.

        Raises:
            ValueError: If `max_queue_size` is not positive.
        """
        
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive.")
        self.source_dataset = source_dataset
        self.max_queue_size = max_queue_size
        self.device = device

        self._queue: queue.Queue[Any] | None = None
        self._worker_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._timeout = timeout

    def __iter__(self) -> Iterator[TensorDict]:
        """Returns an iterator for the dataset.

        Initializes the prefetching queue, stop event, and starts the worker
        thread. If a worker thread from a previous iteration is still active,
        it's cleaned up first.

        Returns:
            Iterator[TensorDict]: An iterator yielding prefetched tensor dictionaries.
        """
        
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._cleanup_worker()

        self._queue = queue.Queue(maxsize=self.max_queue_size)
        self._stop_event = threading.Event()
        
        self_proxy = weakref.proxy(self)

        # self._worker_thread = threading.Thread(target=self._worker_fn, daemon=True)
        self._worker_thread = threading.Thread(
            target=_worker_fn,
            args=(self_proxy,),
            daemon=True,
        )
        self._worker_thread.start()

        return self

    def __next__(self) -> TensorDict:
        """Retrieves the next batch from the prefetch queue.

        Blocks until a batch is available or a timeout occurs. Handles the
        sentinel object to signal `StopIteration` and raises any exceptions
        propagated from the worker thread. If a target device is specified,
        the batch is moved to that device before being returned.

        Returns:
            TensorDict: The next batch of data.

        Raises:
            RuntimeError: If the iterator is not initialized (i.e., `__iter__`
                has not been called) or if the worker thread dies unexpectedly.
            TimeoutError: If waiting for a batch from the queue times out.
            StopIteration: When all batches for the current epoch have been consumed.
            Exception: Any exception raised by the `source_dataset` during iteration
                or batch processing.
        """
        
        if self._queue is None:
            raise RuntimeError(
                "Iterator not initialized. Call iter() on BatchPrefetchDataset first."
            )

        try:
            item = self._queue.get(block=True, timeout=self._timeout)
        except queue.Empty:
            if self._worker_thread is not None and not self._worker_thread.is_alive():
                raise RuntimeError(
                    "Prefetch worker thread died or exited unexpectedly."
                )
            else:  # Worker might be slow or stuck
                raise TimeoutError(
                    "Timeout waiting for batch from prefetch queue. "
                    "Worker may be slow, stuck, or an error occurred in the worker."
                )

        if item is _PREFETCH_SENTINEL:
            self._cleanup_worker()  # Normal end of epoch, join thread
            raise StopIteration

        if isinstance(item, Exception):
            self._cleanup_worker()
            raise item

        # If we are here, item is a TensorDict. Move to device if specified.
        if self.device is not None:
            item = _to_device(item, self.device)

        return item  # type: ignore

    def _cleanup_worker(self) -> None:
        """Cleans up the worker thread and associated resources.

        Signals the worker thread to stop, clears the queue, and joins the
        thread with a timeout. Resets internal state related to the worker.
        """
        
        if self._worker_thread is not None:
            if self._stop_event:
                self._stop_event.set()

            if self._queue:
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break

            self._worker_thread.join(timeout=10)
            if self._worker_thread.is_alive():
                warnings.warn(
                    "Prefetch worker thread did not terminate cleanly.",
                    RuntimeWarning,
                )

        self._worker_thread = None
        self._queue = None
        self._stop_event = None

    def __del__(self) -> None:
        """Ensures the worker thread is cleaned up when the dataset is deleted."""
        self._cleanup_worker()


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
        cache: bool = False,
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
            cache (bool): If True, caches the DataFrame in memory. Default is False.
        """

        _validate_df_schema(df, timestamp_to="int64", tokenizer=tokenizer)
        self._df = df
        self._spark_schema = df.schema
        self._tokenizer = tokenizer

        self.shuffle_per_epoch = shuffle_per_epoch
        self.repartition_num = repartition_num
        self.repartition_cols = repartition_cols
        self.random_seed = random_seed
        self._epoch_count = 0
        self.cache = cache
        
        if not shuffle_per_epoch and random_seed is not None:
            self._df = self._df.orderBy(rand(seed=random_seed))

        if cache:
            self._df = self._df.cache()

            if repartition_num is not None or repartition_cols:
                raise ValueError(
                    "Caching the DataFrame and using repartitioning is not "
                    "supported. Please set cache=False or avoid using repartition_num/"
                    "repartition_cols."
                )

    def __iter__(self) -> Iterator[TensorDict]:
        self._epoch_count += 1

        df_for_this_epoch = self._df
        
        if self.shuffle_per_epoch:
            if self.random_seed is not None:
                seed_for_epoch = self.random_seed + self._epoch_count
                df_for_this_epoch = df_for_this_epoch.orderBy(rand(seed=seed_for_epoch))
            else:
                df_for_this_epoch = df_for_this_epoch.orderBy(rand())

        if (
            self.repartition_num is not None
            and self.repartition_cols
            and len(self.repartition_cols) > 0
        ):
            df_for_this_epoch = df_for_this_epoch.repartition(
                self.repartition_num, *self.repartition_cols
            )
        elif self.repartition_num is not None:
            df_for_this_epoch = df_for_this_epoch.repartition(self.repartition_num)
        elif self.repartition_cols and len(self.repartition_cols) > 0:
            df_for_this_epoch = df_for_this_epoch.repartition(*self.repartition_cols)

        schema_str = str(_ARROW_BATCH_SCHEMA)

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

    This dataset consumes batches from a `source_dataset` (which may have
    variable batch sizes) and re-batches them into `TensorDict`s of a
    specified, fixed `batch_size`. It handles cases where source batches
    need to be split or concatenated to form the target batch size.

    Attributes:
        source_dataset (IterableDataset[TensorDict]): The underlying dataset
            from which to draw batches.
        batch_size (int): The desired number of rows in each output batch.
        device (str | torch.device | None): The target device for output batches.
            If None, batches are not moved.
        batch_transform_fn (Callable[[TensorDict], TensorDict] | None): An
            optional function to apply to each re-batched `TensorDict` before
            it is yielded.
    """

    def __init__(
        self,
        source_dataset: IterableDataset[TensorDict],
        batch_size: int,
        device: str | torch.device | None = None,
        batch_transform_fn: Callable[[TensorDict], TensorDict] | None = None,
    ):
        """Initializes the RebatchingDataset.

        Args:
            source_dataset (IterableDataset[TensorDict]): The dataset from which
                to draw and re-batch data.
            batch_size (int): The target number of rows for each output batch.
                Must be a positive integer.
            device (str | torch.device | None): If specified, output batches
                will be moved to this device.
            batch_transform_fn (Callable[[TensorDict], TensorDict] | None):
                An optional function to transform each `TensorDict` after
                re-batching and before device placement.

        Raises:
            ValueError: If `batch_size` is not positive.
        """
        
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.source_dataset = source_dataset
        self.batch_size = batch_size
        self.device = device
        self.batch_transform_fn = batch_transform_fn

    def __iter__(self) -> Iterator[TensorDict]:
        """Returns an iterator that yields re-batched TensorDicts.

        This iterator pulls data from the `source_dataset`, buffers it, and
        constructs new batches of `self.batch_size`. It handles partial batches
        at the end of the `source_dataset` iteration. If a `batch_transform_fn`
        is provided, it's applied to each batch. If a `device` is set, batches
        are moved to that device before being yielded.

        Yields:
            Iterator[TensorDict]: An iterator of `TensorDict`s, each containing
                `batch_size` rows (or fewer for the last batch if the total
                number of rows is not a multiple of `batch_size`).
        """
        
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

            if to_concat:
                output_batch = to_concat[0]
                for piece in to_concat[1:]:
                    output_batch = _concatenate_tensor_dicts(output_batch, piece)
            elif source_exhausted and rows_in_buffer == 0:
                break

            if self.batch_transform_fn:
                output_batch = self.batch_transform_fn(output_batch)

            if self.device is not None and output_batch:
                output_batch = _to_device(output_batch, self.device)

            if output_batch:
                yield output_batch

            if source_exhausted and rows_in_buffer == 0:
                break


def loader(
    df: DataFrame,
    batch_size: int,
    tokenizer: Callable | None = None,
    device: str | torch.device | None = None,
    pin_memory: bool = False,
    shuffle_per_epoch: bool = False,
    repartition_num: int | None = None,
    repartition_cols: list[str] | None = None,
    random_seed: int | None = None,
    batch_transform_fn: Callable[[TensorDict], TensorDict] | None = None,
    cache: bool = False,
    prefetch_queue_size: int | None = None,
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
        batch_transform_fn (Callable[[TensorDict], TensorDict] | None): An optional
            function to transform each batch of data. This function should take
            a `TensorDict` as input and return a transformed `TensorDict`.
        cache (bool): If True, caches the DataFrame in memory. Default is False.

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
        cache=cache,
    )

    rebatching_device = (
        None if (prefetch_queue_size and prefetch_queue_size > 0) else device
    )

    rebatched_dataset = RebatchingDataset(
        spark_dataset,
        batch_size,
        device=rebatching_device,
        batch_transform_fn=batch_transform_fn,
    )

    final_dataset_for_loader: RebatchingDataset | BatchPrefetchDataset = (
        rebatched_dataset
    )
    if prefetch_queue_size is not None and prefetch_queue_size > 0:
        final_dataset_for_loader = BatchPrefetchDataset(
            rebatched_dataset,
            prefetch_queue_size,
            device=device,
        )

    actual_device_obj = torch.device("cpu")
    if device is not None:
        if isinstance(device, str):
            actual_device_obj = torch.device(device)
        else:
            actual_device_obj = device

    return DataLoader(
        final_dataset_for_loader,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory if actual_device_obj.type != "cpu" else False,
    )
