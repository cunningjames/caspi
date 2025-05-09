import pytest
import torch
from torch.utils.data import IterableDataset
import time
from collections.abc import Iterator

from caspi.torch.loader import BatchPrefetchDataset
from caspi.torch.helpers import TensorDict


class DummySourceDataset(IterableDataset[TensorDict]):
    """A simple IterableDataset for testing prefetching.

    Attributes:
        num_batches (int): The total number of batches this dataset will yield.
        item_delay (float): The delay in seconds after yielding each item,
            simulating I/O or computation time.
    """
    def __init__(self, num_batches: int = 5, item_delay: float = 0.01):
        """Initializes the DummySourceDataset.

        Args:
            num_batches (int): The number of batches this dataset will yield.
            item_delay (float): Delay in seconds after yielding each item.
        """
        super().__init__()
        self.num_batches = num_batches
        self.item_delay = item_delay

    def __iter__(self) -> Iterator[TensorDict]:
        """Yields dummy TensorDict batches.

        Yields:
            Iterator[TensorDict]: An iterator of dummy tensor dictionaries.
        """
        for i in range(self.num_batches):
            yield {"data": torch.tensor([i]), "id": torch.tensor([i])}
            if self.item_delay > 0:
                time.sleep(self.item_delay)


def test_batch_prefetch_dataset_worker_cleanup_on_del() -> None:
    """Tests that the worker thread is cleaned up when BatchPrefetchDataset is deleted.

    This test verifies that the `__del__` method of `BatchPrefetchDataset`
    correctly stops and joins the worker thread.
    """
    source_dataset = DummySourceDataset(num_batches=10, item_delay=0.05)
    
    prefetch_ds_instance = BatchPrefetchDataset(
        source_dataset=source_dataset,
        max_queue_size=1,
        device=None,
        timeout=2
    )

    iterator_ref = iter(prefetch_ds_instance)
    assert iterator_ref is prefetch_ds_instance, "Iterator should be the dataset itself"

    assert prefetch_ds_instance._worker_thread is not None, "Worker thread should be initialized."
    assert prefetch_ds_instance._worker_thread.is_alive(), "Worker thread should be alive after iter()."

    try:
        _ = next(iterator_ref)
    except StopIteration:
        pytest.fail("DummySourceDataset should yield at least one item.")
    except TimeoutError:
        pytest.fail("Timeout getting first item, worker might be stuck.")

    worker_thread_obj = prefetch_ds_instance._worker_thread
    assert worker_thread_obj is not None, "Worker thread object reference should be valid."

    del iterator_ref
    del prefetch_ds_instance

    time.sleep(1.0)

    assert not worker_thread_obj.is_alive(), \
        "Worker thread should not be alive after BatchPrefetchDataset object is deleted."


def test_batch_prefetch_dataset_basic_functionality() -> None:
    """Tests basic prefetching functionality of BatchPrefetchDataset.

    Ensures that all batches from the source dataset are yielded correctly
    and in the expected order.
    """
    num_items = 5
    source_dataset = DummySourceDataset(num_batches=num_items, item_delay=0.01)
    prefetch_ds = BatchPrefetchDataset(source_dataset, max_queue_size=2, timeout=5)

    output_batches = []
    # Obtain the iterator from the dataset
    iterator = iter(prefetch_ds)
    for i, batch in enumerate(iterator):
        output_batches.append(batch)
        assert isinstance(batch, dict), "Batch should be a TensorDict"
        assert "data" in batch, "Batch should contain 'data' key"
        assert torch.equal(batch["data"], torch.tensor([i])), \
            f"Batch data incorrect. Expected {torch.tensor([i])}, got {batch['data']}"

    assert len(output_batches) == num_items, \
        f"Expected {num_items} batches, but got {len(output_batches)}"
    
    # Check that StopIteration is raised after all items are consumed
    with pytest.raises(StopIteration):
        next(iterator) # Use the iterator object, not the dataset instance directly


def test_batch_prefetch_dataset_timeout() -> None:
    """Tests the timeout mechanism of BatchPrefetchDataset.

    Verifies that a TimeoutError is raised if a batch is not available
    in the prefetch queue within the specified timeout period. This is simulated
    by a source dataset that has a long delay for its first item.
    """
    # Source dataset that will delay longer than the timeout
    # Use SlowStartSourceDataset to ensure the delay happens before the first item is produced
    source_dataset = SlowStartSourceDataset(initial_delay=0.2, num_batches=1)
    # Timeout is shorter than the item_delay
    prefetch_ds = BatchPrefetchDataset(source_dataset, max_queue_size=1, timeout=0.1)

    with pytest.raises(TimeoutError):
        next(iter(prefetch_ds))
    
    # Ensure worker is cleaned up even after timeout
    worker_thread = prefetch_ds._worker_thread
    del prefetch_ds
    if worker_thread:
        worker_thread.join(timeout=1) # Give some time for cleanup
        assert not worker_thread.is_alive(), "Worker thread should be cleaned up after timeout and deletion."


class SlowStartSourceDataset(IterableDataset[TensorDict]):
    """A source dataset that has an initial delay before yielding any items."""
    def __init__(self, initial_delay: float, num_batches: int = 1):
        self.initial_delay = initial_delay
        self.num_batches = num_batches

    def __iter__(self) -> Iterator[TensorDict]:
        time.sleep(self.initial_delay)
        for i in range(self.num_batches):
            yield {"data": torch.tensor([i])}


def test_batch_prefetch_dataset_timeout_worker_dies() -> None:
    """Tests that RuntimeError is raised if worker dies and queue becomes empty."""

    class CrashingSourceDataset(IterableDataset[TensorDict]):
        def __iter__(self) -> Iterator[TensorDict]:
            yield {"data": torch.tensor([0])} # Yield one item
            raise RuntimeError("Simulated crash in source dataset")

    source_dataset = CrashingSourceDataset()
    # Short timeout to quickly detect queue empty after worker dies
    prefetch_ds = BatchPrefetchDataset(source_dataset, max_queue_size=1, timeout=0.5)

    iterator = iter(prefetch_ds)
    
    # Get the first item successfully
    try:
        _ = next(iterator)
    except Exception as e:
        pytest.fail(f"Should have received the first item, but got {e}")

    # The worker thread should die after the first item due to the crash
    # Wait a bit for the worker to process and die
    time.sleep(0.2) 
    
    assert prefetch_ds._worker_thread is not None
    assert not prefetch_ds._worker_thread.is_alive(), "Worker thread should have died."

    # The error raised will be the one from the CrashingSourceDataset,
    # as the BatchPrefetchDataset propagates exceptions from the worker.
    with pytest.raises(RuntimeError, match="Simulated crash in source dataset"):
        next(iterator) # This should now raise RuntimeError as queue is empty and worker is dead


def test_batch_prefetch_dataset_max_queue_size() -> None:
    """Tests that the prefetch queue does not exceed its maximum size.

    This test uses a source dataset that produces items quickly and checks
    the internal queue size of the BatchPrefetchDataset.
    """
    max_q_size = 2
    # Source dataset produces items faster than they are consumed.
    # item_delay is 0, so it will try to fill the queue quickly.
    source_dataset = DummySourceDataset(num_batches=10, item_delay=0) 
    prefetch_ds = BatchPrefetchDataset(source_dataset, max_queue_size=max_q_size, timeout=1)

    iterator = iter(prefetch_ds)
    
    # Let the worker thread run for a bit to populate the queue.
    # The time.sleep here should be long enough for the worker to fill the queue
    # up to max_queue_size, but not so long that we start consuming items
    # before checking the queue size.
    time.sleep(0.1) 

    assert prefetch_ds._queue is not None, "Queue should be initialized."
    # The queue size should not exceed max_q_size.
    # It might be less if the worker hasn't filled it completely yet,
    # or if consumption has started, but it should never be more.
    # In this setup, with item_delay=0, it's very likely to hit max_q_size quickly.
    current_queue_size = prefetch_ds._queue.qsize()
    assert current_queue_size <= max_q_size, \
        f"Queue size {current_queue_size} exceeded max_queue_size {max_q_size}."

    # Consume one item to make space
    try:
        _ = next(iterator)
    except (StopIteration, TimeoutError) as e:
        pytest.fail(f"Failed to get an item from prefetch_ds: {e}")
    
    # Give worker a chance to fill the spot
    time.sleep(0.1)
    
    current_queue_size_after_consumption = prefetch_ds._queue.qsize()
    assert current_queue_size_after_consumption <= max_q_size, \
        f"Queue size {current_queue_size_after_consumption} exceeded max_queue_size {max_q_size} after consumption."

    # Consume all items to allow graceful thread shutdown
    for _ in iterator:
        pass
    
    # Ensure worker is cleaned up
    worker_thread = prefetch_ds._worker_thread
    del prefetch_ds
    if worker_thread:
        worker_thread.join(timeout=1)
        assert not worker_thread.is_alive(), "Worker thread should be cleaned up."
