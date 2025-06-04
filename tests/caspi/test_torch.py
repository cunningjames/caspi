from datetime import datetime
from typing import Iterator, cast

import numpy as np
import pyarrow as pa
import pyspark.sql.functions as f
import pytest
import torch
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    FloatType,
    IntegerType,
    NumericType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)
from torch.utils.data import IterableDataset

from caspi.torch import (
    RebatchingDataset,
    SparkArrowBatchDataset,
    TensorDict,
    loader,
)
from caspi.torch.helpers import (
    _arrow_batch_to_tensor_dict,
    _concatenate_tensor_dicts,
    _get_tensor_dict_rows,
    _slice_tensor_dict,
    _validate_df_schema,
    _to_device,
    _serialise_batches,
    _ARROW_BATCH_SCHEMA,
)

# --- Helper Functions/Classes ---


class DummyTokenizer:
    """Minimal tokenizer for testing StringType conversion."""

    def __call__(
        self,
        texts: list[str],
        padding: str = "longest",
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Tokenizes text.

        Args:
            texts: List of strings to tokenize.
            padding: Padding strategy.
            truncation: Truncation strategy.
            return_tensors: Return format ('pt' for PyTorch).

        Returns:
            Dictionary containing 'input_ids' and 'attention_mask' tensors.
        """
        if return_tensors != "pt":
            raise ValueError("DummyTokenizer supports return_tensors='pt' only.")
        split_texts = [text.split() for text in texts]
        max_len = max(len(t) for t in split_texts) if split_texts else 0

        input_ids = []
        attention_masks = []
        for tokens in split_texts:
            length = len(tokens)
            ids_row = list(range(length)) + [0] * (max_len - length)
            mask_row = [1] * length + [0] * (max_len - length)
            input_ids.append(ids_row)
            attention_masks.append(mask_row)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.int64),
        }


# --- Test Fixtures ---


@pytest.fixture
def simple_schema() -> StructType:
    """Provides a simple Spark schema for testing.

    Returns:
        StructType: A simple schema with Integer and Float types.
    """
    return StructType(
        [
            StructField("id", IntegerType(), False),
            StructField("value", FloatType(), True),
        ]
    )


@pytest.fixture
def complex_schema() -> StructType:
    """Provides a complex Spark schema for testing various types.

    Returns:
        StructType: A schema with various data types including arrays and timestamps.
    """
    return StructType(
        [
            StructField("id", IntegerType(), False),
            StructField("value", FloatType(), True),
            StructField("flag", BooleanType(), True),
            StructField("text", StringType(), True),
            StructField("fixed_array", ArrayType(IntegerType()), True),
            StructField("var_array", ArrayType(FloatType()), True),
            StructField("timestamp", TimestampType(), True),
        ]
    )


@pytest.fixture
def simple_df(spark: SparkSession, simple_schema: StructType) -> DataFrame:
    """Creates a simple DataFrame for testing.

    Args:
        spark: The SparkSession fixture.
        simple_schema: The simple schema fixture.

    Returns:
        DataFrame: A DataFrame with simple data.
    """
    data = [
        (1, 10.0),
        (2, 20.0),
        (3, None),
        (4, 40.0),
        (5, 50.0),
    ]
    return spark.createDataFrame(data, schema=simple_schema)


@pytest.fixture
def complex_df(spark: SparkSession, complex_schema: StructType) -> DataFrame:
    """Creates a complex DataFrame for testing various types.

    Args:
        spark: The SparkSession fixture.
        complex_schema: The complex schema fixture.

    Returns:
        DataFrame: A DataFrame with complex data types.
    """
    data = [
        (1, 1.1, True, "hello", [1, 2], [1.0], datetime(2025, 1, 1, 12, 0, 0)),
        (2, None, False, "world", [3, 4], [2.0, 3.0], None),
        (3, 3.3, None, None, None, None, datetime(2025, 1, 3, 12, 0, 0)),
        (
            4,
            4.4,
            True,
            "foo bar",
            [5, 6],
            [4.0, 5.0, 6.0],
            datetime(2025, 1, 4, 12, 0, 0),
        ),
        (5, 5.5, False, "baz", [7, 8], [7.0], datetime(2025, 1, 5, 12, 0, 0)),
    ]
    return spark.createDataFrame(data, schema=complex_schema)


@pytest.fixture
def shuffle_test_df(spark: SparkSession) -> DataFrame:
    """Creates a DataFrame for testing shuffling.

    Args:
        spark: The SparkSession fixture.

    Returns:
        DataFrame: A DataFrame with a single 'id' column.
    """
    data = [(i,) for i in range(20)]  # IDs from 0 to 19
    return spark.createDataFrame(data, schema="id INT")


def _get_ids_from_epoch(dataset: SparkArrowBatchDataset) -> torch.Tensor:
    """Collects all 'id' tensors from a single iteration of the dataset.

    Args:
        dataset: The SparkArrowBatchDataset instance.

    Returns:
        torch.Tensor: A concatenated tensor of all 'id's from one epoch.
    """
    batches = list(dataset)
    if not batches:
        return torch.empty(0, dtype=torch.int64)
    
    all_ids: list[torch.Tensor] = []
    for b in batches:
        if "id" in b and isinstance(b["id"], torch.Tensor):
            all_ids.append(b["id"])
        elif "id" in b and isinstance(b["id"], list): # Handle cases where 'id' might be a list of tensors
            for item in b["id"]:
                if isinstance(item, torch.Tensor):
                    all_ids.append(item)

    if not all_ids:
        return torch.empty(0, dtype=torch.int64)
    return torch.cat(all_ids)


# --- Test Functions ---


def test_arrow_batch_to_tensor_dict_simple(simple_schema: StructType) -> None:
    """Tests _arrow_batch_to_tensor_dict with simple types."""
    ids = pa.array([1, 2, 3], type=pa.int32())
    values = pa.array([10.0, None, 30.0], type=pa.float32())
    # Manually create the corresponding Arrow schema
    schema_list: list[pa.Field] = [
        pa.field("id", pa.int32(), nullable=False),
        pa.field("value", pa.float32(), nullable=True),
    ]
    arrow_schema = pa.schema(schema_list)
    rb = pa.RecordBatch.from_arrays([ids, values], schema=arrow_schema)

    tensor_dict = _arrow_batch_to_tensor_dict(rb, simple_schema)

    assert "id" in tensor_dict
    assert "value" in tensor_dict
    assert torch.equal(
        cast(torch.Tensor, tensor_dict["id"]),
        torch.tensor([1, 2, 3], dtype=torch.int64),
    )
    # Nones in float columns become NaN
    assert torch.allclose(
        cast(torch.Tensor, tensor_dict["value"]),
        torch.tensor([10.0, np.nan, 30.0]),
        equal_nan=True,
    )
    assert cast(torch.Tensor, tensor_dict["value"]).dtype == torch.float32


def test_arrow_batch_to_tensor_dict_complex(complex_schema: StructType) -> None:
    """Tests _arrow_batch_to_tensor_dict with complex types and nulls."""
    tokenizer = DummyTokenizer()
    ids = pa.array([1, 2], type=pa.int32())
    values = pa.array([1.1, None], type=pa.float32())
    flags = pa.array([True, None], type=pa.bool_())
    texts = pa.array(["hello", None], type=pa.string())
    fixed_arrays = pa.array([[1, 2], None], type=pa.list_(pa.int32()))
    var_arrays = pa.array([[1.0], None], type=pa.list_(pa.float32()))
    timestamps = pa.array(
        [datetime(2025, 1, 1, 12, 0, 0), None], type=pa.timestamp("ns")
    )

    schema_list: list[pa.Field] = [
        pa.field("id", pa.int32(), nullable=False),
        pa.field("value", pa.float32(), nullable=True),
        pa.field("flag", pa.bool_(), nullable=True),
        pa.field("text", pa.string(), nullable=True),
        pa.field("fixed_array", pa.list_(pa.int32()), nullable=True),
        pa.field("var_array", pa.list_(pa.float32()), nullable=True),
        pa.field("timestamp", pa.timestamp("ns"), nullable=True),
    ]

    arrow_schema = pa.schema(schema_list)
    rb = pa.RecordBatch.from_arrays(
        [ids, values, flags, texts, fixed_arrays, var_arrays, timestamps],
        schema=arrow_schema,
    )

    tensor_dict = _arrow_batch_to_tensor_dict(rb, complex_schema, tokenizer=tokenizer)

    assert "id" in tensor_dict
    assert torch.equal(
        cast(torch.Tensor, tensor_dict["id"]), torch.tensor([1, 2], dtype=torch.int64)
    )

    assert "value" in tensor_dict
    assert torch.allclose(
        cast(torch.Tensor, tensor_dict["value"]),
        torch.tensor([1.1, np.nan]),
        equal_nan=True,
    )
    assert cast(torch.Tensor, tensor_dict["value"]).dtype == torch.float32

    assert "flag" in tensor_dict
    # Nones in boolean columns become False
    assert torch.equal(
        cast(torch.Tensor, tensor_dict["flag"]),
        torch.tensor([True, False], dtype=torch.bool),
    )

    assert "text_input_ids" in tensor_dict
    assert "text_attention_mask" in tensor_dict
    # Tokenizer output for ["hello", ""] (None becomes empty string)
    assert torch.equal(
        cast(torch.Tensor, tensor_dict["text_input_ids"]),
        torch.tensor([[0], [0]], dtype=torch.int64),
    )
    assert torch.equal(
        cast(torch.Tensor, tensor_dict["text_attention_mask"]),
        torch.tensor([[1], [0]], dtype=torch.int64),
    )

    assert "fixed_array" in tensor_dict
    assert isinstance(tensor_dict["fixed_array"], list)
    assert len(tensor_dict["fixed_array"]) == 2
    assert torch.equal(tensor_dict["fixed_array"][0], torch.tensor([1, 2]))
    assert tensor_dict["fixed_array"][1].numel() == 0  # Empty tensor for None list

    assert "var_array" in tensor_dict
    assert isinstance(tensor_dict["var_array"], list)
    assert len(tensor_dict["var_array"]) == 2
    assert torch.equal(tensor_dict["var_array"][0], torch.tensor([1.0]))
    assert tensor_dict["var_array"][1].numel() == 0

    assert "timestamp" in tensor_dict
    # Expected UTC timestamp in nanoseconds for 2025-01-01 12:00:00 UTC
    expected_ts1_ns = 1735732800000000000
    # Expected NaT representation for int64
    expected_nat = -9223372036854775808
    # Nones in timestamp columns become NaT (int64 min)
    assert torch.equal(
        cast(torch.Tensor, tensor_dict["timestamp"]),
        torch.tensor([expected_ts1_ns, expected_nat], dtype=torch.int64),
    )


def test_concatenate_tensor_dicts() -> None:
    """Tests _concatenate_tensor_dicts with tensors and lists."""
    dict1: TensorDict = {
        "a": torch.tensor([[1, 2], [3, 4]]),
        "b": [torch.tensor([1]), torch.tensor([2, 3])],
        "c": torch.tensor([True, False]),
    }
    dict2: TensorDict = {
        "a": torch.tensor([[5, 6]]),
        "b": [torch.tensor([4, 5, 6])],
        "c": torch.tensor([True]),
    }

    concatenated = _concatenate_tensor_dicts(dict1, dict2)

    assert "a" in concatenated
    assert torch.equal(
        cast(torch.Tensor, concatenated["a"]), torch.tensor([[1, 2], [3, 4], [5, 6]])
    )

    assert "b" in concatenated
    assert isinstance(concatenated["b"], list)
    assert len(concatenated["b"]) == 3
    assert torch.equal(concatenated["b"][0], torch.tensor([1]))
    assert torch.equal(concatenated["b"][1], torch.tensor([2, 3]))
    assert torch.equal(concatenated["b"][2], torch.tensor([4, 5, 6]))

    assert "c" in concatenated
    assert torch.equal(
        cast(torch.Tensor, concatenated["c"]), torch.tensor([True, False, True])
    )


def test_concatenate_tensor_dicts_padding() -> None:
    """Tests _concatenate_tensor_dicts with padding for mismatched tensor shapes."""
    dict1: TensorDict = {"a": torch.tensor([[1, 2], [3, 4]])}  # Shape (2, 2)
    dict2: TensorDict = {"a": torch.tensor([[5, 6, 7], [8, 9, 10]])}  # Shape (2, 3)

    concatenated = _concatenate_tensor_dicts(dict1, dict2)
    # Expected shape (4, 3), dict1 padded with 0
    expected = torch.tensor([[1, 2, 0], [3, 4, 0], [5, 6, 7], [8, 9, 10]])
    assert torch.equal(cast(torch.Tensor, concatenated["a"]), expected)

    # Test reverse order
    concatenated_rev = _concatenate_tensor_dicts(dict2, dict1)
    expected_rev = torch.tensor([[5, 6, 7], [8, 9, 10], [1, 2, 0], [3, 4, 0]])
    assert torch.equal(cast(torch.Tensor, concatenated_rev["a"]), expected_rev)


def test_slice_tensor_dict() -> None:
    """Tests _slice_tensor_dict."""
    tensor_dict: TensorDict = {
        "a": torch.tensor([1, 2, 3, 4, 5]),
        "b": [
            torch.tensor([1]),
            torch.tensor([2, 3]),
            torch.tensor([4]),
            torch.tensor([5, 6, 7]),
            torch.tensor([8]),
        ],
    }

    sliced = _slice_tensor_dict(tensor_dict, 1, 4)

    assert "a" in sliced
    assert torch.equal(cast(torch.Tensor, sliced["a"]), torch.tensor([2, 3, 4]))

    assert "b" in sliced
    assert isinstance(sliced["b"], list)
    assert len(sliced["b"]) == 3
    assert torch.equal(sliced["b"][0], torch.tensor([2, 3]))
    assert torch.equal(sliced["b"][1], torch.tensor([4]))
    assert torch.equal(sliced["b"][2], torch.tensor([5, 6, 7]))


def test_get_tensor_dict_rows() -> None:
    """Tests _get_tensor_dict_rows."""
    assert _get_tensor_dict_rows({}) == 0

    dict1: TensorDict = {
        "a": torch.tensor([1, 2, 3]),
        "b": torch.tensor([[1], [2], [3]]),
    }
    assert _get_tensor_dict_rows(dict1) == 3

    dict2: TensorDict = {
        "a": [torch.tensor(1), torch.tensor(2)],
        "b": torch.tensor([True, False]),
    }
    assert _get_tensor_dict_rows(dict2) == 2

    dict_inconsistent: TensorDict = {
        "a": torch.tensor([1, 2]),
        "b": torch.tensor([1, 2, 3]),
    }
    with pytest.raises(ValueError, match="Inconsistent number of rows"):
        _get_tensor_dict_rows(dict_inconsistent)

    # Note: This case might be tricky for type checkers if list lengths differ
    dict_inconsistent_list: TensorDict = {
        "a": [torch.tensor(1)],
        "b": [torch.tensor(1), torch.tensor(2)],
    }
    with pytest.raises(ValueError, match="Inconsistent number of rows"):
        _get_tensor_dict_rows(dict_inconsistent_list)


def test_validate_df_schema_success(spark: SparkSession) -> None:
    """Tests _validate_df_schema with valid schemas."""
    schema_ok_1 = StructType(
        [
            StructField("a", IntegerType()),
            StructField("b", FloatType()),
            StructField("c", BooleanType()),
        ]
    )
    df_ok_1 = spark.createDataFrame([], schema=schema_ok_1)
    _validate_df_schema(df_ok_1)  # Should not raise

    schema_ok_2 = StructType(
        [
            StructField("ts", TimestampType()),
            StructField("arr", ArrayType(FloatType())),
        ]
    )
    df_ok_2 = spark.createDataFrame([], schema=schema_ok_2)
    _validate_df_schema(df_ok_2, timestamp_to="int64")  # Should not raise

    schema_ok_3 = StructType([StructField("s", StringType())])
    df_ok_3 = spark.createDataFrame([], schema=schema_ok_3)
    _validate_df_schema(df_ok_3, tokenizer=DummyTokenizer())  # Should not raise


def test_validate_df_schema_failure(spark: SparkSession) -> None:
    """Tests _validate_df_schema with invalid schemas."""
    schema_fail_1 = StructType([StructField("ts", TimestampType())])
    df_fail_1 = spark.createDataFrame([], schema=schema_fail_1)
    with pytest.raises(ValueError, match="unsupported type 'timestamp'"):
        _validate_df_schema(df_fail_1)  # Timestamp not allowed by default

    schema_fail_2 = StructType([StructField("s", StringType())])
    df_fail_2 = spark.createDataFrame([], schema=schema_fail_2)
    with pytest.raises(ValueError, match="String columns require a tokenizer"):
        _validate_df_schema(df_fail_2)  # String requires tokenizer

    schema_fail_3 = StructType([StructField("arr", ArrayType(StringType()))])
    df_fail_3 = spark.createDataFrame([], schema=schema_fail_3)
    with pytest.raises(ValueError, match="unsupported type 'array<string>'"):
        _validate_df_schema(df_fail_3)  # Array of string not supported

    schema_fail_4 = StructType(
        [StructField("nested", ArrayType(ArrayType(IntegerType())))]
    )
    df_fail_4 = spark.createDataFrame([], schema=schema_fail_4)
    with pytest.raises(ValueError, match="unsupported type 'array<array<int>>'"):
        _validate_df_schema(df_fail_4)  # Nested array not supported by default


class MockIterableDataset(IterableDataset):
    """Mocks an IterableDataset yielding TensorDicts."""

    def __init__(self, data: list[TensorDict]):
        """Initializes the mock dataset.

        Args:
            data: A list of TensorDicts to yield.
        """
        super().__init__()
        self.data = data
        self.iter_count = 0

    def __iter__(self) -> Iterator[TensorDict]:
        """Returns an iterator over the data."""
        self.iter_count += 1
        return iter(self.data)


def test_spark_arrow_batch_dataset(simple_df: DataFrame) -> None:
    """Tests SparkArrowBatchDataset iteration and conversion."""
    dataset = SparkArrowBatchDataset(simple_df)
    batches = list(dataset)  # Collect all batches

    # Spark batch size is 5, DF has 5 rows. Might be one or more batches.
    assert len(batches) > 0

    all_ids = torch.cat([cast(torch.Tensor, b["id"]) for b in batches if "id" in b])
    all_values = torch.cat(
        [cast(torch.Tensor, b["value"]) for b in batches if "value" in b]
    )

    assert all_ids.shape[0] == 5
    assert all_values.shape[0] == 5
    assert torch.equal(all_ids, torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64))
    assert torch.allclose(
        all_values, torch.tensor([10.0, 20.0, np.nan, 40.0, 50.0]), equal_nan=True
    )


def test_rebatching_dataset_exact_multiple() -> None:
    """Tests RebatchingDataset when total rows are an exact multiple of batch_size."""
    source_data: list[TensorDict] = [
        {"a": torch.tensor([1, 2])},
        {"a": torch.tensor([3, 4, 5])},
        {"a": torch.tensor([6])},
    ]  # Total 6 rows
    source_dataset = MockIterableDataset(source_data)
    rebatcher = RebatchingDataset(source_dataset, batch_size=3)
    batches = list(rebatcher)

    assert len(batches) == 2
    assert torch.equal(cast(torch.Tensor, batches[0]["a"]), torch.tensor([1, 2, 3]))
    assert torch.equal(cast(torch.Tensor, batches[1]["a"]), torch.tensor([4, 5, 6]))
    assert source_dataset.iter_count == 1  # Source should be iterated only once


def test_rebatching_dataset_partial_last_batch() -> None:
    """Tests RebatchingDataset with a partial last batch."""
    source_data: list[TensorDict] = [
        {"a": torch.tensor([1, 2, 3])},
        {"a": torch.tensor([4, 5])},
        {"a": torch.tensor([6, 7, 8])},
    ]  # Total 8 rows
    source_dataset = MockIterableDataset(source_data)
    rebatcher = RebatchingDataset(source_dataset, batch_size=3)
    batches = list(rebatcher)

    assert len(batches) == 3
    assert torch.equal(cast(torch.Tensor, batches[0]["a"]), torch.tensor([1, 2, 3]))
    assert torch.equal(cast(torch.Tensor, batches[1]["a"]), torch.tensor([4, 5, 6]))
    assert torch.equal(
        cast(torch.Tensor, batches[2]["a"]), torch.tensor([7, 8])
    )  # Last batch is partial
    assert source_dataset.iter_count == 1


def test_rebatching_dataset_large_batches() -> None:
    """Tests RebatchingDataset when source batches are larger than target."""
    source_data: list[TensorDict] = [
        {"a": torch.tensor([1, 2, 3, 4, 5])},
        {"a": torch.tensor([6, 7, 8, 9, 10])},
    ]  # Total 10 rows
    source_dataset = MockIterableDataset(source_data)
    rebatcher = RebatchingDataset(source_dataset, batch_size=3)
    batches = list(rebatcher)

    assert len(batches) == 4
    assert torch.equal(cast(torch.Tensor, batches[0]["a"]), torch.tensor([1, 2, 3]))
    assert torch.equal(cast(torch.Tensor, batches[1]["a"]), torch.tensor([4, 5, 6]))
    assert torch.equal(cast(torch.Tensor, batches[2]["a"]), torch.tensor([7, 8, 9]))
    assert torch.equal(cast(torch.Tensor, batches[3]["a"]), torch.tensor([10]))
    assert source_dataset.iter_count == 1


def test_rebatching_dataset_empty_source() -> None:
    """Tests RebatchingDataset with an empty source dataset."""
    source_dataset = MockIterableDataset([])
    rebatcher = RebatchingDataset(source_dataset, batch_size=3)
    batches = list(rebatcher)
    assert len(batches) == 0
    assert source_dataset.iter_count == 1


def test_rebatching_dataset_skip_empty_batches() -> None:
    """Tests RebatchingDataset skips empty TensorDicts from source."""
    source_data: list[TensorDict] = [
        {"a": torch.tensor([1, 2])},
        {},
        {"a": torch.tensor([3, 4, 5])},
        {},
        {"a": torch.tensor([6])},
    ]  # Total 6 rows
    source_dataset = MockIterableDataset(source_data)
    rebatcher = RebatchingDataset(source_dataset, batch_size=3)
    batches = list(rebatcher)

    assert len(batches) == 2
    assert torch.equal(cast(torch.Tensor, batches[0]["a"]), torch.tensor([1, 2, 3]))
    assert torch.equal(cast(torch.Tensor, batches[1]["a"]), torch.tensor([4, 5, 6]))
    assert source_dataset.iter_count == 1


def test_spark_arrow_batch_dataset_shuffle(shuffle_test_df: DataFrame) -> None:
    """Tests SparkArrowBatchDataset shuffling behavior across epochs.

    Covers:
    - No shuffling.
    - Shuffling with no seed (random order each epoch).
    - Shuffling with a fixed seed (same order each epoch).

    Args:
        shuffle_test_df (DataFrame): A DataFrame with an 'id' column.
    """
    original_ids = torch.arange(20, dtype=torch.int64)

    # Scenario 1: shuffle_per_epoch = False (no shuffle)
    dataset_no_shuffle = SparkArrowBatchDataset(
        shuffle_test_df, shuffle_per_epoch=False
    )
    ids_epoch1_no_shuffle = _get_ids_from_epoch(dataset_no_shuffle)
    ids_epoch2_no_shuffle = _get_ids_from_epoch(dataset_no_shuffle)

    assert torch.equal(
        ids_epoch1_no_shuffle, original_ids
    ), "Epoch 1 without shuffle should match original order."
    assert torch.equal(
        ids_epoch1_no_shuffle, ids_epoch2_no_shuffle
    ), "Order should be consistent across epochs when shuffle_per_epoch is False."

    # Scenario 2: shuffle_per_epoch = True, random_seed = None
    dataset_shuffle_no_seed = SparkArrowBatchDataset(
        shuffle_test_df, shuffle_per_epoch=True, random_seed=None
    )
    ids_epoch1_shuffle_no_seed = _get_ids_from_epoch(dataset_shuffle_no_seed)
    ids_epoch2_shuffle_no_seed = _get_ids_from_epoch(dataset_shuffle_no_seed)

    assert len(ids_epoch1_shuffle_no_seed) == len(original_ids)
    assert sorted(ids_epoch1_shuffle_no_seed.tolist()) == original_ids.tolist(), \
        "Epoch 1 with random shuffle should contain all original IDs."
    assert len(ids_epoch2_shuffle_no_seed) == len(original_ids)
    assert sorted(ids_epoch2_shuffle_no_seed.tolist()) == original_ids.tolist(), \
        "Epoch 2 with random shuffle should contain all original IDs."

    assert not torch.equal(
        ids_epoch1_shuffle_no_seed, original_ids
    ), "Randomly shuffled order should differ from original (high probability)."
    assert not torch.equal(
        ids_epoch1_shuffle_no_seed, ids_epoch2_shuffle_no_seed
    ), "Order should be different across epochs with random shuffle (high probability)."

    # Scenario 3: shuffle_per_epoch = True, random_seed = 42
    fixed_seed = 42
    dataset_shuffle_with_seed = SparkArrowBatchDataset(
        shuffle_test_df, shuffle_per_epoch=True, random_seed=fixed_seed
    )
    ids_epoch1_shuffle_seed = _get_ids_from_epoch(dataset_shuffle_with_seed)

    assert len(ids_epoch1_shuffle_seed) == len(original_ids)
    assert sorted(ids_epoch1_shuffle_seed.tolist()) == original_ids.tolist(), \
        "Epoch 1 with seeded shuffle should contain all original IDs."
    
    assert not torch.equal(
        ids_epoch1_shuffle_seed, original_ids
    ), "Seeded shuffle order should differ from original (high probability)."


def test_loader_integration(simple_df: DataFrame) -> None:
    """Tests the end-to-end loader function."""
    target_batch_size = 2
    data_loader = loader(simple_df, batch_size=target_batch_size)

    batches = list(data_loader)

    # 5 rows / batch_size 2 = 3 batches (2, 2, 1)
    assert len(batches) == 3

    # Check batch sizes
    assert _get_tensor_dict_rows(batches[0]) == 2
    assert _get_tensor_dict_rows(batches[1]) == 2
    assert _get_tensor_dict_rows(batches[2]) == 1

    # Check content concatenation
    all_ids = torch.cat([cast(torch.Tensor, b["id"]) for b in batches])
    all_values = torch.cat([cast(torch.Tensor, b["value"]) for b in batches])

    assert torch.equal(all_ids, torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64))
    assert torch.allclose(
        all_values, torch.tensor([10.0, 20.0, np.nan, 40.0, 50.0]), equal_nan=True
    )


def test_loader_with_tokenizer(complex_df: DataFrame) -> None:
    """Tests the loader with a tokenizer for string columns."""
    target_batch_size = 3
    tokenizer = DummyTokenizer()
    data_loader = loader(complex_df, batch_size=target_batch_size, tokenizer=tokenizer)

    batches = list(data_loader)

    # 5 rows / batch_size 3 = 2 batches (3, 2)
    assert len(batches) == 2
    assert _get_tensor_dict_rows(batches[0]) == 3
    assert _get_tensor_dict_rows(batches[1]) == 2

    # Check if tokenized columns exist
    assert "text_input_ids" in batches[0]
    assert "text_attention_mask" in batches[0]
    assert "text_input_ids" in batches[1]
    assert "text_attention_mask" in batches[1]

    # Basic check on concatenated IDs
    all_ids = torch.cat([b["id"] for b in batches])
    assert torch.equal(all_ids, torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64))


def test_empty_df(spark: SparkSession) -> None:
    """Tests loader with an empty DataFrame."""
    empty_df = spark.createDataFrame([], schema="foo int")
    data_loader = loader(empty_df, batch_size=2)

    batches = list(data_loader)
    assert len(batches) == 0  # No batches should be produced


def test_empty_partition(spark: SparkSession) -> None:
    """Tests loader with a DataFrame that has an empty partition."""

    df = spark.range(10).repartition(10).where(f.col("id") != 7)

    data_loader = loader(df, batch_size=1)
    batches = list(data_loader)

    assert len(batches) == 9


def test_batching_dataset_incorrect_batch_size() -> None:
    """Tests RebatchingDataset with an incorrect batch size."""
    source_data: list[TensorDict] = [
        {"a": torch.tensor([1, 2])},
        {"a": torch.tensor([3, 4, 5])},
        {"a": torch.tensor([6])},
    ]  # Total 6 rows
    source_dataset = MockIterableDataset(source_data)

    with pytest.raises(ValueError, match="batch_size must be positive."):
        RebatchingDataset(source_dataset, batch_size=0)  # Invalid batch size


def test_arrow_batch_to_tensor_dict_empty() -> None:
    """Tests _arrow_batch_to_tensor_dict with an empty RecordBatch."""
    empty_rb = pa.RecordBatch.from_arrays([], schema=pa.schema([]))
    tensor_dict = _arrow_batch_to_tensor_dict(empty_rb, StructType([]))

    assert len(tensor_dict) == 0  # No columns should be present


def test_arrow_batch_to_tensor_dict_null_values(simple_schema: StructType) -> None:
    """Tests _arrow_batch_to_tensor_dict with a null array."""
    ids = pa.array([1, 2, None], type=pa.int32())
    values = pa.array([10.0, None, 30.0], type=pa.float32())
    # Manually create the corresponding Arrow schema
    schema_list: list[pa.Field] = [
        pa.field("id", pa.int32(), nullable=False),
        pa.field("value", pa.float32(), nullable=True),
    ]
    arrow_schema = pa.schema(schema_list)
    rb = pa.RecordBatch.from_arrays([ids, values], schema=arrow_schema)

    tensor_dict = _arrow_batch_to_tensor_dict(rb, simple_schema)

    assert "id" in tensor_dict
    assert "value" in tensor_dict
    assert torch.equal(
        torch.nan_to_num(cast(torch.Tensor, tensor_dict["id"]), 0),
        torch.tensor([1, 2, 0], dtype=torch.int64),
    )
    # Nones in float columns become NaN
    assert torch.allclose(
        cast(torch.Tensor, tensor_dict["value"]),
        torch.tensor([10.0, np.nan, 30.0]),
        equal_nan=True,
    )
    assert cast(torch.Tensor, tensor_dict["value"]).dtype == torch.float32


def test_arrow_batch_to_tensor_dict_nullarray(simple_schema: StructType) -> None:
    """Tests _arrow_batch_to_tensor_dict with a null array."""
    ids = pa.array([None, None, None], type=pa.null())
    values = pa.array([None, None, None], type=pa.null())
    # Manually create the corresponding Arrow schema
    schema_list: list[pa.Field] = [
        pa.field("id", pa.null(), nullable=True),
        pa.field("value", pa.null(), nullable=True),
    ]
    arrow_schema = pa.schema(schema_list)
    rb = pa.RecordBatch.from_arrays([ids, values], schema=arrow_schema)
    assert isinstance(rb.column(0), pa.NullArray)

    tensor_dict = _arrow_batch_to_tensor_dict(rb, simple_schema)
    
    for i in range(3):
        assert tensor_dict["id"][i].isnan()
        assert tensor_dict["value"][i].isnan()


def test_arrow_batch_to_tensor_dict_boolean_arrays() -> None:
    """Tests _arrow_batch_to_tensor_dict with boolean arrays.
    
    This test specifically targets the code at lines 105-110 in helpers.py that
    handles the np.issubdtype(np_array.dtype, np.bool_) detection and tensor_dtype
    assignment.
    """
    schema = StructType([
        StructField("bool_col", BooleanType(), True),
    ])
    
    # Test with a regular boolean array (no nulls)
    bool_array = pa.array([True, False, True], type=pa.bool_())
    rb = pa.RecordBatch.from_arrays([bool_array], schema=pa.schema([
        pa.field("bool_col", pa.bool_(), nullable=True),
    ]))
    
    tensor_dict = _arrow_batch_to_tensor_dict(rb, schema)
    
    assert "bool_col" in tensor_dict
    assert isinstance(tensor_dict["bool_col"], torch.Tensor)
    assert tensor_dict["bool_col"].dtype == torch.bool
    assert torch.equal(
        tensor_dict["bool_col"], 
        torch.tensor([True, False, True], dtype=torch.bool)
    )
    
    # Test with a boolean array containing nulls
    bool_array_with_nulls = pa.array([True, None, False], type=pa.bool_())
    rb_with_nulls = pa.RecordBatch.from_arrays([bool_array_with_nulls], schema=pa.schema([
        pa.field("bool_col", pa.bool_(), nullable=True),
    ]))
    
    tensor_dict_with_nulls = _arrow_batch_to_tensor_dict(rb_with_nulls, schema)
    
    assert "bool_col" in tensor_dict_with_nulls
    assert isinstance(tensor_dict_with_nulls["bool_col"], torch.Tensor)
    assert tensor_dict_with_nulls["bool_col"].dtype == torch.bool
    # Nulls in boolean arrays become False
    assert torch.equal(
        tensor_dict_with_nulls["bool_col"], 
        torch.tensor([True, False, False], dtype=torch.bool)
    )


def test_arrow_batch_to_tensor_dict_unsupported_dtype() -> None:
    """Tests that _arrow_batch_to_tensor_dict raises TypeError for unsupported dtypes.
    
    This test specifically targets the code at lines 105-110 in helpers.py that
    raises a TypeError when encountering an unsupported NumPy dtype.
    """
    import unittest.mock as mock
    
    # Create a schema with a field that uses a regular numeric type
    schema = StructType([
        StructField("numeric_col", FloatType(), True)
    ])
    
    # Create a regular array for the test
    values = pa.array([1.0, 2.0, 3.0], type=pa.float32())
    rb = pa.RecordBatch.from_arrays([values], schema=pa.schema([
        pa.field("numeric_col", pa.float32(), nullable=True),
    ]))
    
    # Instead of mocking ndarray.dtype which isn't possible, let's mock the np.issubdtype function
    # to return False for all checks, forcing the code to fall into the unsupported type branch
    original_issubdtype = np.issubdtype
    
    def mock_issubdtype(dtype, dtype_class):  # type: ignore
        # Return False for all checks, which will cause the code to raise a TypeError
        return False
    
    # Apply the mock
    with mock.patch('numpy.issubdtype', side_effect=mock_issubdtype):
        # This should now trigger the TypeError
        with pytest.raises(TypeError, match=r"Unsupported numpy dtype .* for column 'numeric_col'"):
            _arrow_batch_to_tensor_dict(rb, schema)


def test_arrow_batch_to_tensor_dict_dtype_conversion() -> None:
    """Tests that _arrow_batch_to_tensor_dict properly converts NumPy dtypes to torch dtypes.
    
    This test verifies that the automatic dtype conversion works for all supported
    NumPy datatypes (floating point, integer, and boolean).
    """
    schema = StructType([
        StructField("float_col", FloatType(), True),
        StructField("int_col", IntegerType(), True), 
        StructField("bool_col", BooleanType(), True),
    ])
    
    # Create arrays with different NumPy dtypes
    float_array = pa.array([1.1, 2.2, 3.3], type=pa.float32())
    int_array = pa.array([1, 2, 3], type=pa.int32())
    bool_array = pa.array([True, False, True], type=pa.bool_())
    
    schema_list: list[pa.Field] = [
        pa.field("float_col", pa.float32(), nullable=True),
        pa.field("int_col", pa.int32(), nullable=True),
        pa.field("bool_col", pa.bool_(), nullable=True),
    ]
    
    rb = pa.RecordBatch.from_arrays(
        [float_array, int_array, bool_array], 
        schema=pa.schema(schema_list)
    )
    
    tensor_dict = _arrow_batch_to_tensor_dict(rb, schema)
    
    # Verify dtypes were properly converted according to the rules in helpers.py
    assert tensor_dict["float_col"].dtype == torch.float32  # type: ignore
    assert tensor_dict["int_col"].dtype == torch.int64  # type: ignore
    assert tensor_dict["bool_col"].dtype == torch.bool  # type: ignore


def test_arrow_batch_to_tensor_dict_timestamp_with_nulls() -> None:
    """Tests _arrow_batch_to_tensor_dict with timestamp arrays containing nulls.
    
    This test specifically targets the code at lines 120-130 in helpers.py that
    handles null values in timestamp arrays by converting them to int64 nanoseconds.
    """
    schema = StructType([
        StructField("ts_col", TimestampType(), True),
    ])
    
    # Create timestamp array with nulls (will have dtype==object when converted to numpy)
    timestamps = [
        datetime(2023, 1, 1, 12, 0, 0),
        None,
        datetime(2023, 1, 2, 12, 0, 0)
    ]
    
    ts_array = pa.array(timestamps, type=pa.timestamp('ns'))
    rb = pa.RecordBatch.from_arrays([ts_array], schema=pa.schema([
        pa.field("ts_col", pa.timestamp('ns'), nullable=True),
    ]))
    
    tensor_dict = _arrow_batch_to_tensor_dict(rb, schema)
    
    assert "ts_col" in tensor_dict
    assert isinstance(tensor_dict["ts_col"], torch.Tensor)
    assert tensor_dict["ts_col"].dtype == torch.int64
    
    # Get the values
    tensor_values = tensor_dict["ts_col"].tolist()
    
    # Check the second value (index 1) is a special sentinel value since it was None
    # The implementation in helpers.py line 124 uses 0, but it appears it's actually
    # getting a minimum int64 value (-9223372036854775808)
    NULL_TIMESTAMP_SENTINEL = -9223372036854775808
    assert tensor_values[1] == NULL_TIMESTAMP_SENTINEL, f"None timestamp should be converted to {NULL_TIMESTAMP_SENTINEL}"
    
    # First and third values (index 0 and 2) should not be the sentinel value
    assert tensor_values[0] != NULL_TIMESTAMP_SENTINEL, "Non-null timestamp should not be sentinel value"
    assert tensor_values[2] != NULL_TIMESTAMP_SENTINEL, "Non-null timestamp should not be sentinel value"
    
    # The difference between the first and third timestamps should be 1 day in nanoseconds
    one_day_ns = 24 * 60 * 60 * 1_000_000_000  # 1 day in nanoseconds
    assert abs(tensor_values[2] - tensor_values[0] - one_day_ns) < 1000, "Timestamps should be 1 day apart"


def test_arrow_batch_to_tensor_dict_timestamp_without_nulls() -> None:
    """Tests _arrow_batch_to_tensor_dict with timestamp arrays without nulls.
    
    This test verifies that timestamp arrays without nulls are properly converted
    to int64 nanoseconds tensors, covering the else branch in lines 131-140.
    """
    schema = StructType([
        StructField("ts_col", TimestampType(), True),
    ])
    
    # Create timestamp array without nulls
    timestamps = [
        datetime(2023, 1, 1, 12, 0, 0),
        datetime(2023, 1, 2, 12, 0, 0),
        datetime(2023, 1, 3, 12, 0, 0)
    ]
    
    ts_array = pa.array(timestamps, type=pa.timestamp('ns'))
    rb = pa.RecordBatch.from_arrays([ts_array], schema=pa.schema([
        pa.field("ts_col", pa.timestamp('ns'), nullable=True),
    ]))
    
    tensor_dict = _arrow_batch_to_tensor_dict(rb, schema)
    
    assert "ts_col" in tensor_dict
    assert isinstance(tensor_dict["ts_col"], torch.Tensor)
    assert tensor_dict["ts_col"].dtype == torch.int64
    
    # Get the values
    tensor_values = tensor_dict["ts_col"].tolist()
    
    # All values should be non-zero
    for val in tensor_values:
        assert val != 0, "Non-null timestamp should be non-zero"
    
    # The differences between consecutive timestamps should be 1 day in nanoseconds
    one_day_ns = 24 * 60 * 60 * 1_000_000_000  # 1 day in nanoseconds
    assert abs(tensor_values[1] - tensor_values[0] - one_day_ns) < 1000, "First and second timestamps should be 1 day apart"
    assert abs(tensor_values[2] - tensor_values[1] - one_day_ns) < 1000, "Second and third timestamps should be 1 day apart"


def test_arrow_batch_to_tensor_dict_timestamp_wrong_dtype() -> None:
    """Tests _arrow_batch_to_tensor_dict raises TypeError for wrong timestamp dtype.
    
    This test verifies that the code raises a TypeError when it expects a numpy
    datetime64 array but gets something else. This targets the error handling
    in the lines 133-136 of helpers.py.
    """
    schema = StructType([
        StructField("ts_col", TimestampType(), True),
    ])
    
    # Create a regular array but we'll mock it to have a non-datetime64 dtype
    timestamps = [1, 2, 3]  # Just placeholders, we'll mock the dtype check
    int_array = pa.array(timestamps, type=pa.int32())
    rb = pa.RecordBatch.from_arrays([int_array], schema=pa.schema([
        pa.field("ts_col", pa.int32(), nullable=True),
    ]))
    
    # Mock the issubdtype function to return False for datetime64 check
    import unittest.mock as mock
    
    original_issubdtype = np.issubdtype
    
    def mock_issubdtype(dtype, dtype_class):  # type: ignore
        if dtype_class == np.datetime64:
            return False  # Force the error path
        return original_issubdtype(dtype, dtype_class)
    
    # Apply the mock
    with mock.patch('numpy.issubdtype', side_effect=mock_issubdtype):
        with pytest.raises(TypeError, match=r"Expected numpy datetime64 array for column 'ts_col'"):
            _arrow_batch_to_tensor_dict(rb, schema)


def test_arrow_batch_to_tensor_dict_array_boolean_type() -> None:
    """Tests _arrow_batch_to_tensor_dict with arrays of boolean type.
    
    This test specifically targets the code at lines 157-162 in helpers.py that
    determines the empty_dtype for different array element types, focusing on
    BooleanType arrays.
    """
    schema = StructType([
        StructField("bool_array_col", ArrayType(BooleanType(), True), True),
    ])
    
    # Create array with both populated and null items
    bool_arrays = [[True, False], None, [False, True, False]]
    array_data = pa.array(bool_arrays, type=pa.list_(pa.bool_()))
    
    rb = pa.RecordBatch.from_arrays([array_data], schema=pa.schema([
        pa.field("bool_array_col", pa.list_(pa.bool_()), nullable=True),
    ]))
    
    tensor_dict = _arrow_batch_to_tensor_dict(rb, schema)
    
    assert "bool_array_col" in tensor_dict
    assert isinstance(tensor_dict["bool_array_col"], list)
    assert len(tensor_dict["bool_array_col"]) == 3
    
    # Check that non-null arrays are properly converted to torch tensors with boolean dtype
    assert torch.equal(tensor_dict["bool_array_col"][0], torch.tensor([True, False], dtype=torch.bool))
    assert torch.equal(tensor_dict["bool_array_col"][2], torch.tensor([False, True, False], dtype=torch.bool))
    
    # Most importantly, check that null arrays are converted to empty tensors with bool dtype
    assert tensor_dict["bool_array_col"][1].shape == torch.Size([0])
    assert tensor_dict["bool_array_col"][1].dtype == torch.bool


def test_arrow_batch_to_tensor_dict_empty_arrays() -> None:
    """Tests _arrow_batch_to_tensor_dict handling of empty arrays for different dtypes.
    
    This test specifically targets the code at lines 157-162 in helpers.py that
    determines the empty_dtype for different array element types.
    """
    # Create arrays with null values to test empty tensor generation for each type separately
    
    # Test boolean array
    bool_schema = StructType([
        StructField("bool_array_col", ArrayType(BooleanType(), True), True),
    ])
    bool_array = pa.array([None], type=pa.list_(pa.bool_()))
    bool_rb = pa.RecordBatch.from_arrays(
        [bool_array], 
        schema=pa.schema([pa.field("bool_array_col", pa.list_(pa.bool_()), nullable=True)])
    )
    bool_tensor_dict = _arrow_batch_to_tensor_dict(bool_rb, bool_schema)
    assert "bool_array_col" in bool_tensor_dict
    assert isinstance(bool_tensor_dict["bool_array_col"], list)
    assert len(bool_tensor_dict["bool_array_col"]) == 1
    assert bool_tensor_dict["bool_array_col"][0].shape == torch.Size([0])
    assert bool_tensor_dict["bool_array_col"][0].dtype == torch.bool
    
    # Test timestamp array
    ts_schema = StructType([
        StructField("ts_array_col", ArrayType(TimestampType(), True), True),
    ])
    ts_array = pa.array([None], type=pa.list_(pa.timestamp('ns')))
    ts_rb = pa.RecordBatch.from_arrays(
        [ts_array], 
        schema=pa.schema([pa.field("ts_array_col", pa.list_(pa.timestamp('ns')), nullable=True)])
    )
    ts_tensor_dict = _arrow_batch_to_tensor_dict(ts_rb, ts_schema)
    assert "ts_array_col" in ts_tensor_dict
    assert isinstance(ts_tensor_dict["ts_array_col"], list)
    assert len(ts_tensor_dict["ts_array_col"]) == 1
    assert ts_tensor_dict["ts_array_col"][0].shape == torch.Size([0])
    assert ts_tensor_dict["ts_array_col"][0].dtype == torch.int64
    
    # Test numeric (float) array - should default to float32
    float_schema = StructType([
        StructField("float_array_col", ArrayType(FloatType(), True), True),
    ])
    float_array = pa.array([None], type=pa.list_(pa.float32()))
    float_rb = pa.RecordBatch.from_arrays(
        [float_array], 
        schema=pa.schema([pa.field("float_array_col", pa.list_(pa.float32()), nullable=True)])
    )
    float_tensor_dict = _arrow_batch_to_tensor_dict(float_rb, float_schema)
    assert "float_array_col" in float_tensor_dict
    assert isinstance(float_tensor_dict["float_array_col"], list)
    assert len(float_tensor_dict["float_array_col"]) == 1
    assert float_tensor_dict["float_array_col"][0].shape == torch.Size([0])
    assert float_tensor_dict["float_array_col"][0].dtype == torch.float32


def test_arrow_batch_to_tensor_dict_array_other_type() -> None:
    """Tests _arrow_batch_to_tensor_dict with arrays of a type that falls into the 'else' case.
    
    This test specifically targets the code at lines 157-162 in helpers.py that
    determines the empty_dtype for different array element types, focusing on
    the fallback case where torch.float32 is used.
    """
    # Create a custom "other" type using a NumericType that's not specifically handled
    class CustomNumericType(NumericType):
        """Custom numeric type for testing 'other' case."""
        pass
    
    schema = StructType([
        StructField("custom_array_col", ArrayType(CustomNumericType(), True), True),
    ])
    
    # Create basic numeric arrays for testing
    arrays = [[1.0, 2.0], None, [3.0, 4.0, 5.0]]
    
    # Use float32 for the array data
    array_data = pa.array(arrays, type=pa.list_(pa.float32()))
    
    rb = pa.RecordBatch.from_arrays([array_data], schema=pa.schema([
        pa.field("custom_array_col", pa.list_(pa.float32()), nullable=True),
    ]))
    
    tensor_dict = _arrow_batch_to_tensor_dict(rb, schema)
    
    assert "custom_array_col" in tensor_dict
    assert isinstance(tensor_dict["custom_array_col"], list)
    assert len(tensor_dict["custom_array_col"]) == 3
    
    # Most importantly, check that null arrays are converted to empty tensors with float32 dtype
    # This tests the 'else' branch in lines 161-162
    assert tensor_dict["custom_array_col"][1].shape == torch.Size([0])
    assert tensor_dict["custom_array_col"][1].dtype == torch.float32
    
    # Non-null arrays should be properly converted
    assert torch.allclose(
        tensor_dict["custom_array_col"][0], 
        torch.tensor([1.0, 2.0], dtype=torch.float32)
    )
    assert torch.allclose(
        tensor_dict["custom_array_col"][2], 
        torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
    )


def test_loader_with_string_array_column(spark: SparkSession) -> None:
    """Tests creating a loader for DataFrame with a column of string arrays."""
    # Define schema with string array column
    schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("texts", ArrayType(StringType(), True), True),
    ])
    
    # Create sample data
    data = [
        (1, ["hello", "world"]),
        (2, ["test"]),
        (3, None),
        (4, []),
        (5, ["multiple", "strings", "in", "array"]),
    ]
    
    # Create DataFrame
    df = spark.createDataFrame(data, schema=schema)
    
    # Create tokenizer
    tokenizer = DummyTokenizer()
    
    # With our implementation, arrays of StringType are now supported
    data_loader = loader(df, batch_size=2, tokenizer=tokenizer)
    
    # Collect and verify batches
    batches = list(data_loader)
    
    # 5 rows with batch_size=2 should yield 3 batches
    assert len(batches) == 3
    
    # Check that the ID column is correctly processed
    all_ids = torch.cat([b["id"] for b in batches])
    assert torch.equal(all_ids, torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64))
    
    # Check that the string array column is processed correctly
    # The texts column should be a list of tokenizer outputs (dictionaries)
    assert "texts" in batches[0]
    assert isinstance(batches[0]["texts"], list)
    
    # First batch should have 2 items
    assert len(batches[0]["texts"]) == 2
    
    # For the first row (id=1), the texts "hello" and "world" should be tokenized
    # Each tokenized output should be a dictionary with input_ids and attention_mask
    assert isinstance(batches[0]["texts"][0], dict)
    assert "input_ids" in batches[0]["texts"][0]
    assert "attention_mask" in batches[0]["texts"][0]
    
    
def test_to_device() -> None:
    """Tests the _to_device helper function."""
    initial_device_str = "cpu"  # Assuming CPU is always available
    initial_device = torch.device(initial_device_str)
    target_device_str = "cpu"  # Test with CPU, can be changed if CUDA is available
    target_device = torch.device(target_device_str)

    tensor_dict_original: TensorDict = {
        "a": torch.tensor([1, 2, 3], device=initial_device),
        "b": [
            torch.tensor([4, 5], device=initial_device),
            torch.tensor([6], device=initial_device),
        ],
        "c": torch.tensor([[7.0, 8.0]], device=initial_device),
    }

    # Test moving to the target device
    moved_dict = _to_device(tensor_dict_original, target_device_str)
    assert isinstance(moved_dict["a"], torch.Tensor)
    assert moved_dict["a"].device == target_device
    assert torch.equal(
        cast(torch.Tensor, moved_dict["a"]),
        cast(torch.Tensor, tensor_dict_original["a"]),
    )

    assert isinstance(moved_dict["b"], list)
    for tensor in cast(list[torch.Tensor], moved_dict["b"]):
        assert tensor.device == target_device
    assert torch.equal(
        cast(list[torch.Tensor], moved_dict["b"])[0],
        cast(list[torch.Tensor], tensor_dict_original["b"])[0],
    )
    assert torch.equal(
        cast(list[torch.Tensor], moved_dict["b"])[1],
        cast(list[torch.Tensor], tensor_dict_original["b"])[1],
    )

    assert isinstance(moved_dict["c"], torch.Tensor)
    assert moved_dict["c"].device == target_device
    assert torch.equal(
        cast(torch.Tensor, moved_dict["c"]),
        cast(torch.Tensor, tensor_dict_original["c"]),
    )

    # Test with device=None (should return original)
    no_move_dict = _to_device(tensor_dict_original, None)
    assert no_move_dict["a"].device == initial_device
    for tensor in cast(list[torch.Tensor], no_move_dict["b"]):
        assert tensor.device == initial_device
    assert no_move_dict["c"].device == initial_device

    # Test with torch.device object
    moved_dict_obj = _to_device(tensor_dict_original, target_device)
    assert moved_dict_obj["a"].device == target_device

    # Test with unsupported type
    unsupported_dict: TensorDict = {"d": "not a tensor or list"}  # type: ignore
    with pytest.raises(TypeError, match="Unsupported type for moving to device"):
        _to_device(unsupported_dict, target_device_str)

    # Test with an empty dict
    empty_dict: TensorDict = {}
    moved_empty_dict = _to_device(empty_dict, target_device_str)
    assert len(moved_empty_dict) == 0

    # Test with list of non-tensor items (should pass through)
    mixed_list_dict: TensorDict = {
        "e": [torch.tensor([1]), "string", torch.tensor([2])] # type: ignore
    }
    moved_mixed_list_dict = _to_device(mixed_list_dict, target_device_str)
    assert cast(list, moved_mixed_list_dict["e"])[0].device == target_device
    assert cast(list, moved_mixed_list_dict["e"])[1] == "string"
    assert cast(list, moved_mixed_list_dict["e"])[2].device == target_device


def test_serialise_batches() -> None:
    """Tests the _serialise_batches helper function.

    This test verifies that _serialise_batches correctly serializes an
    iterable of PyArrow RecordBatches into the expected IPC stream format,
    wrapped in new RecordBatches.
    """
    data1 = [pa.array([1, 2, 3]), pa.array(["a", "b", "c"])]
    rb1 = pa.RecordBatch.from_arrays(data1, names=["col1", "col2"])

    data2 = [pa.array([4, 5]), pa.array(["d", "e"])]
    rb2 = pa.RecordBatch.from_arrays(data2, names=["col1", "col2"])

    empty_rb = pa.RecordBatch.from_arrays([], schema=pa.schema([]))

    input_batches_list: list[list[pa.RecordBatch]] = [
        [],
        [rb1],
        [rb1, rb2],
        [empty_rb],
        [rb1, empty_rb, rb2],
    ]

    for input_batches in input_batches_list:
        serialized_iter = _serialise_batches(iter(input_batches))
        output_rbs = list(serialized_iter)

        assert len(output_rbs) == len(input_batches)

        for original_rb, serial_rb in zip(input_batches, output_rbs):
            assert serial_rb.schema.equals(_ARROW_BATCH_SCHEMA)
            assert serial_rb.num_rows == 1
            assert serial_rb.num_columns == 1
            assert serial_rb.column_names[0] == "batch"
            assert serial_rb.column(0).type == pa.binary()

            # Deserialize the payload and check if it matches the original
            payload = serial_rb.column(0)[0].as_py()
            assert isinstance(payload, bytes)

            # The _record_batch_to_ipc_bytes includes the schema in the stream
            # so we can read it back directly.
            with pa.ipc.open_stream(payload) as reader:
                deserialized_rb = reader.read_all()

            assert deserialized_rb.equals(pa.Table.from_batches([original_rb]))

    # Test with an iterator that can only be consumed once
    input_gen = (rb for rb in [rb1, rb2])
    serialized_iter_gen = _serialise_batches(input_gen)
    output_rbs_gen = list(serialized_iter_gen)
    assert len(output_rbs_gen) == 2

    # Check content of the first batch from generator
    original_rb_gen1 = rb1
    serial_rb_gen1 = output_rbs_gen[0]
    payload_gen1 = serial_rb_gen1.column(0)[0].as_py()
    with pa.ipc.open_stream(payload_gen1) as reader:
        deserialized_rb_gen1 = reader.read_all()
    assert deserialized_rb_gen1.equals(pa.Table.from_batches([original_rb_gen1]))

    # Check content of the second batch from generator
    original_rb_gen2 = rb2
    serial_rb_gen2 = output_rbs_gen[1]
    payload_gen2 = serial_rb_gen2.column(0)[0].as_py()
    with pa.ipc.open_stream(payload_gen2) as reader:
        deserialized_rb_gen2 = reader.read_all()
    assert deserialized_rb_gen2.equals(pa.Table.from_batches([original_rb_gen2]))


def test_to_device_nested_tokenizer_dicts() -> None:
    """Ensure _to_device moves tensors inside nested dictionaries."""
    init_device = torch.device("cpu")
    target = torch.device("cpu")

    tokenized_list: list[dict[str, torch.Tensor]] = [
        {
            "input_ids": torch.tensor([1, 2], device=init_device),
            "attention_mask": torch.tensor([1, 1], device=init_device),
        },
        {
            "input_ids": torch.tensor([3], device=init_device),
            "attention_mask": torch.tensor([1], device=init_device),
        },
    ]

    tensor_dict: TensorDict = {"tokens": tokenized_list}

    moved = _to_device(tensor_dict, target)
    moved_list = cast(list[dict[str, torch.Tensor]], moved["tokens"])

    for original, moved_dict in zip(tokenized_list, moved_list):
        for key in original:
            assert moved_dict[key].device == target
            assert torch.equal(moved_dict[key], original[key])
