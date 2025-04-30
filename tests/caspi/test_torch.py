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