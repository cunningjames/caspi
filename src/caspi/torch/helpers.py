"""Helper functions for PyTorch integration with Spark."""
from collections.abc import Iterator
from typing import Callable, Iterable

import numpy as np
import pyarrow as pa
import torch
import torch.nn.functional as f
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    NumericType,
    StructType,
    TimestampType,
    StringType,
)


# Type hint for the final output batch
TensorBatch = torch.Tensor | list[torch.Tensor]
TensorDict = dict[str, TensorBatch]

# Constants
_ARROW_BATCH_SCHEMA = pa.schema([pa.field("batch", pa.binary())])


def _arrow_batch_to_tensor_dict(
    record_batch: pa.RecordBatch,
    spark_schema: StructType,
    tokenizer: Callable | None = None,
) -> TensorDict:
    """Converts a pyarrow.RecordBatch to a dictionary of tensors.

    Args:
        record_batch: Input Arrow RecordBatch.
        spark_schema: Original Spark DataFrame schema to interpret types.
        tokenizer: Optional tokenizer used to process StringType columns.

    Returns:
        A dictionary mapping column names to torch tensors or lists of tensors.
    """

    tensors: TensorDict = {}
    if record_batch.num_rows == 0:
        return {}  # Handle empty batches

    for i, field in enumerate(spark_schema.fields):
        col_name = field.name
        col_type = field.dataType
        arrow_array = record_batch.column(i)

        if isinstance(col_type, (NumericType, BooleanType)):
            # Fast path for scalar types.  Arrow normally gives us a NumPy array
            # of primitive dtype, but when NULLs are present it may fall back to
            # dtype=object.  We handle that case explicitly.
            np_array = arrow_array.to_numpy(zero_copy_only=False)

            # Fallback when Arrow produced an object array due to NULLs
            if np_array.dtype == object:
                py_list2 = arrow_array.to_pylist()
                if isinstance(col_type, NumericType):
                    # Represent NULLs as NaN and cast the whole column to float32
                    np_array = np.array(
                        [float(x) if x is not None else np.nan for x in py_list2],
                        dtype=np.float32,
                    )
                    tensor_dtype = torch.float32
                elif isinstance(col_type, BooleanType):
                    np_array = np.array(
                        [bool(x) if x is not None else False for x in py_list2],
                        dtype=bool,
                    )
                    tensor_dtype = torch.bool
                else:
                    raise TypeError(
                        f"Unsupported scalar type with NULLs for column '{col_name}'."
                    )
            else:
                # Normal primitive NumPy array branch
                if np.issubdtype(np_array.dtype, np.floating):
                    tensor_dtype = torch.float32
                elif np.issubdtype(np_array.dtype, np.integer):
                    tensor_dtype = torch.int64
                elif np.issubdtype(np_array.dtype, np.bool_):
                    tensor_dtype = torch.bool
                else:
                    raise TypeError(
                        f"Unsupported numpy dtype {np_array.dtype} for column '{col_name}'."
                    )

            tensor = torch.from_numpy(np_array.copy())
            if tensor_dtype is not None and tensor.dtype != tensor_dtype:
                tensor = tensor.to(tensor_dtype)
            tensors[col_name] = tensor
        elif isinstance(col_type, TimestampType):
            # Convert Arrow Timestamp array to numpy array
            np_array = arrow_array.to_numpy(zero_copy_only=False)

            # Handle potential object dtype if nulls are present within the batch
            if np_array.dtype == object:
                py_list3 = arrow_array.to_pylist()
                # Convert datetime objects to int64 nanoseconds, handle None (using 0 like NullArray case)
                np_array = np.array(
                    [
                        int(x.timestamp() * 1e9) if x is not None else 0
                        for x in py_list3
                    ],
                    dtype=np.int64,
                )
            else:
                # Convert numpy datetime64 to int64 nanoseconds
                if not np.issubdtype(np_array.dtype, np.datetime64):
                    raise TypeError(
                        f"Expected numpy datetime64 array for column '{col_name}', got {np_array.dtype}"
                    )
                # Ensure nanosecond precision before converting to int64
                np_array = np_array.astype("datetime64[ns]").astype(np.int64)

            # Convert numpy array to torch tensor
            tensor = torch.from_numpy(np_array).to(torch.int64)
            tensors[col_name] = tensor
        elif isinstance(col_type, StringType):
            py_list4 = ["" if x is None else str(x) for x in arrow_array.to_pylist()]

            assert tokenizer is not None, (
                "Tokenizer must be provided for StringType columns."
            )
            encodings = tokenizer(
                py_list4,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
            for k, v in encodings.items():
                tensors[f"{col_name}_{k}"] = v
        elif isinstance(col_type, ArrayType):
            # Always emit a *list* of tensors to guarantee consistent typing
            # across Arrow batches. This avoids mismatches such as one batch
            # producing a stacked tensor (when all inner sequences share the
            # same length) and another producing a Python list (when lengths
            # differ). Downstream concatenation logic can safely assume
            # `list[torch.Tensor]` for every ArrayType column.
            processed_list: list[torch.Tensor] = []
            inner_type = col_type.elementType

            if isinstance(inner_type, NumericType):
                empty_dtype = torch.float32
            elif isinstance(inner_type, BooleanType):
                empty_dtype = torch.bool
            elif isinstance(inner_type, TimestampType):
                empty_dtype = torch.int64
            else:
                empty_dtype = torch.float32

            for item in arrow_array.to_pylist():
                if item is None:
                    processed_list.append(torch.empty(0, dtype=empty_dtype))
                else:
                    # `item` may already be a list/np.ndarray; torch.as_tensor handles
                    # both
                    processed_list.append(torch.as_tensor(item))

            tensors[col_name] = processed_list
        else:
            raise TypeError(
                f"Column '{col_name}' has unsupported type '{col_type.simpleString()}'"
                " for tensor conversion."
            )

    return tensors


def _concatenate_tensor_dicts(dict1: TensorDict, dict2: TensorDict) -> TensorDict:
    """Concatenates two TensorDicts along the batch dimension.

    Assumes keys are the same in both dictionaries.
    Handles torch.Tensor and list[torch.Tensor].

    Args:
        dict1: The first TensorDict (can be empty).
        dict2: The second TensorDict.

    Returns:
        A new TensorDict containing the concatenated data.

    Raises:
        ValueError: If keys don't match or concatenation fails.
    """

    if not dict1:
        return dict2.copy()  # Return a copy to avoid modifying original
    if not dict2:
        return dict1.copy()

    if dict1.keys() != dict2.keys():
        raise ValueError("Cannot concatenate TensorDicts with different keys.")

    concatenated: TensorDict = {}
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]

        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            # If trailing dimensions differ (e.g., different sequence lengths),
            # pad the smaller tensor on the right to match the larger one.
            if val1.dim() >= 2 and val1.shape[1:] != val2.shape[1:]:
                # Currently handle only padding along the last dimension(s).
                # Compute the maximum size for each trailing dimension.
                max_trailing = tuple(
                    max(s1, s2) for s1, s2 in zip(val1.shape[1:], val2.shape[1:])
                )

                def _pad_to(
                    t: torch.Tensor, target_shape: tuple[int, ...]
                ) -> torch.Tensor:
                    # Only pad along the *last* dimension(s) (right‑side padding).
                    pad = []
                    # Build pad spec back‑to‑front
                    for current, target in reversed(
                        list(zip(t.shape[1:], target_shape))
                    ):
                        pad.extend([0, target - current])
                    if any(pad):
                        return f.pad(t, pad)
                    return t

                val1 = _pad_to(val1, max_trailing)
                val2 = _pad_to(val2, max_trailing)
            target_device = val1.device
            if val2.device != target_device:
                val2 = val2.to(target_device)
            concatenated[key] = torch.cat((val1, val2), dim=0)
        elif isinstance(val1, list) and isinstance(val2, list):
            # Harmonise devices for each tensor within the lists
            target_device = (
                val1[0].device
                if val1
                else (val2[0].device if val2 else torch.device("cpu"))
            )
            val1_conv = [
                t.to(target_device)
                if isinstance(t, torch.Tensor) and t.device != target_device
                else t
                for t in val1
            ]
            val2_conv = [
                t.to(target_device)
                if isinstance(t, torch.Tensor) and t.device != target_device
                else t
                for t in val2
            ]
            concatenated[key] = val1_conv + val2_conv
        else:
            raise TypeError(
                f"Type mismatch or unsupported type for key '{key}': "
                f"{type(val1)} and {type(val2)}"
            )
    return concatenated


def _slice_tensor_dict(tensor_dict: TensorDict, start: int, end: int) -> TensorDict:
    """Slices a TensorDict along the batch dimension.

    Args:
        tensor_dict: The TensorDict to slice.
        start: The starting index (inclusive).
        end: The ending index (exclusive).

    Returns:
        A new TensorDict containing the sliced data.
    """

    sliced: TensorDict = {}
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            sliced[key] = value[start:end]
        elif isinstance(value, list):
            sliced[key] = value[start:end]
        else:
            raise TypeError(f"Unsupported type for slicing key '{key}': {type(value)}")
    return sliced


def _get_tensor_dict_rows(tensor_dict: TensorDict) -> int:
    """Gets the number of rows (batch size) in a TensorDict.

    Args:
        tensor_dict: The TensorDict.

    Returns:
        The number of rows. Returns 0 for an empty dict.

    Raises:
        ValueError: If the dict is inconsistent or empty.
    """

    if not tensor_dict:
        return 0

    rows = -1
    for key, value in tensor_dict.items():
        current_rows: int
        if isinstance(value, torch.Tensor):
            current_rows = value.shape[0]
        elif isinstance(value, list):
            current_rows = len(value)
        else:
            raise TypeError(f"Unsupported type for key '{key}': {type(value)}")

        if rows == -1:
            rows = current_rows
        elif rows != current_rows:
            raise ValueError(
                f"Inconsistent number of rows found in TensorDict. "
                f"Key '{key}' has {current_rows}, expected {rows}."
            )

    if rows == -1:
        # This case should ideally not be reached if tensor_dict is not empty
        # and contains supported types, but added for robustness.
        raise ValueError("Could not determine row count from non-empty TensorDict.")
    return rows


def _record_batch_to_ipc_bytes(rb: pa.RecordBatch) -> bytes:
    """Wrap one RecordBatch in a self-contained IPC stream (schema + batch)."""

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, rb.schema) as writer:
        writer.write_batch(rb)
    return sink.getvalue().to_pybytes()


def _serialise_batches(
    batches: Iterable[pa.RecordBatch],
) -> Iterator[pa.RecordBatch]:
    """Arrow-to-Arrow UDF: emits one-row RecordBatches with IPC-encoded payloads."""

    for rb in batches:
        payload = _record_batch_to_ipc_bytes(rb)
        array = pa.array([payload], type=pa.binary())
        yield pa.RecordBatch.from_arrays([array], schema=_ARROW_BATCH_SCHEMA)


def _validate_df_schema(
    df: DataFrame, timestamp_to: str | None = None, tokenizer: Callable | None = None
) -> None:
    """Validate that df schema contains only types convertible to numpy/tensor.

    Args:
        df: PySpark DataFrame to validate.
        timestamp_to: Optional target dtype for Spark `TimestampType` columns.
            If None (default), timestamp columns are rejected. Supported values:
            "int64" (epoch-nanoseconds) or "float64" (epoch-seconds).
        tokenizer: Optional tokenizer to allow StringType columns.

    Raises:
        ValueError: If a column has unsupported type for tensor conversion.
    """

    allowed_array_element_types: tuple[type, ...] = (NumericType, BooleanType)
    if timestamp_to is not None:
        allowed_array_element_types += (TimestampType,)

    for field in df.schema.fields:
        dt = field.dataType
        if isinstance(dt, (NumericType, BooleanType)):
            continue
        if timestamp_to is not None and isinstance(dt, TimestampType):
            continue
        if isinstance(dt, StringType):
            if tokenizer is not None:
                continue
            raise ValueError("String columns require a tokenizer to be provided.")
        if isinstance(dt, ArrayType):
            element_type = dt.elementType
            if isinstance(element_type, allowed_array_element_types):
                continue

            # Example: Allow Array<Array<Numeric>>
            # if isinstance(element_type, ArrayType) and \
            #    isinstance(element_type.elementType, allowed_array_element_types):
            #      continue

        raise ValueError(
            f"Column '{field.name}' has unsupported type '{dt.simpleString()}' "
            f"for tensor conversion. Allowed scalar types: Numeric, Boolean. "
            f"Allowed array element types: {allowed_array_element_types} (currently one level deep)."
        )
