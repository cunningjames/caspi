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
    """Converts a PyArrow RecordBatch to a dictionary of PyTorch tensors.

    This function iterates through the columns of an Arrow RecordBatch,
    converting each column into a PyTorch tensor or a list of tensors based
    on the corresponding Spark data type defined in `spark_schema`. It handles
    various data types including numeric, boolean, timestamp, string (with
    optional tokenization), and arrays of primitives. Special care is taken
    to handle null values appropriately for each type.

    Args:
        record_batch (pa.RecordBatch): The input Arrow RecordBatch.
        spark_schema (StructType): The original Spark DataFrame schema, used to
            determine the correct data type for each column.
        tokenizer (Callable | None): An optional tokenizer function. If provided
            and a StringType column is encountered, this function is used to
            convert the strings into tokenized tensors. The resulting tensors
            (e.g., 'input_ids', 'attention_mask') are added to the output
            dictionary with keys prefixed by the original column name.

    Returns:
        TensorDict: A dictionary where keys are column names (or derived names
            for tokenized strings) and values are either `torch.Tensor` (for
            scalar types, timestamps, tokenized strings) or `list[torch.Tensor]`
            (for array types). Returns an empty dictionary if the input
            `record_batch` has zero rows.

    Raises:
        TypeError: If a column contains a data type that is not supported for
            conversion (e.g., complex structs, maps) or if an unexpected
            intermediate type (like NumPy object array for non-numeric/bool)
            is encountered.
        AssertionError: If a StringType column is present but `tokenizer` is None.
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
    """Concatenates two TensorDicts along the batch dimension (dim=0).

    Merges the tensors or lists of tensors from `dict2` into `dict1` for each
    corresponding key. It handles both `torch.Tensor` values (concatenated using
    `torch.cat`) and `list[torch.Tensor]` values (by extending the list).

    If tensors have differing trailing dimensions (e.g., sequence lengths in
    NLP tasks), it pads the shorter tensor along the trailing dimensions before
    concatenation. Device compatibility is handled by moving tensors from `dict2`
    to the device of the corresponding tensor in `dict1` if necessary.

    Args:
        dict1 (TensorDict): The first TensorDict. Can be empty, in which case
            a copy of `dict2` is returned.
        dict2 (TensorDict): The second TensorDict to concatenate onto the first.
            Can be empty, in which case a copy of `dict1` is returned.

    Returns:
        TensorDict: A new dictionary containing the concatenated tensors or
            lists of tensors.

    Raises:
        ValueError: If `dict1` and `dict2` are non-empty and have different
            sets of keys.
        TypeError: If the values associated with the same key in `dict1` and
            `dict2` have incompatible types (e.g., one is a Tensor and the
            other is a list) or are of an unsupported type for concatenation.
    """
    if not dict1:
        return dict2.copy()
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
    """Slices all tensors or lists within a TensorDict along the batch dimension.

    Creates a new TensorDict where each value (torch.Tensor or list) is a
    slice of the corresponding value in the input `tensor_dict`. The slice
    is taken along the first dimension (axis 0), representing the batch dimension.

    Args:
        tensor_dict (TensorDict): The dictionary of tensors or lists of tensors
            to slice.
        start (int): The starting index for the slice (inclusive).
        end (int): The ending index for the slice (exclusive).

    Returns:
        TensorDict: A new dictionary containing the sliced tensors or lists.

    Raises:
        TypeError: If any value in `tensor_dict` is not a `torch.Tensor` or a
            `list`.
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
    """Determines the number of rows (batch size) in a TensorDict.

    Checks the size of the first dimension of `torch.Tensor` values or the
    length of `list` values. It assumes all values in the dictionary represent
    the same batch and therefore should have the same size along the first
    dimension.

    Args:
        tensor_dict (TensorDict): The dictionary containing tensors or lists
            of tensors.

    Returns:
        int: The number of rows (batch size). Returns 0 if the input dictionary
             is empty.

    Raises:
        ValueError: If the `tensor_dict` is not empty but contains values with
            inconsistent first dimension sizes (batch sizes) or if it's non-empty
            but a row count cannot be determined (e.g., contains unsupported types).
        TypeError: If a value in the dictionary is not a `torch.Tensor` or `list`.
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
    """Serializes a single PyArrow RecordBatch into the Arrow IPC stream format.

    This creates a self-contained binary payload that includes both the schema
    and the data for the given RecordBatch. This format is suitable for sending
    the batch over a network or storing it.

    Args:
        rb (pa.RecordBatch): The PyArrow RecordBatch to serialize.

    Returns:
        bytes: A bytes object containing the serialized RecordBatch in Arrow
               IPC stream format.
    """
    sink = pa.BufferOutputStream()
    # Use new_stream to include schema with the batch
    with pa.ipc.new_stream(sink, rb.schema) as writer:
        writer.write_batch(rb)
    return sink.getvalue().to_pybytes()


def _serialise_batches(batches: Iterable[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
    """Serializes an iterable of RecordBatches for Spark's mapInArrow.

    This function takes an iterator yielding PyArrow RecordBatches (as provided
    by Spark's `mapInArrow` input function) and transforms it into an iterator
    yielding new RecordBatches. Each output RecordBatch contains a single row
    and a single column named "batch". The value in this column is the
    binary Arrow IPC stream representation of one of the input RecordBatches,
    generated by `_record_batch_to_ipc_bytes`.

    This structure is required by the `mapInArrow` pattern where the UDF must
    return an iterator of RecordBatches conforming to the specified output schema
    (`_ARROW_BATCH_SCHEMA`).

    Args:
        batches (Iterable[pa.RecordBatch]): An iterable of PyArrow RecordBatches
            as input from Spark.

    Yields:
        Iterator[pa.RecordBatch]: An iterator producing RecordBatches, each
            containing one row with a single binary column "batch" holding
            the serialized data of an input RecordBatch.
    """
    for rb in batches:
        # Serialize each input batch into IPC bytes
        payload = _record_batch_to_ipc_bytes(rb)
        # Create a new RecordBatch with one row, one column containing the payload
        array = pa.array([payload], type=pa.binary())
        yield pa.RecordBatch.from_arrays([array], schema=_ARROW_BATCH_SCHEMA)


def _to_device(tensor_dict: TensorDict, device: str | torch.device | None) -> TensorDict:
    """Moves all tensors in a TensorDict to the specified device.

    Iterates through a dictionary of tensors or lists of tensors and moves each
    element to the requested device. This function handles both direct torch.Tensor
    values and lists of torch.Tensor values.

    Args:
        tensor_dict (TensorDict): The dictionary containing tensors or lists of
            tensors to be moved to the device.
        device (str | torch.device | None): The target device to move tensors to.
            If None, the tensors will remain on their current devices.

    Returns:
        TensorDict: A new dictionary with the same keys but with all tensors
            moved to the specified device.

    Raises:
        TypeError: If any value in the dictionary is not a torch.Tensor or a
            list of torch.Tensor objects.
    """
    if device is None:
        return tensor_dict

    if isinstance(device, str):
        device = torch.device(device)

    result: TensorDict = {}
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, list):
            # Handle list of tensors
            result[key] = [
                t.to(device) if isinstance(t, torch.Tensor) else t
                for t in value
            ]
        else:
            raise TypeError(f"Unsupported type for moving to device: {type(value)}")

    return result


def _validate_df_schema(
    df: DataFrame, timestamp_to: str | None = None, tokenizer: Callable | None = None
) -> None:
    """Validates if a Spark DataFrame's schema is compatible with tensor conversion.

    Checks each column in the DataFrame's schema to ensure its data type is
    supported for conversion into PyTorch tensors by the `_arrow_batch_to_tensor_dict`
    function. Supported types include Numeric, Boolean, Timestamp (if `timestamp_to`
    is specified), String (if `tokenizer` is provided), and Arrays of these
    primitive types (currently only one level deep).

    Args:
        df (DataFrame): The PySpark DataFrame whose schema needs validation.
        timestamp_to (str | None): Specifies how to handle Spark `TimestampType`
            columns. If "int64", timestamps are converted to nanoseconds since
            epoch. If None (default), TimestampType columns are considered
            unsupported. Other values are currently not supported.
        tokenizer (Callable | None): A tokenizer function. If provided, columns
            of `StringType` are considered supported. If None, `StringType`
            columns are considered unsupported.

    Raises:
        ValueError: If any column in the DataFrame has a data type that is not
            supported according to the specified rules (e.g., StructType, MapType,
            StringType without a tokenizer, TimestampType without `timestamp_to`,
            nested arrays beyond one level, or arrays of unsupported types).
    """
    allowed_array_element_types: tuple[type, ...] = (NumericType, BooleanType)
    if timestamp_to == "int64":  # Be specific about supported conversion
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
