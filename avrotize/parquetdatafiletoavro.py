# coding: utf-8
"""
Convert Parquet data files to Avro Object Container File (OCF) data files.

Unlike :mod:`avrotize.parquettoavro`, which only converts a Parquet schema to
an Avro schema (JSON), this module performs a full data conversion. The Avro
schema is derived from the Parquet schema automatically and embedded into the
OCF header, so no user-supplied Avro schema is required.

The conversion is streamed row-group/batch by row-group/batch; whole Parquet
files are never loaded into memory.

Any Parquet/Arrow type that cannot be represented in Avro without information
loss (unions, durations, views, nanosecond time/timestamp values, non-string
map keys, unsigned 64-bit values outside the signed ``long`` range, ...)
raises :class:`ParquetToAvroDataError`. Data is never silently truncated or
dropped and a partially written output file is never left behind.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

from avrotize.common import avro_name

#: Maximum value of a signed Avro ``long`` (and therefore of a Parquet uint64
#: that can be converted losslessly).
_MAX_SIGNED_LONG = (1 << 63) - 1

#: Supported OCF compression codecs accepted by fastavro.
_SUPPORTED_CODECS = ("null", "deflate", "snappy", "zstandard", "bzip2", "xz")

#: Default number of rows processed per streamed batch.
_DEFAULT_BATCH_SIZE = 1024


class ParquetToAvroDataError(Exception):
    """Raised when a Parquet file cannot be converted to Avro OCF losslessly."""


# ---------------------------------------------------------------------------
# Schema conversion
# ---------------------------------------------------------------------------


class _SchemaBuilder:
    """Builds an Avro schema (JSON-serializable Python objects) from an Arrow
    schema, assigning unique names to all generated named types (records and
    fixed) and rejecting types that have no lossless Avro representation."""

    def __init__(self, namespace: str, root_name: str):
        self.namespace = namespace
        self.root_name = root_name
        self._used_names: set[str] = {root_name}

    def _unique_name(self, base: str) -> str:
        candidate = avro_name(base) or "_"
        if candidate not in self._used_names:
            self._used_names.add(candidate)
            return candidate
        suffix = 1
        while f"{candidate}_{suffix}" in self._used_names:
            suffix += 1
        unique = f"{candidate}_{suffix}"
        self._used_names.add(unique)
        return unique

    def build_field(self, field: pa.Field) -> Dict[str, Any]:
        avro_type = self.build_type(field.type, field.name)
        if field.nullable:
            if isinstance(avro_type, list):
                avro_type = ["null", *avro_type]
            else:
                avro_type = ["null", avro_type]
        avro_field: Dict[str, Any] = {"name": field.name, "type": avro_type}
        if field.metadata and b"description" in field.metadata:
            avro_field["doc"] = field.metadata[b"description"].decode("utf-8")
        return avro_field

    def build_type(self, parquet_type: pa.DataType, context_name: str) -> Any:
        # Dictionary encoding is a storage detail: the logical column has the
        # dictionary's value type.
        if pa.types.is_dictionary(parquet_type):
            return self.build_type(parquet_type.value_type, context_name)
        # Integer types
        if pa.types.is_int8(parquet_type) or pa.types.is_int16(parquet_type) or pa.types.is_int32(parquet_type):
            return "int"
        if pa.types.is_int64(parquet_type):
            return "long"
        if pa.types.is_uint8(parquet_type) or pa.types.is_uint16(parquet_type):
            # Max 65535 always fits into an Avro int.
            return "int"
        if pa.types.is_uint32(parquet_type):
            # Max 4294967295 always fits into an Avro long.
            return "long"
        if pa.types.is_uint64(parquet_type):
            # Range is validated per value during data conversion.
            return "long"
        # Floating point / boolean / string / bytes
        if pa.types.is_float32(parquet_type):
            return "float"
        if pa.types.is_float64(parquet_type):
            return "double"
        if pa.types.is_boolean(parquet_type):
            return "boolean"
        if pa.types.is_string(parquet_type) or pa.types.is_large_string(parquet_type):
            return "string"
        if pa.types.is_fixed_size_binary(parquet_type):
            return {
                "type": "fixed",
                "name": self._unique_name(f"{context_name}_fixed"),
                "namespace": self.namespace,
                "size": int(parquet_type.byte_width),
            }
        if pa.types.is_binary(parquet_type) or pa.types.is_large_binary(parquet_type):
            return "bytes"
        if pa.types.is_null(parquet_type):
            return "null"
        # Temporal types
        if pa.types.is_decimal(parquet_type):
            return {
                "type": "bytes",
                "logicalType": "decimal",
                "precision": int(parquet_type.precision),
                "scale": int(parquet_type.scale),
            }
        if pa.types.is_date32(parquet_type) or pa.types.is_date64(parquet_type):
            return {"type": "int", "logicalType": "date"}
        if pa.types.is_time32(parquet_type):
            # time32 is second or millisecond resolution.
            if parquet_type.unit == "ms":
                return {"type": "int", "logicalType": "time-millis"}
            if parquet_type.unit == "s":
                # Seconds are a subset of millisecond resolution.
                return {"type": "int", "logicalType": "time-millis"}
            raise ParquetToAvroDataError(
                f"Unsupported time32 unit '{parquet_type.unit}' for field '{context_name}'"
            )
        if pa.types.is_time64(parquet_type):
            if parquet_type.unit == "us":
                return {"type": "long", "logicalType": "time-micros"}
            raise ParquetToAvroDataError(
                f"Parquet time64 type with unit '{parquet_type.unit}' on field "
                f"'{context_name}' cannot be represented in Avro (Avro time types "
                "only support millisecond or microsecond resolution)"
            )
        if pa.types.is_timestamp(parquet_type):
            unit = parquet_type.unit
            tz_aware = parquet_type.tz is not None
            if unit == "s" or unit == "ms":
                logical = "timestamp-millis" if tz_aware else "local-timestamp-millis"
                return {"type": "long", "logicalType": logical}
            if unit == "us":
                logical = "timestamp-micros" if tz_aware else "local-timestamp-micros"
                return {"type": "long", "logicalType": logical}
            raise ParquetToAvroDataError(
                f"Parquet timestamp type with unit '{unit}' on field "
                f"'{context_name}' cannot be represented in Avro (nanosecond "
                "resolution would be truncated; Avro only supports millisecond "
                "and microsecond timestamp resolution)"
            )
        # Nested types
        if pa.types.is_list(parquet_type) or pa.types.is_large_list(parquet_type) or pa.types.is_fixed_size_list(parquet_type):
            item_type = self.build_type(parquet_type.value_type, context_name)
            if parquet_type.value_field.nullable:
                item_type = ["null", item_type]
            return {
                "type": "array",
                "items": item_type,
            }
        if pa.types.is_map(parquet_type):
            key_type = parquet_type.key_type
            if not pa.types.is_string(key_type) and not pa.types.is_large_string(key_type):
                raise ParquetToAvroDataError(
                    f"Parquet map field '{context_name}' has non-string key type "
                    f"'{key_type}'; Avro maps require string keys"
                )
            value_type = self.build_type(parquet_type.item_type, context_name)
            if parquet_type.item_field.nullable:
                value_type = ["null", value_type]
            return {
                "type": "map",
                "values": value_type,
            }
        if pa.types.is_struct(parquet_type):
            record_name = self._unique_name(f"{context_name}_type")
            fields = [self.build_field(nested) for nested in parquet_type]
            record: Dict[str, Any] = {
                "type": "record",
                "name": record_name,
                "namespace": self.namespace,
                "fields": fields,
            }
            return record
        raise ParquetToAvroDataError(
            f"Parquet/Arrow type '{parquet_type}' on field '{context_name}' has "
            "no lossless Avro representation"
        )


# ---------------------------------------------------------------------------
# Data conversion
# ---------------------------------------------------------------------------

# A value converter maps a Python object produced by ``Array.to_pylist`` (or
# ``None``) into an Avro-compatible Python object.
ValueConverter = Callable[[Any], Any]


def _identity(value: Any) -> Any:
    return value


def _make_uint64_converter(context_name: str) -> ValueConverter:
    def convert(value: Any) -> Any:
        if value is not None and value > _MAX_SIGNED_LONG:
            raise ParquetToAvroDataError(
                f"Unsigned 64-bit value {value} in field '{context_name}' exceeds "
                f"the signed Avro long range (max {_MAX_SIGNED_LONG}) and cannot "
                "be converted without overflow"
            )
        return value

    return convert


def _make_struct_converter(field_converters: Sequence[Tuple[str, ValueConverter]]) -> ValueConverter:
    def convert(value: Any) -> Any:
        if value is None:
            return None
        return {name: conv(value.get(name)) for name, conv in field_converters}

    return convert


def _make_list_converter(item_converter: ValueConverter) -> ValueConverter:
    def convert(value: Any) -> Any:
        if value is None:
            return None
        return [item_converter(item) for item in value]

    return convert


def _make_map_converter(context_name: str, value_converter: ValueConverter) -> ValueConverter:
    def convert(value: Any) -> Any:
        if value is None:
            return None
        result: Dict[str, Any] = {}
        for entry in value:
            # Arrow maps are returned as a list of (key, value) tuples.
            key, item = entry
            if not isinstance(key, str):
                raise ParquetToAvroDataError(
                    f"Parquet map field '{context_name}' contains a non-string "
                    f"key {key!r}; Avro maps require string keys"
                )
            result[key] = value_converter(item)
        return result

    return convert


def _build_converter(parquet_type: pa.DataType, context_name: str) -> ValueConverter:
    """Create a converter for non-null values of ``parquet_type``."""
    if pa.types.is_uint64(parquet_type):
        return _make_uint64_converter(context_name)
    if pa.types.is_list(parquet_type) or pa.types.is_large_list(parquet_type) or pa.types.is_fixed_size_list(parquet_type):
        return _make_list_converter(_build_converter(parquet_type.value_type, context_name))
    if pa.types.is_map(parquet_type):
        return _make_map_converter(context_name, _build_converter(parquet_type.item_type, context_name))
    if pa.types.is_struct(parquet_type):
        child_converters = [
            (nested.name, _build_field_converter(nested)) for nested in parquet_type
        ]
        return _make_struct_converter(child_converters)
    if pa.types.is_dictionary(parquet_type):
        # Dictionary-encoded arrays are decoded by to_pylist; convert based on
        # the decoded value type.
        return _build_converter(parquet_type.value_type, context_name)
    # All scalar types with direct Avro representations (int/long/float/double/
    # bool/str/bytes, date, time(ms/us), timestamp(ms/us), decimal, fixed,
    # null) yield Python objects fastavro accepts unchanged.
    return _identity


def _build_field_converter(field: pa.Field) -> ValueConverter:
    inner = _build_converter(field.type, field.name)

    if not field.nullable:
        return inner

    def nullable_convert(value: Any) -> Any:
        if value is None:
            return None
        return inner(value)

    return nullable_convert


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_parquet_data_to_avro(
    parquet_file_path: str,
    avro_ocf_path: str,
    namespace: str = "",
    batch_size: int = _DEFAULT_BATCH_SIZE,
    codec: str = "null",
) -> None:
    """Convert a Parquet data file into an Avro Object Container File (OCF).

    The Avro schema is generated from the Parquet schema and embedded into the
    OCF header. Data is streamed one batch (at most ``batch_size`` rows) at a
    time, so memory use is bounded regardless of the source file size.

    :param parquet_file_path: Path to the source Parquet file.
    :param avro_ocf_path: Path of the ``.avro`` OCF file to write.
    :param namespace: Optional Avro namespace for generated records.
    :param batch_size: Maximum number of rows converted per batch.
    :param codec: OCF compression codec (``null``, ``deflate``, ``snappy``,
        ``zstandard``, ``bzip2`` or ``xz``).
    :raises FileNotFoundError: if the Parquet file does not exist.
    :raises ParquetToAvroDataError: if the data cannot be converted losslessly.
    """
    if not os.path.exists(parquet_file_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_file_path}")
    if batch_size is None or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if codec not in _SUPPORTED_CODECS:
        raise ValueError(
            f"Unsupported Avro OCF codec '{codec}'. Supported codecs: "
            + ", ".join(_SUPPORTED_CODECS)
        )

    # Imported here so that the module remains importable in environments
    # without fastavro for the existing schema-only tooling.
    import fastavro

    parquet_file = pq.ParquetFile(parquet_file_path)
    arrow_schema = parquet_file.schema_arrow

    root_name = avro_name(os.path.basename(parquet_file_path).split(".")[0])
    builder = _SchemaBuilder(namespace=namespace, root_name=root_name)
    avro_schema: Dict[str, Any] = {
        "type": "record",
        "name": root_name,
        "namespace": namespace,
        "fields": [builder.build_field(field) for field in arrow_schema],
    }
    parsed_schema = fastavro.parse_schema(avro_schema)

    field_converters = [
        (field.name, _build_field_converter(field)) for field in arrow_schema
    ]

    def convert_batch(batch: pa.RecordBatch) -> List[Dict[str, Any]]:
        columns = {name: batch.column(i).to_pylist() for i, (name, _) in enumerate(field_converters)}
        records: List[Dict[str, Any]] = []
        for row_index in range(batch.num_rows):
            record: Dict[str, Any] = {}
            for name, converter in field_converters:
                record[name] = converter(columns[name][row_index])
            records.append(record)
        return records

    output_dir = os.path.dirname(os.path.abspath(avro_ocf_path))
    os.makedirs(output_dir, exist_ok=True)

    # Write to a temporary file first and atomically replace the target only
    # after a complete, successful conversion, so an error never leaves a
    # truncated/corrupt OCF behind.
    temp_fd, temp_path = tempfile.mkstemp(prefix=".avrotize-pq2ocf-", suffix=".avro", dir=output_dir)
    try:
        with os.fdopen(temp_fd, "wb") as output_stream:
            def record_stream():
                for batch in parquet_file.iter_batches(batch_size=batch_size):
                    for record in convert_batch(batch):
                        yield record

            # fastavro accepts an empty iterator and still emits a valid OCF
            # (header + schema, zero blocks), covering zero-row Parquet files.
            fastavro.writer(
                output_stream,
                parsed_schema,
                record_stream(),
                codec=codec,
            )
        os.replace(temp_path, avro_ocf_path)
    except BaseException:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
