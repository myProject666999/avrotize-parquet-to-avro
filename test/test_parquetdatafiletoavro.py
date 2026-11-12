#  pylint: disable=missing-function-docstring, wrong-import-order, wrong-import-position, missing-module-docstring, missing-class-docstring

"""Tests for Parquet data file -> Avro OCF data file conversion.

The suite writes Parquet files into local temporary directories (no external
services) and reads the produced OCF back with fastavro, asserting row-by-row
equivalence with the source data across the whole supported type system.
"""

import datetime
import decimal
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import fastavro
import pyarrow as pa
import pyarrow.parquet as pq

from avrotize.parquetdatafiletoavro import (
    ParquetToAvroDataError,
    convert_parquet_data_to_avro,
)

ROOT = Path(__file__).resolve().parents[1]
NS = "example.pq2ocf"
UTC = datetime.timezone.utc


def normalize_table_row(row, schema):
    """Normalize a row from ``Table.to_pylist`` into the shape fastavro
    returns. Arrow represents maps as lists of ``(key, value)`` tuples (empty
    maps as empty lists), so the Arrow schema is needed to distinguish an empty
    map from an empty array. Timezones are canonicalized to
    ``datetime.timezone.utc`` (Arrow uses ``zoneinfo.ZoneInfo('UTC')``, which is
    semantically identical). Everything else is structurally identical.
    """
    result = {}
    for field in schema:
        result[field.name] = _normalize_typed(row[field.name], field.type)
    return result


def _normalize_typed(value, arrow_type):
    if value is None:
        return None
    if isinstance(value, datetime.datetime) and value.tzinfo is not None:
        return value.astimezone(UTC)
    if pa.types.is_map(arrow_type):
        return {key: _normalize_typed(item, arrow_type.item_type) for key, item in value}
    if pa.types.is_struct(arrow_type):
        return {
            nested.name: _normalize_typed(value.get(nested.name), nested.type)
            for nested in arrow_type
        }
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
        return [_normalize_typed(entry, arrow_type.value_type) for entry in value]
    return value


class ParquetDataToAvroTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def write_parquet(self, name, schema, rows, *, row_group_size=None):
        path = self.tmp / name
        table = pa.Table.from_pylist(rows, schema=schema)
        if row_group_size is not None:
            pq.write_table(table, path, row_group_size=row_group_size)
        else:
            pq.write_table(table, path)
        return path

    def convert(self, parquet_path, *, namespace=NS, **kwargs):
        avro_path = parquet_path.with_suffix(".avro")
        convert_parquet_data_to_avro(str(parquet_path), str(avro_path), namespace=namespace, **kwargs)
        return avro_path

    def read_ocf(self, avro_path):
        with open(avro_path, "rb") as handle:
            reader = fastavro.reader(handle)
            return reader.writer_schema, list(reader)

    def assert_round_trip_equivalent(self, schema, rows, *, row_group_size=None, **kwargs):
        parquet_path = self.write_parquet("data.parquet", schema, rows, row_group_size=row_group_size)
        avro_path = self.convert(parquet_path, **kwargs)
        _, records = self.read_ocf(avro_path)
        expected = [normalize_table_row(row, schema) for row in rows]
        self.assertEqual(records, expected)
        return records

    def assert_avro_matches_source(self, parquet_path, avro_path):
        """Assert OCF records are row-by-row equivalent to the Parquet source,
        normalizing Arrow's map tuple-list representation into Avro dicts."""
        table = pq.read_table(parquet_path)
        source = table.to_pylist()
        _, records = self.read_ocf(avro_path)
        expected = [normalize_table_row(row, table.schema) for row in source]
        self.assertEqual(records, expected)
        return records


class TestScalarTypes(ParquetDataToAvroTestBase):
    def test_all_basic_scalar_types_round_trip(self):
        schema = pa.schema([
            pa.field("f_int8", pa.int8()),
            pa.field("f_int16", pa.int16()),
            pa.field("f_int32", pa.int32()),
            pa.field("f_int64", pa.int64()),
            pa.field("f_uint8", pa.uint8()),
            pa.field("f_uint16", pa.uint16()),
            pa.field("f_uint32", pa.uint32()),
            pa.field("f_uint64", pa.uint64()),
            pa.field("f_float32", pa.float32()),
            pa.field("f_float64", pa.float64()),
            pa.field("f_boolean", pa.bool_()),
            pa.field("f_string", pa.string()),
            pa.field("f_large_string", pa.large_string()),
        ])
        rows = [
            {
                "f_int8": -8, "f_int16": -16000, "f_int32": -32000, "f_int64": -6_400_000_000,
                "f_uint8": 255, "f_uint16": 65535, "f_uint32": 4_294_967_295,
                "f_uint64": 2**63 - 1, "f_float32": 1.25, "f_float64": 3.125,
                "f_boolean": True, "f_string": "plain", "f_large_string": "large",
            },
            {name: None for name in schema.names},
        ]
        records = self.assert_round_trip_equivalent(schema, rows)
        # Exact integer boundaries survive without truncation.
        self.assertEqual(records[0]["f_uint32"], 4_294_967_295)
        self.assertEqual(records[0]["f_uint64"], 2**63 - 1)

    def test_binary_and_fixed_size_binary_round_trip(self):
        schema = pa.schema([
            pa.field("f_binary", pa.binary()),
            pa.field("f_large_binary", pa.large_binary()),
            pa.field("f_fixed", pa.binary(5)),
        ])
        rows = [
            {"f_binary": b"\x00\x01\x02\xff", "f_large_binary": b"large-bytes", "f_fixed": b"abcde"},
            {"f_binary": None, "f_large_binary": None, "f_fixed": None},
        ]
        records = self.assert_round_trip_equivalent(schema, rows)
        avro_schema, _ = self.read_ocf(self.tmp / "data.avro")
        field_types = {field["name"]: field["type"] for field in avro_schema["fields"]}
        # fixed-size binary becomes an Avro fixed with the exact byte width.
        fixed = field_types["f_fixed"][1]
        self.assertEqual(fixed["type"], "fixed")
        self.assertEqual(fixed["size"], 5)
        self.assertEqual(records[0]["f_fixed"], b"abcde")
        self.assertEqual(field_types["f_binary"][1], "bytes")

    def test_dictionary_encoded_column_is_decoded(self):
        schema = pa.schema([pa.field("f_dict", pa.dictionary(pa.int8(), pa.string()))])
        rows = [{"f_dict": "alpha"}, {"f_dict": "beta"}, {"f_dict": "alpha"}, {"f_dict": None}]
        records = self.assert_round_trip_equivalent(schema, rows)
        self.assertEqual([row["f_dict"] for row in records], ["alpha", "beta", "alpha", None])


class TestTemporalAndDecimalTypes(ParquetDataToAvroTestBase):
    def test_decimal_precision_and_scale_preserved(self):
        schema = pa.schema([
            pa.field("d128", pa.decimal128(10, 2)),
            pa.field("d256", pa.decimal256(40, 5)),
            pa.field("d0", pa.decimal128(5, 0)),
        ])
        rows = [
            {
                "d128": decimal.Decimal("12345678.90"),
                "d256": decimal.Decimal("12345678901234567890123456789012345.12345"),
                "d0": decimal.Decimal("42"),
            },
            {"d128": None, "d256": None, "d0": None},
        ]
        records = self.assert_round_trip_equivalent(schema, rows)
        self.assertEqual(records[0]["d128"], decimal.Decimal("12345678.90"))
        self.assertEqual(records[0]["d256"], decimal.Decimal("12345678901234567890123456789012345.12345"))
        avro_schema, _ = self.read_ocf(self.tmp / "data.avro")
        dec_field = next(f for f in avro_schema["fields"] if f["name"] == "d128")["type"][1]
        self.assertEqual(dec_field, {"type": "bytes", "logicalType": "decimal", "precision": 10, "scale": 2})

    def test_date_types_map_to_avro_date(self):
        schema = pa.schema([
            pa.field("d32", pa.date32()),
            pa.field("d64", pa.date64()),
        ])
        rows = [
            {"d32": datetime.date(2024, 2, 29), "d64": datetime.datetime(1970, 1, 1)},
            {"d32": None, "d64": None},
        ]
        parquet_path = self.write_parquet("dates.parquet", schema, rows)
        avro_path = self.convert(parquet_path)
        avro_schema, records = self.read_ocf(avro_path)
        # Both Parquet date flavors map to the Avro date logical type, whose
        # canonical read-back is datetime.date (date64's midnight time component
        # carries no information beyond the calendar date).
        self.assertEqual(records[0]["d32"], datetime.date(2024, 2, 29))
        self.assertEqual(records[0]["d64"], datetime.date(1970, 1, 1))
        for name in ("d32", "d64"):
            field = next(f for f in avro_schema["fields"] if f["name"] == name)["type"][1]
            self.assertEqual(field, {"type": "int", "logicalType": "date"})
        self.assertIsNone(records[1]["d32"])
        self.assertIsNone(records[1]["d64"])

    def test_time_types_map_to_time_millis_and_micros(self):
        schema = pa.schema([
            pa.field("t32_s", pa.time32("s")),
            pa.field("t32_ms", pa.time32("ms")),
            pa.field("t64_us", pa.time64("us")),
        ])
        rows = [
            {
                "t32_s": datetime.time(3, 4, 5),
                "t32_ms": datetime.time(3, 4, 5, 6000),
                "t64_us": datetime.time(3, 4, 5, 678901),
            },
            {"t32_s": None, "t32_ms": None, "t64_us": None},
        ]
        records = self.assert_round_trip_equivalent(schema, rows)
        self.assertEqual(records[0]["t32_s"], datetime.time(3, 4, 5))
        self.assertEqual(records[0]["t32_ms"], datetime.time(3, 4, 5, 6000))
        self.assertEqual(records[0]["t64_us"], datetime.time(3, 4, 5, 678901))
        avro_schema, _ = self.read_ocf(self.tmp / "data.avro")
        logical = {
            f["name"]: f["type"][1]["logicalType"] for f in avro_schema["fields"]
        }
        self.assertEqual(logical, {"t32_s": "time-millis", "t32_ms": "time-millis", "t64_us": "time-micros"})

    def test_timestamp_millis_and_micros_with_and_without_timezone(self):
        schema = pa.schema([
            pa.field("ts_ms_naive", pa.timestamp("ms")),
            pa.field("ts_us_naive", pa.timestamp("us")),
            pa.field("ts_ms_tz", pa.timestamp("ms", tz="UTC")),
            pa.field("ts_us_tz", pa.timestamp("us", tz="UTC")),
        ])
        rows = [
            {
                "ts_ms_naive": datetime.datetime(2024, 1, 2, 3, 4, 5, 123000),
                "ts_us_naive": datetime.datetime(2024, 1, 2, 3, 4, 5, 678901),
                "ts_ms_tz": datetime.datetime(2024, 1, 2, 3, 4, 5, 123000, tzinfo=UTC),
                "ts_us_tz": datetime.datetime(2024, 1, 2, 3, 4, 5, 678901, tzinfo=UTC),
            },
            {name: None for name in schema.names},
        ]
        records = self.assert_round_trip_equivalent(schema, rows)
        first = records[0]
        self.assertEqual(first["ts_ms_naive"], datetime.datetime(2024, 1, 2, 3, 4, 5, 123000))
        self.assertEqual(first["ts_us_naive"], datetime.datetime(2024, 1, 2, 3, 4, 5, 678901))
        self.assertEqual(first["ts_ms_tz"], datetime.datetime(2024, 1, 2, 3, 4, 5, 123000, tzinfo=UTC))
        self.assertEqual(first["ts_us_tz"], datetime.datetime(2024, 1, 2, 3, 4, 5, 678901, tzinfo=UTC))
        avro_schema, _ = self.read_ocf(self.tmp / "data.avro")
        logical = {f["name"]: f["type"][1]["logicalType"] for f in avro_schema["fields"]}
        self.assertEqual(logical, {
            "ts_ms_naive": "local-timestamp-millis",
            "ts_us_naive": "local-timestamp-micros",
            "ts_ms_tz": "timestamp-millis",
            "ts_us_tz": "timestamp-micros",
        })

    def test_timezone_aware_timestamps_are_normalized_to_utc(self):
        schema = pa.schema([pa.field("ts", pa.timestamp("us", tz="+01:00"))])
        rows = [{"ts": datetime.datetime(2024, 1, 2, 4, 4, 5, 678901, tzinfo=datetime.timezone(datetime.timedelta(hours=1)))}]
        records = self.assert_round_trip_equivalent(schema, rows)
        self.assertEqual(records[0]["ts"], datetime.datetime(2024, 1, 2, 3, 4, 5, 678901, tzinfo=UTC))


class TestNestedTypes(ParquetDataToAvroTestBase):
    def test_nested_struct_round_trip(self):
        schema = pa.schema([
            pa.field("order", pa.struct([
                pa.field("id", pa.int64(), nullable=False),
                pa.field("customer", pa.struct([
                    pa.field("name", pa.string()),
                    pa.field("tier", pa.int8()),
                ])),
                pa.field("tags", pa.list_(pa.string())),
            ])),
        ])
        rows = [
            {"order": {"id": 1, "customer": {"name": "ada", "tier": 3}, "tags": ["x", None, "z"]}},
            {"order": None},
            {"order": {"id": 2, "customer": None, "tags": []}},
        ]
        records = self.assert_round_trip_equivalent(schema, rows)
        self.assertEqual(records[0]["order"]["customer"], {"name": "ada", "tier": 3})
        self.assertIsNone(records[1]["order"])
        self.assertEqual(records[2]["order"]["tags"], [])

    def test_list_types_including_nested_and_null_elements(self):
        schema = pa.schema([
            pa.field("ints", pa.list_(pa.int32())),
            pa.field("nested", pa.list_(pa.list_(pa.int32()))),
            pa.field("fixed", pa.list_(pa.int32(), 3)),
            pa.field("large", pa.large_list(pa.string())),
        ])
        rows = [
            {"ints": [1, None, 3], "nested": [[1, 2], None, []], "fixed": [4, 5, 6], "large": ["a", None]},
            {"ints": None, "nested": None, "fixed": None, "large": None},
        ]
        self.assert_round_trip_equivalent(schema, rows)

    def test_map_type_round_trip(self):
        schema = pa.schema([
            pa.field("scores", pa.map_(pa.string(), pa.int64())),
            pa.field("nested_map", pa.map_(pa.string(), pa.list_(pa.string()))),
        ])
        rows = [
            {"scores": [("alice", 9), ("bob", None)], "nested_map": [("k", ["a", "b"]), ("empty", [])]},
            {"scores": None, "nested_map": None},
            {"scores": [], "nested_map": []},
        ]
        parquet_path = self.write_parquet("maps.parquet", schema, rows)
        avro_path = self.convert(parquet_path)
        records = self.assert_avro_matches_source(parquet_path, avro_path)
        self.assertEqual(records[0]["scores"], {"alice": 9, "bob": None})
        self.assertEqual(records[0]["nested_map"], {"k": ["a", "b"], "empty": []})
        # Empty Arrow maps convert to empty Avro maps.
        self.assertEqual(records[2], {"scores": {}, "nested_map": {}})
        # Null maps stay null.
        self.assertEqual(records[1], {"scores": None, "nested_map": None})

    def test_deeply_nested_list_map_struct_round_trip(self):
        inner = pa.struct([
            pa.field("x", pa.int32()),
            pa.field("y", pa.timestamp("us", tz="UTC")),
        ])
        deep_type = pa.list_(pa.map_(pa.string(), pa.list_(inner)))
        schema = pa.schema([pa.field("deep", deep_type)])
        ts = datetime.datetime(2025, 6, 7, 8, 9, 10, 654321, tzinfo=UTC)
        struct_arr = pa.array(
            [{"x": 1, "y": ts}, {"x": 2, "y": None}, {"x": 3, "y": ts}], type=inner
        )
        map_values = pa.ListArray.from_arrays(pa.array([0, 2, 3]), struct_arr)
        keys = pa.array(["k0", "k1"], pa.string())
        map_arr = pa.MapArray.from_arrays(pa.array([0, 1, 2, 2]), keys, map_values)
        outer = pa.ListArray.from_arrays(pa.array([0, 2, 3, 3]), map_arr)
        parquet_path = self.tmp / "deep.parquet"
        pq.write_table(
            pa.Table.from_arrays([outer], schema=schema), parquet_path, row_group_size=2
        )
        source = pq.read_table(parquet_path).to_pylist()
        avro_path = self.convert(parquet_path)
        _, records = self.read_ocf(avro_path)
        expected = [normalize_table_row(row, schema) for row in source]
        self.assertEqual(records, expected)

    def test_generated_record_names_are_unique_and_valid(self):
        schema = pa.schema([
            pa.field("a-b", pa.struct([pa.field("c.d", pa.int32())])),
            pa.field("a_b", pa.struct([pa.field("e", pa.int32())])),
        ])
        rows = [{"a-b": {"c.d": 1}, "a_b": {"e": 2}}]
        parquet_path = self.write_parquet("names.parquet", schema, rows)
        avro_path = self.convert(parquet_path)
        avro_schema, _ = self.read_ocf(avro_path)
        # fastavro.parse_schema would have raised on duplicate/invalid names.
        names = [field["name"] for field in avro_schema["fields"]]
        self.assertEqual(names, ["a-b", "a_b"])


class TestNullability(ParquetDataToAvroTestBase):
    def test_nullable_and_required_fields_use_unions_correctly(self):
        schema = pa.schema([
            pa.field("optional", pa.string(), nullable=True),
            pa.field("required", pa.string(), nullable=False),
            pa.field("struct_opt", pa.struct([pa.field("v", pa.int32(), nullable=False)]), nullable=True),
        ])
        rows = [
            {"optional": None, "required": "v", "struct_opt": {"v": 1}},
            {"optional": "x", "required": "y", "struct_opt": None},
        ]
        parquet_path = self.write_parquet("nulls.parquet", schema, rows)
        avro_path = self.convert(parquet_path)
        avro_schema, records = self.read_ocf(avro_path)
        fields = {f["name"]: f["type"] for f in avro_schema["fields"]}
        self.assertEqual(fields["optional"], ["null", "string"])
        self.assertEqual(fields["required"], "string")
        self.assertIsNone(records[0]["optional"])
        self.assertIsNone(records[1]["struct_opt"])
        self.assertEqual(records, [
            {"optional": None, "required": "v", "struct_opt": {"v": 1}},
            {"optional": "x", "required": "y", "struct_opt": None},
        ])

    def test_null_type_column_round_trips(self):
        schema = pa.schema([pa.field("only_null", pa.null())])
        records = self.assert_round_trip_equivalent(schema, [{"only_null": None}, {"only_null": None}])
        self.assertTrue(all(row["only_null"] is None for row in records))


class TestStreamingAndEmptyFiles(ParquetDataToAvroTestBase):
    def test_multiple_row_groups_stream_in_order(self):
        schema = pa.schema([pa.field("id", pa.int64()), pa.field("tag", pa.string())])
        rows = [{"id": i, "tag": f"tag-{i}"} for i in range(2500)]
        parquet_path = self.write_parquet("rg.parquet", schema, rows, row_group_size=100)
        self.assertEqual(pq.ParquetFile(parquet_path).num_row_groups, 25)
        avro_path = self.convert(parquet_path, batch_size=64)
        _, records = self.read_ocf(avro_path)
        self.assertEqual(len(records), 2500)
        self.assertEqual([row["id"] for row in records], list(range(2500)))
        self.assertEqual(records, pq.read_table(parquet_path).to_pylist())

    def test_batches_smaller_and_larger_than_row_groups(self):
        schema = pa.schema([pa.field("id", pa.int64())])
        rows = [{"id": i} for i in range(100)]
        for batch_size in (1, 37, 10_000):
            parquet_path = self.write_parquet(f"b{batch_size}.parquet", schema, rows, row_group_size=10)
            avro_path = self.convert(parquet_path, batch_size=batch_size)
            _, records = self.read_ocf(avro_path)
            self.assertEqual([row["id"] for row in records], list(range(100)))

    def test_empty_parquet_file_produces_valid_empty_ocf(self):
        schema = pa.schema([
            pa.field("id", pa.int32()),
            pa.field("nested", pa.struct([pa.field("v", pa.string())])),
        ])
        parquet_path = self.write_parquet("empty.parquet", schema, [])
        avro_path = self.convert(parquet_path)
        with open(avro_path, "rb") as handle:
            self.assertEqual(handle.read(4), b"Obj\x01")
        avro_schema, records = self.read_ocf(avro_path)
        self.assertEqual(avro_schema["fields"][0]["name"], "id")
        self.assertEqual(records, [])

    def test_compression_codecs_produce_valid_ocf(self):
        schema = pa.schema([pa.field("id", pa.int64())])
        rows = [{"id": i} for i in range(50)]
        for codec in ("null", "deflate"):
            parquet_path = self.write_parquet(f"{codec}.parquet", schema, rows)
            avro_path = self.convert(parquet_path, codec=codec)
            with open(avro_path, "rb") as handle:
                reader = fastavro.reader(handle)
                self.assertEqual(reader.metadata["avro.codec"], codec)
                self.assertEqual(len(list(reader)), 50)


class TestErrorPaths(ParquetDataToAvroTestBase):
    def _expect_error(self, schema, rows, message_fragment, *, name="err.parquet"):
        parquet_path = self.write_parquet(name, schema, rows)
        avro_path = self.tmp / "err.avro"
        with self.assertRaises(ParquetToAvroDataError) as context:
            convert_parquet_data_to_avro(str(parquet_path), str(avro_path))
        self.assertIn(message_fragment, str(context.exception))
        self.assertFalse(avro_path.exists(), "failed conversion must not leave an output file")

    def test_uint64_above_signed_long_is_rejected(self):
        self._expect_error(
            pa.schema([pa.field("v", pa.uint64())]),
            [{"v": 2**63}],
            "signed Avro long range",
        )

    def test_uint64_overflow_inside_nested_list_is_rejected(self):
        self._expect_error(
            pa.schema([pa.field("v", pa.list_(pa.uint64()))]),
            [{"v": [1, 2**63]}],
            "signed Avro long range",
            name="nested-err.parquet",
        )

    def test_non_string_map_key_is_rejected(self):
        schema = pa.schema([pa.field("m", pa.map_(pa.int32(), pa.string()))])
        parquet_path = self.tmp / "mk.parquet"
        keys = pa.array([1], pa.int32())
        values = pa.array(["a"], pa.string())
        map_arr = pa.MapArray.from_arrays(pa.array([0, 1]), keys, values)
        pq.write_table(pa.Table.from_arrays([map_arr], schema=schema), parquet_path)
        avro_path = self.tmp / "mk.avro"
        with self.assertRaises(ParquetToAvroDataError) as context:
            convert_parquet_data_to_avro(str(parquet_path), str(avro_path))
        self.assertIn("non-string key", str(context.exception))
        self.assertFalse(avro_path.exists())

    def test_nanosecond_timestamp_is_rejected(self):
        self._expect_error(
            pa.schema([pa.field("v", pa.timestamp("ns"))]), [], "nanosecond"
        )

    def test_nanosecond_time_is_rejected(self):
        self._expect_error(
            pa.schema([pa.field("v", pa.time64("ns"))]), [], "millisecond or microsecond"
        )

    def test_unrepresentable_arrow_types_are_rejected(self):
        cases = [
            (pa.duration("us"), "no lossless Avro representation"),
            (pa.float16(), "no lossless Avro representation"),
            (pa.list_view(pa.int32()), "no lossless Avro representation"),
            (pa.string_view(), "no lossless Avro representation"),
            (pa.binary_view(), "no lossless Avro representation"),
        ]
        for index, (arrow_type, fragment) in enumerate(cases):
            with self.subTest(arrow_type=str(arrow_type)):
                self._expect_error(
                    pa.schema([pa.field("v", arrow_type)]),
                    [],
                    fragment,
                    name=f"unsupported-{index}.parquet",
                )

    def test_missing_input_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            convert_parquet_data_to_avro(str(self.tmp / "missing.parquet"), str(self.tmp / "x.avro"))

    def test_invalid_arguments_raise_value_error(self):
        parquet_path = self.write_parquet(
            "ok.parquet", pa.schema([pa.field("x", pa.int32())]), [{"x": 1}]
        )
        with self.assertRaises(ValueError):
            convert_parquet_data_to_avro(str(parquet_path), str(self.tmp / "a.avro"), batch_size=0)
        with self.assertRaises(ValueError):
            convert_parquet_data_to_avro(str(parquet_path), str(self.tmp / "b.avro"), codec="bogus")


class TestEmbeddedSchemaAndCli(ParquetDataToAvroTestBase):
    def test_avro_schema_is_embedded_in_ocf_header(self):
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        parquet_path = self.write_parquet("embed.parquet", schema, [{"id": 1}])
        avro_path = self.convert(parquet_path, namespace="header.test")
        with open(avro_path, "rb") as handle:
            reader = fastavro.reader(handle)
            raw_schema = json.loads(reader.metadata["avro.schema"])
        self.assertEqual(raw_schema["type"], "record")
        # fastavro serializes the parsed schema, so the namespace may be
        # expressed either explicitly or merged into the record full name.
        self.assertEqual(
            raw_schema.get("namespace", "") + ("." if raw_schema.get("namespace") else "") + raw_schema["name"],
            "header.test.embed",
        )
        self.assertEqual(raw_schema["fields"], [{"name": "id", "type": "long"}])
        # The schema is embedded in the OCF header and parseable on its own,
        # i.e. no external schema supply is needed by readers.
        parsed = fastavro.parse_schema(raw_schema)
        self.assertEqual(parsed["name"], "header.test.embed")

    def test_field_description_metadata_becomes_avro_doc(self):
        schema = pa.schema([
            pa.field("documented", pa.string(), metadata={b"description": "café ❤"})
        ])
        parquet_path = self.write_parquet("doc.parquet", schema, [{"documented": "x"}])
        avro_path = self.convert(parquet_path)
        avro_schema, _ = self.read_ocf(avro_path)
        field = avro_schema["fields"][0]
        self.assertEqual(field["doc"], "café ❤")

    def test_cli_invocation_writes_readable_ocf(self):
        schema = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("when", pa.timestamp("us", tz="UTC")),
        ])
        rows = [
            {"id": i, "when": datetime.datetime(2024, 1, 1, tzinfo=UTC)}
            for i in range(5)
        ]
        parquet_path = self.write_parquet("cli.parquet", schema, rows, row_group_size=2)
        avro_path = self.tmp / "cli.avro"
        result = subprocess.run(
            [
                sys.executable, "-m", "avrotize", "pq2ocf",
                str(parquet_path), "--out", str(avro_path),
                "--namespace", NS, "--batch-size", "2", "--codec", "deflate",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        avro_schema, records = self.read_ocf(avro_path)
        self.assertEqual(len(records), 5)
        self.assertEqual(records[0]["id"], 0)
        self.assertEqual(avro_schema["fields"][1]["type"][1]["logicalType"], "timestamp-micros")

    def test_cli_error_returns_nonzero_exit_and_no_output(self):
        parquet_path = self.write_parquet(
            "cli-bad.parquet", pa.schema([pa.field("v", pa.uint64())]), [{"v": 2**63}]
        )
        avro_path = self.tmp / "cli-bad.avro"
        result = subprocess.run(
            [sys.executable, "-m", "avrotize", "pq2ocf", str(parquet_path), "--out", str(avro_path)],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("signed Avro long range", result.stdout + result.stderr)
        self.assertFalse(avro_path.exists())


if __name__ == "__main__":
    unittest.main()
