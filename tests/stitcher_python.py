import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np


def load_stitcher():
    path = Path(__file__).resolve().parents[1] / "tools/stitcher.py"
    spec = importlib.util.spec_from_file_location("stitcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stitcher = load_stitcher()

SYNCTIME = 1700000000
STRIDE = 8
SAMPLE_PERIOD = 4e-6


def write_band(path, *, band, fch1, foff, start, rows, channels=4, ifs=2, ibeam=1,
               synctime=SYNCTIME, stride=STRIDE, masked=(), grid=True):
    with h5py.File(path, "w") as handle:
        handle.attrs.create("CLASS", np.bytes_(b"FILTERBANK"))
        handle.attrs.create("VERSION", np.bytes_(b"1.0"))
        data = np.zeros((rows, ifs, channels), dtype=np.float32)
        for row in range(rows):
            data[row] = band * 1_000_000 + (start // stride + row)
        mask = np.zeros(data.shape, dtype=np.uint8)
        for row in masked:
            data[row] = 0.0
            mask[row] = 1
        dataset = handle.create_dataset("data", data=data, maxshape=(None, ifs, channels), chunks=(1, ifs, channels))
        handle.create_dataset("mask", data=mask, maxshape=(None, ifs, channels), chunks=(1, ifs, channels))
        for name, value in (("fch1", fch1), ("foff", foff), ("tsamp", stride * SAMPLE_PERIOD),
                            ("tstart", 40587.0 + (synctime + start * SAMPLE_PERIOD) / 86400.0)):
            dataset.attrs.create(name, value, dtype=np.float64)
        dataset.attrs.create("DIMENSION_LABELS", ["time", "feed_id", "frequency"], dtype=h5py.string_dtype())
        for name, value in (("nchans", channels), ("nifs", ifs), ("nbits", 32), ("telescope_id", 6), ("ibeam", ibeam)):
            dataset.attrs.create(name, value, dtype=np.int32)
        if grid:
            dataset.attrs.create("rf_start_sample", start, dtype=np.uint64)
            dataset.attrs.create("synctime", synctime, dtype=np.uint64)
            dataset.attrs.create("rf_samples_per_row", stride, dtype=np.uint64)


class StitcherTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="stelline-test-")
        self.root = Path(self.scratch.name)

    def tearDown(self):
        self.scratch.cleanup()

    def path(self, name):
        return str(self.root / name)

    def run_cli(self, *args):
        stderr = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
            status = stitcher.main(list(args))
        return status, stderr.getvalue()

    def test_stitches_staggered_bands_over_their_common_rows(self):
        write_band(self.path("b.fbh5"), band=2, fch1=1001.0, foff=0.25, start=10 * STRIDE, rows=20, masked=(5,))
        write_band(self.path("a.fbh5"), band=1, fch1=1000.0, foff=0.25, start=12 * STRIDE, rows=10)
        write_band(self.path("c.fbh5"), band=3, fch1=1002.0, foff=0.25, start=8 * STRIDE, rows=16)

        status, error = self.run_cli(self.path("out.fbh5"), self.path("b.fbh5"), self.path("a.fbh5"), self.path("c.fbh5"))
        self.assertEqual(status, 0, error)

        with h5py.File(self.path("out.fbh5"), "r") as handle:
            data = handle["data"]
            mask = handle["mask"][()]
            self.assertEqual(data.shape, (10, 2, 12))
            self.assertEqual(data.attrs["rf_start_sample"], 12 * STRIDE)
            self.assertEqual(data.attrs["rf_start_sample"].dtype, np.uint64)
            self.assertEqual(data.attrs["synctime"], SYNCTIME)
            self.assertEqual(data.attrs["rf_samples_per_row"], STRIDE)
            self.assertEqual(data.attrs["nchans"], 12)
            self.assertEqual(data.attrs["nchans"].dtype, np.int32)
            self.assertEqual(data.attrs["fch1"], 1000.0)
            self.assertEqual(data.attrs["foff"], 0.25)
            self.assertEqual(data.attrs["tstart"], 40587.0 + (SYNCTIME + 12 * STRIDE * SAMPLE_PERIOD) / 86400.0)
            self.assertEqual(handle.attrs["CLASS"], b"FILTERBANK")
            self.assertEqual(list(data.attrs["DIMENSION_LABELS"]), ["time", "feed_id", "frequency"])
            values = data[()]
            for row in range(10):
                grid_row = 12 + row
                np.testing.assert_array_equal(values[row, :, 0:4], 1_000_000 + grid_row)
                expected_b = 0 if grid_row == 15 else 2_000_000 + grid_row
                np.testing.assert_array_equal(values[row, :, 4:8], expected_b)
                np.testing.assert_array_equal(values[row, :, 8:12], 3_000_000 + grid_row)
                np.testing.assert_array_equal(mask[row, :, 4:8], 1 if grid_row == 15 else 0)
            self.assertEqual(int(mask[:, :, 0:4].sum() + mask[:, :, 8:12].sum()), 0)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["a.fbh5", "b.fbh5", "c.fbh5", "out.fbh5"])

    def test_keeps_descending_frequency_order(self):
        write_band(self.path("high.fbh5"), band=1, fch1=1002.0, foff=-0.25, start=0, rows=4, ibeam=7)
        write_band(self.path("low.fbh5"), band=2, fch1=1001.0, foff=-0.25, start=0, rows=4, ibeam=8)
        status, error = self.run_cli(self.path("out.fbh5"), self.path("low.fbh5"), self.path("high.fbh5"))
        self.assertEqual(status, 0, error)
        with h5py.File(self.path("out.fbh5"), "r") as handle:
            self.assertEqual(handle["data"].attrs["fch1"], 1002.0)
            self.assertEqual(handle["data"].attrs["ibeam"], 7)
            self.assertEqual(handle["data"][0, 0, 0], 1_000_000)
            self.assertEqual(handle["data"][0, 0, 4], 2_000_000)

    def test_rejects_inputs_that_cannot_be_stitched(self):
        write_band(self.path("a.fbh5"), band=1, fch1=1000.0, foff=0.25, start=0, rows=4)
        cases = {
            "gap": dict(fch1=1001.25),
            "overlap": dict(fch1=1000.75),
            "synctime": dict(synctime=SYNCTIME + 1),
            "stride": dict(stride=STRIDE * 2),
            "off-grid": dict(start=3),
            "no-overlap": dict(start=4 * STRIDE),
            "width": dict(foff=0.5),
            "ifs": dict(ifs=1),
            "no-grid": dict(grid=False),
        }
        for name, overrides in cases.items():
            with self.subTest(name):
                options = dict(band=2, fch1=1001.0, foff=0.25, start=0, rows=4)
                options.update(overrides)
                write_band(self.path("b.fbh5"), **options)
                status, error = self.run_cli(self.path("out.fbh5"), self.path("a.fbh5"), self.path("b.fbh5"))
                self.assertEqual(status, 1)
                self.assertTrue(error.startswith("stitcher: "), error)
                self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["a.fbh5", "b.fbh5"])

    def test_rejects_channel_width_drift_across_the_joined_axis(self):
        foff = 0.25
        drifted = foff * (1 + 9e-10)
        write_band(self.path("a.fbh5"), band=1, fch1=1000.0, foff=foff, start=0, rows=2, ifs=1)
        write_band(self.path("b.fbh5"), band=2, fch1=1001.0, foff=drifted, start=0, rows=2, ifs=1, channels=4096)
        write_band(self.path("c.fbh5"), band=3, fch1=1001.0 + 4096 * drifted, foff=foff, start=0, rows=2, ifs=1)
        status, error = self.run_cli(self.path("out.fbh5"), self.path("a.fbh5"), self.path("b.fbh5"), self.path("c.fbh5"))
        self.assertEqual(status, 1)
        self.assertIn("off the joined axis", error)
        self.assertFalse(Path(self.path("out.fbh5")).exists())

    def test_rejects_more_channels_than_nchans_can_hold(self):
        write_band(self.path("a.fbh5"), band=1, fch1=1000.0, foff=0.25, start=0, rows=2)
        write_band(self.path("b.fbh5"), band=2, fch1=1001.0, foff=0.25, start=0, rows=2)
        limit = stitcher.MAX_CHANNELS
        stitcher.MAX_CHANNELS = 7
        try:
            status, error = self.run_cli(self.path("out.fbh5"), self.path("a.fbh5"), self.path("b.fbh5"))
        finally:
            stitcher.MAX_CHANNELS = limit
        self.assertEqual(status, 1)
        self.assertIn("nchans limit", error)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["a.fbh5", "b.fbh5"])

    def test_chunks_match_the_copy_blocks(self):
        for rows, ifs, channels in ((100_000, 4, 3 * 524_288), (100_000, 2, 64), (3, 1, 4), (10, 4, 300_000)):
            with self.subTest(rows=rows, ifs=ifs, channels=channels):
                chunks, step = stitcher.chunk_layout(rows, ifs, channels, 4, ifs * channels * 5)
                depth, chunk_ifs, tile = chunks
                self.assertEqual(chunk_ifs, ifs)
                self.assertTrue(1 <= depth <= rows)
                self.assertTrue(1 <= tile <= channels)
                self.assertEqual(step % depth, 0)
                self.assertLessEqual(depth * ifs * tile * 4, max(stitcher.CHUNK_BYTES, ifs * 4))

    def test_respects_an_existing_output(self):
        write_band(self.path("a.fbh5"), band=1, fch1=1000.0, foff=0.25, start=0, rows=4)
        write_band(self.path("b.fbh5"), band=2, fch1=1001.0, foff=0.25, start=0, rows=4)
        Path(self.path("out.fbh5")).write_text("previous")

        status, _ = self.run_cli(self.path("out.fbh5"), self.path("a.fbh5"), self.path("b.fbh5"))
        self.assertEqual(status, 1)
        self.assertEqual(Path(self.path("out.fbh5")).read_text(), "previous")

        status, error = self.run_cli(self.path("out.fbh5"), self.path("a.fbh5"), self.path("b.fbh5"), "--overwrite")
        self.assertEqual(status, 0, error)
        with h5py.File(self.path("out.fbh5"), "r") as handle:
            self.assertEqual(handle["data"].shape, (4, 2, 8))
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["a.fbh5", "b.fbh5", "out.fbh5"])

    def test_rejects_an_input_as_the_output(self):
        write_band(self.path("a.fbh5"), band=1, fch1=1000.0, foff=0.25, start=0, rows=4)
        write_band(self.path("b.fbh5"), band=2, fch1=1001.0, foff=0.25, start=0, rows=4)
        status, _ = self.run_cli(self.path("a.fbh5"), self.path("a.fbh5"), self.path("b.fbh5"), "--overwrite")
        self.assertEqual(status, 1)
        os.link(self.path("a.fbh5"), self.path("alias.fbh5"))
        status, error = self.run_cli(self.path("alias.fbh5"), self.path("a.fbh5"), self.path("b.fbh5"), "--overwrite")
        self.assertEqual(status, 1)
        self.assertIn("cannot be one of the inputs", error)
        with h5py.File(self.path("a.fbh5"), "r") as handle:
            self.assertEqual(handle["data"].shape, (4, 2, 4))


if __name__ == "__main__":
    unittest.main()
