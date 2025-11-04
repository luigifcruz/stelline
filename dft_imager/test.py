import holoscan as hs
from holoscan.core import Operator, OperatorSpec
import xarray as xr
import cupy as cp
import numpy as np
from numpy.typing import NDArray
import dask.array as da
from pathlib import Path
from holoscan.conditions import CountCondition

LIGHTSPEED = 3e8


class XarrayZarrReaderOperator(Operator):
    """
    Reads zarr slices via xarray, transfers to GPU, emits pointer.
    """
    
    def __init__(self, fragment, *args, zarr_path: str,
                 data_column: str = 'VISIBILITY', 
                 stokes: str = 'IQUV', **kwargs):
        self.zarr_path = zarr_path
        self.data_column = data_column
        self.stokes = stokes
        self.current_index = 0
        self.ntime = None
        super().__init__(fragment, *args, **kwargs)
    
    def setup(self, spec: OperatorSpec):
        spec.output("output")
    
    def start(self):
        dz = xr.open_datatree(
            self.zarr_path, 
            engine='zarr',
            chunks='auto'
        )
        # only the first data group for now (will need to iterate)
        self.dataset = dz[list(dz.children)[0]].ds
        self.ntime = self.dataset['time'].size
        self.current_index = 0

        # get additional metadata
        self.out_times = np.mean(self.dataset.time.values, keepdims=True)
        self.times = hs.as_tensor(cp.asarray(self.dataset.time.values))
        self.out_freqs = np.mean(self.dataset.frequency.values, keepdims=True)
        self.frequencies = hs.as_tensor(cp.asarray(self.dataset.frequency.values))
        
        # will need this eventually
        mask = ['antenna_xds' in g for g in list(dz.groups)]
        ant_idx = next(i for i, x in enumerate(mask) if x)
        ds_ant = dz[dz.groups[ant_idx]].ds
        self.nant = ds_ant['antenna_name'].size
    
    def compute(self, op_input, op_output, context):
        if self.current_index >= self.ntime:
            return
        
        # Read slice from zarr to CPU
        ds_slice = self.dataset.isel(
            {'time': slice(self.current_index, self.current_index+1)},  # could use larger chunks
        )
        
        # Transfer variales required for imaging to GPU
        image_vars = ['UVW', 'VISIBILITY', 'WEIGHT', 'FLAG', 'time']
        output = {'FREQ': self.frequencies}
        for var in image_vars:
            output[var] = hs.as_tensor(cp.asarray(ds_slice[var].values))
        
        # Emit dict of tensors
        op_output.emit(output, "output")
        
        self.current_index += 1
    
    def stop(self):
        pass


class CoreAlgorithmOperator(Operator):
    """
    Receives GPU pointer, computes on GPU, emits GPU pointer.
    """
    def __init__(self, fragment, cellx, celly, nx, ny, *args,
                 pol: str = 'linear', **kwargs):
        self.pol = pol
        self.cellx = cellx
        self.celly = celly
        self.nx = nx
        self.ny = ny
        super().__init__(fragment, *args, **kwargs)

    def start(self):
        # these operators map stokes_to_corr/corr_to_stokes
        if self.pol == 'linear':
            self.s2c = cp.array([[1.0, 1.0, 0, 0],
                                 [0, 0, 1.0, 1.0j],
                                 [0, 0, 1.0, -1.0j],
                                 [1.0, -1.0, 0, 0]])
            self.c2s = cp.array([[0.5, 0.0, 0, 0.5],
                                 [0.5, 0, 0.0, -0.5],
                                 [0, 0.5, 0.5, 0],
                                 [0, -0.5j, 0.5j, 0]])
        else:
            raise NotImplementedError

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")
    
    def compute(self, op_input, op_output, context):
        # Receive tensor (GPU pointer)
        input_tensor = op_input.receive("input")
        
        # Get cupy view (no copy, just pointer)
        image_vars = {}
        for key, val in input_tensor.items():
            image_vars[key] = cp.asarray(val)
        
        # Your core algorithm here - all on GPU
        result = self._process_on_gpu(image_vars)
        
        # not sure of hs.as_tensor is required (Claude recommendation)
        # in general there will be more than one image in the output (e.g. the PSF, BEAM etc.)
        op_output.emit(result, "output")

    def _corr_to_stokes(self, vis, wgt):
        # this can actually be done analytically including Jones matrix application
        stokes_wgt = self.s2c.conj().T @ (wgt[:, None] * self.s2c)
        stokes_vis = cp.linalg.solve(stokes_wgt, self.s2c.conj().T @ (wgt * vis))
        return stokes_vis, cp.diag(stokes_wgt)  # keep only diagonal wgt

    
    def _process_on_gpu(self, image_vars: dict[cp.ndarray]) -> cp.ndarray:
        """
        Eventually need to duplicate functionality here

        https://github.com/ratt-ru/pfb-imaging/blob/main/pfb/utils/stokes2im.py
        """
        # Example processing
        uvw = image_vars['UVW']
        vis = image_vars['VISIBILITY']
        wgt = image_vars['WEIGHT']
        flag = image_vars['FLAG']
        freqs = image_vars['FREQ']
        times = image_vars['time']
        # currently assume reduction over time and freq
        time_out = cp.mean(times, keepdims=True)
        freq_out = cp.mean(freqs, keepdims=True)
        x, y = cp.meshgrid(*[-ss / 2 + cp.arange(ss) for ss in [self.nx, self.ny]], indexing="ij")
        x *= self.cellx
        y *= self.celly
        eps = x**2 + y**2
        apply_w = True  # probably not necessary
        if apply_w:
            nm1 = -eps / (cp.sqrt(1.0 - eps) + 1.0)
            n = (nm1 + 1)[None, :, :]
        else:
            nm1 = 0.0
            n = 1.0
        ntime, nbl, nchan, ncorr = vis.shape
        res = cp.zeros((ncorr, self.nx, self.ny), dtype=wgt.dtype)
        # this should be vectorized for better paralellism
        for t in range(ntime):  # probably only one of these
            for bl in range(nbl):
                for chan in range(nchan):
                    # skip if any corrrelation is flagged
                    if flag[t, bl, chan].any():
                        continue
                    # convert to corrected Stokes vis and weights (harder to vectorize with gains)
                    stokes_vis, stokes_wgt = self._corr_to_stokes(vis[t, bl, chan], wgt[t, bl, chan])
                    u, v, w = uvw[t, bl]
                    phase = freqs[chan] / LIGHTSPEED * (x * u + y * v - w * nm1)
                    cphase = cp.exp(2j * cp.pi * phase)
                    res += (stokes_vis[:, None, None] * stokes_wgt[:, None, None] * cphase[None, :, :]).real
        res /= n
        # axes placeholders
        res = res[:, None, None, :, :]
        out_dict = {
            'cube': hs.as_tensor(res),
            'time_out': hs.as_tensor(time_out),
            'freq_out':hs.as_tensor(freq_out)
        }
        return out_dict


class ResultWriterOperator(Operator):
    """
    The output format for tron actually looks like this

    https://github.com/ratt-ru/pfb-imaging/blob/bb5af97cd4b887cf26560c6c02cad7a34612aad4/pfb/workers/hci.py#L503
    """
    
    def __init__(self, fragment, nstokes, nfreq_out, ntime, ny, nx, *args,
                 output_dataset: str = None, out_stokes: NDArray = None,
                 out_freqs: NDArray = None, out_times: NDArray = None,
                 out_ras: NDArray = None, out_decs: NDArray = None, **kwargs):
        self.output_dataset = output_dataset
        self.nstokes = nstokes
        self.nfreq_out = nfreq_out
        self.ntime = ntime
        self.ny = ny
        self.nx = nx
        # x and y is swapped to avoid transpose when writing to fits
        self.cube_dims = (nstokes, nfreq_out, ntime, ny, nx)
        self.cube_chunks = (nstokes, 1, 1, ny, nx)
        self.mean_dims = (nstokes, nfreq_out, ny, nx)
        self.mean_chunks = (nstokes, 1, ny, nx)
        self.out_stokes = out_stokes if out_stokes is not None else np.array(['I', 'Q', 'U', 'V'])[0:nstokes]
        self.out_ras = out_ras if out_ras is not None else np.arange(nx)
        self.out_decs = out_decs if out_decs is not None else np.arange(ny)
        self.coords = {
            "TIME": (("TIME",), out_times if out_times is not None else np.arange(ntime)),
            "STOKES": (("STOKES",), self.out_stokes),
            "FREQ": (("FREQ",), out_freqs if out_freqs is not None else np.arange(nfreq_out)),
            "X": (("X",), self.out_ras),
            "Y": (("Y",), self.out_decs)
        }
        super().__init__(fragment, *args, **kwargs)
    
    def start(self):
        # here we use dask to create a xarray scaffold to write data to
        # note the use of da.empty to avoid actually writing the data

        data_vars = {
            'cube': (
                ("STOKES", "FREQ", "TIME", "Y", "X"),
                da.empty(self.cube_dims, chunks=self.cube_chunks, dtype=np.float32)
            ),
            'cube_mean': (
                ("STOKES", "FREQ", "TIME", "Y", "X"),
                da.empty(self.cube_dims, chunks=self.cube_chunks, dtype=np.float32)
            )
        }
        attrs = {
            'just': 1.0,
            'testing': 2.0
        }
        ds = xr.Dataset(
            data_vars=data_vars,
            coords=self.coords,
            attrs=attrs,
        )
        ds.to_zarr(self.output_dataset, mode='w', compute=True)

    def setup(self, spec: OperatorSpec):
        spec.input("input")
    
    def compute(self, op_input, op_output, context):
        result_dict = op_input.receive("input")
        cube = cp.asnumpy(cp.asarray(result_dict['cube']))
        freq_out = cp.asnumpy(cp.asarray(result_dict['freq_out']))
        time_out = cp.asnumpy(cp.asarray(result_dict['time_out']))
        
        # these allow us to use region=auto
        coords = {
            'FREQ': (('FREQ',), freq_out),
            'TIME': (('TIME',), time_out),
            'STOKES': (('STOKES',), self.out_stokes),
            'X': (('X',), self.out_ras),
            'Y': (('Y',), self.out_decs),
        }

        dso = xr.Dataset(
            data_vars = {'cube' : (('STOKES', 'FREQ', 'TIME', 'Y', 'X'), cube)},
            coords = coords
        )
        
        dso.to_zarr(
            self.output_dataset,
            region="auto",
        )
    
    def stop(self):
        # need to compute the mean here
        pass


class StreamingPipeline(hs.core.Application):
    def __init__(self, MSv4_zarr_path: Path, output_dataset: Path, *args,
                 data_column: str = 'VISIBILITY', 
                 stokes: str = 'IQUV', **kwargs):
        self.zarr_path = MSv4_zarr_path
        self.output_dataset = output_dataset
        self.data_column = data_column
        self.stokes = np.array(list(stokes))
        super().__init__(*args, **kwargs)

        # get view of data
        dz = xr.open_datatree(
            self.zarr_path, 
            engine='zarr',
            chunks='auto'
        )
        # only the first data group for now (will need to iterate)
        self.dataset = dz[list(dz.children)[0]].ds
        self.nstokes = self.stokes.size
        self.ntime = self.dataset.time.size
        self.out_times = self.dataset.time.values
        self.ntime_out = self.out_times.size
        self.out_freqs = np.mean(self.dataset.frequency.values, keepdims=True)
        self.nfreq_out = self.out_freqs.size

        # get image size
        uvw = self.dataset.UVW.values
        freq = self.dataset.frequency.values
        umax = np.abs(uvw[:, :, 0]).max()
        vmax = np.abs(uvw[:, :, 1]).max()
        uv_max = np.maximum(umax, vmax)
        cell = 1.0 / (2 * uv_max * freq.max() / LIGHTSPEED)  # Nyquist in radians
        field_of_view_deg = 1.0
        nx = int(np.ceil(np.deg2rad(field_of_view_deg) / cell))
        if nx % 2:
            nx += 1
        self.nx = nx 
        self.ny = nx
        self.cellx = cell
        self.celly = cell



    def compose(self):
        reader = XarrayZarrReaderOperator(
            self,
            CountCondition(self, self.ntime),
            name="reader",
            zarr_path=self.zarr_path,
            data_column=self.data_column,
            stokes=self.stokes
        )
        
        algorithm = CoreAlgorithmOperator(
            self,
            self.cellx,
            self.celly,
            self.nx,
            self.ny,
            name="core_algorithm",
        )
        
        writer = ResultWriterOperator(
            self,
            self.nstokes,
            self.nfreq_out,
            self.ntime_out,
            self.ny,
            self.nx,
            name="writer",
            output_dataset=self.output_dataset,
            out_stokes=self.stokes,
            out_freqs=self.out_freqs,
            out_times=self.out_times
        )
        
        # Connect the pipeline
        self.add_flow(reader, algorithm, {("output", "input")})
        self.add_flow(algorithm, writer, {("output", "input")})


if __name__ == "__main__":
    app = StreamingPipeline('/data/test_ascii_1h60.0s.zarr',  # input MSv4.zarr
                            '/data/test_stream.zarr')  # output_dataset.zarr
    app.config('config.yaml')
    app.run()