dep_lst += [
    dependency('cuda', version: '>=11', modules: ['cuda', 'culibos', 'cufile'], required: true),
    subproject('vfd-gds').get_variable('lib_hdf5_vfd_gds_dep'),
    subproject('uvh5c99').get_variable('lib_uvh5c99_dep'),
]

src_lst += files([
    'block_impl.cc',
    'module_impl.cc',
    'module_impl_native_cuda.cc',
])
