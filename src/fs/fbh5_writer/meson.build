dep_lst += [
    dependency('cuda', version: '>=11', modules: ['cuda', 'culibos', 'cufile'], required: true),
    stelline_hdf5_vfd_gds_dep,
    stelline_h5dsc99_dep,
    stelline_filterbankc99_dep,
]

src_lst += files([
    'block_impl.cc',
    'module_impl.cc',
    'module_impl_native_cuda.cc',
])
