dep_lst += [
    dependency('cuda', version: '>=11', modules: ['cuda', 'culibos', 'cufile'], required: true),
    subproject('vfd-gds').get_variable('lib_hdf5_vfd_gds_dep'),
    subproject('h5dsc99').get_variable('lib_h5dsc99_dep'),
    subproject('filterbankc99').get_variable('lib_filterbankc99_dep'),
]

src_lst += files([
    'block_impl.cc',
    'module_impl.cc',
    'module_impl_native_cuda.cc',
])
