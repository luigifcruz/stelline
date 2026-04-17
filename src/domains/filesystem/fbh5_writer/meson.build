dep_lst += [
    dependency('cuda', version: '>=11', modules: ['cuda', 'culibos', 'cufile'], required: true),
    dependency('hdf5_vfd_gds', required: true),
    dependency('h5dsc99', required: true),
    dependency('filterbankc99', required: true),
]

src_lst += files([
    'block_impl.cc',
    'module_impl.cc',
    'module_impl_native_cuda.cc',
])
