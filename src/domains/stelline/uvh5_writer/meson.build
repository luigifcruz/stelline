dep_lst += [
    dependency('cuda', version: '>=11', modules: ['cuda', 'culibos', 'cufile'], required: true),
    dependency('hdf5_vfd_gds', required: true),
    dependency('uvh5c99', version: '>= 0.9.0', required: true),
]

src_lst += files([
    'block_impl.cc',
    'module_impl.cc',
    'module_impl_native_cuda.cc',
])
