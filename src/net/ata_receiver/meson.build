src_lst += files([
    'block_impl.cc',
    'module_impl.cc',
])

if plugin_device_cuda_enabled
    src_lst += files([
        'detail/daqiri_config.cc',
        'detail/scatter_pool.cc',
        'module_impl_native_cuda.cc',
    ])
    cuda_lst += files(['detail/kernel.cu'])
endif
