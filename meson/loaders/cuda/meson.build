cuda_deps = []
cuda_found = false

if plugin_device_cuda_requested
    cuda_deps += dependency(
        'cuda',
        version: '>=11',
        modules: ['cuda', 'culibos', 'cufile'],
        required: not plugin_devices_auto,
    )

    cuda_found = true
    foreach cuda_dep : cuda_deps
        cuda_found = cuda_found and cuda_dep.found()
    endforeach
endif

if cuda_found
    cfg_lst.set('STELLINE_LOADER_CUDA_AVAILABLE', true)
    cfg_lst.set('STELLINE_DEVICE_CUDA_AVAILABLE', true)
    dep_lst += cuda_deps
    plugin_device_cuda_enabled = true
endif

ldr_lst += {'CUDA': cuda_found}
