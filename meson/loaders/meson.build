ldr_lst = {}

subdir('threads')
subdir('cuda')
subdir('jetstream')
subdir('radiointerferometry')

if plugin_device_cuda_enabled
    subdir('hdf5')
    subdir('daqiri')
endif

summary(ldr_lst, section: 'Loaders', bool_yn: true)
