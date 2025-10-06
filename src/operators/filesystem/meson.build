deps = [
    'STELLINE_LOADER_HOLOSCAN_CORE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_OPERATOR_FILESYSTEM', true)
    src_lst += files([
        'simple_writer.cc',
        'simple_writer_rdma.cc',
        'dummy_writer.cc',
    ])
    subdir('utils')
endif

if cfg_lst.get('STELLINE_LOADER_FBH5', false)
    src_lst += files([
        'fbh5_writer_rdma.cc',
    ])
endif

if cfg_lst.get('STELLINE_LOADER_UVH5', false)
    src_lst += files([
        'uvh5_writer_rdma.cc',
    ])
endif

operator_lst += {'Filesystem': all_deps_found}
