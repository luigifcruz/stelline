deps = [
    'STELLINE_LOADER_HOLOSCAN_CORE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_OPERATOR_IO', true)
    src_lst += files([
        'simple_sink.cc',
        'simple_sink_rdma.cc',
        'dummy_sink.cc',
    ])
endif

operator_lst += {'I/O': all_deps_found}
