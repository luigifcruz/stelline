deps = [
    dependency(
        'AdvancedNetwork',
        method : 'cmake',
        modules: [
            'holoscan::ops::advanced_network_common',
            'holoscan::ops::advanced_network_rx',
        ],
        required: false,
    ),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_LOADER_HOLOSCAN_ANO', true)
    dep_lst += deps
endif

ldr_lst += {'Holoscan Networking': all_deps_found}
