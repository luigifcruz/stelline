deps = [
    dependency(
        'holoscan',
        modules: [
            'holoscan::core',
            'holoscan::logger',
            'holoscan::pybind11',
        ],
        required: false,
    ),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_LOADER_HOLOSCAN_CORE', true)
    dep_lst += deps
endif

ldr_lst += {'Holoscan Core': all_deps_found}
