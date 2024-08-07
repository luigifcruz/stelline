deps = [
    dependency(
        'matx',
        modules: [
            'matx::matx',
        ],
        required: false,
    ),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_LOADER_MATX', true)
    dep_lst += deps
endif

ldr_lst += {'MatX': all_deps_found}
