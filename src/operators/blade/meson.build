deps = [
    'STELLINE_LOADER_HOLOSCAN_CORE',
    'STELLINE_LOADER_BLADE',
    'STELLINE_LOADER_MATX',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_OPERATOR_BLADE', true)
    src_lst += files([
        'correlator.cc',
        'frbnn.cc',
    ])
endif

operator_lst += {'BLADE': all_deps_found}