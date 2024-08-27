deps = [
    'STELLINE_LOADER_HOLOSCAN_CORE',
    'STELLINE_LOADER_HOLOSCAN_ANO',
    'STELLINE_LOADER_MATX',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_OPERATOR_TRANSPORT', true)
    src_lst += files([
        'base.cc',
        'kernel.cu',
    ])
endif

operator_lst += {'Transport': all_deps_found}