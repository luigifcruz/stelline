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
    cuda_lst += files([
        'ata_kernel.cu',
    ])
    src_lst += files([
        'ata_receiver.cc',
        'dummy_receiver.cc',
        'sorter.cc',
    ])
endif

operator_lst += {'Transport': all_deps_found}
