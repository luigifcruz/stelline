deps = [
    'STELLINE_OPERATOR_BLADE',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_BIT_BLADE', true)
endif

bits_lst += {'BLADE': all_deps_found}