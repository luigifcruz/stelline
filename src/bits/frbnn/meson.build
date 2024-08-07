deps = [
    'STELLINE_OPERATOR_FRBNN',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_BIT_FRBNN', true)
endif
