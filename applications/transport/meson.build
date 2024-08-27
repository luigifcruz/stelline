deps = [
    'STELLINE_BIT_TRANSPORT',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    subdir('test')
endif

apps_lst += {'Transport': all_deps_found}
