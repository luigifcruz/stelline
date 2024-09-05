deps = [
    'STELLINE_LOADER_CURL',
    'STELLINE_LOADER_TRTEXEC',
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and cfg_lst.get(x_dep, false)
endforeach

if all_deps_found
    subdir('model')
    cfg_lst.set('STELLINE_RESOURCE_FRBNN_MODEL', true)
endif

res_lst += {'FRBNN Model': all_deps_found}