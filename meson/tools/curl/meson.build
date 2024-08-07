deps = [
    find_program('curl', required: false)
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_LOADER_CURL', true)
    curl = deps[0]
endif

ldr_lst += {'cURL': all_deps_found}
