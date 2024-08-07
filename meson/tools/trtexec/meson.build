deps = [
    find_program('trtexec', required: false),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_LOADER_TRTEXEC', true)
    trtexec = deps[0]
endif

ldr_lst += {'TensorRT Compiler': all_deps_found}
