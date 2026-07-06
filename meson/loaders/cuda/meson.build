deps = [
    dependency('cuda', version: '>=11', modules: ['cuda', 'culibos', 'cufile'], required: true),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_LOADER_CUDA_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'CUDA': all_deps_found}
