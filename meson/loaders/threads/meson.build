deps = [
    dependency('threads'),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    dep_lst += deps
endif

ldr_lst += {'threads': all_deps_found}
