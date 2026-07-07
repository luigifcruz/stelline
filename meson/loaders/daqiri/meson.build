deps = [
    subproject(
        'daqiri',
        default_options: [
            'default_library=static',
            'b_staticpic=true',
        ],
    ).get_variable('daqiri_dep'),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_LOADER_DAQIRI_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'DAQIRI': all_deps_found}
