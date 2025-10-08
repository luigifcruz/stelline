deps = [
    dependency('hdf5_vfd_gds', required: false),
    dependency('h5dsc99', required: false),
]

format_deps = [
    dependency('filterbankc99', required: false),
    dependency('uvh5c99', required: false),
]

basic_deps_found = true
foreach x_dep : deps
    basic_deps_found = basic_deps_found and x_dep.found()
endforeach

fbh5_dep = dependency('filterbankc99', required: false)
fbh5_dep_found = fbh5_dep.found()
uvh5_dep = dependency('uvh5c99', version: '>= 0.9.0', required: false)
uvh5_dep_found = uvh5_dep.found()

if basic_deps_found and (fbh5_dep_found or uvh5_dep_found)
    cfg_lst.set('STELLINE_LOADER_HDF5', true)
    dep_lst += deps

    if fbh5_dep_found
        cfg_lst.set('STELLINE_LOADER_FBH5', true)
        dep_lst += [fbh5_dep]
    endif
    if uvh5_dep_found
        cfg_lst.set('STELLINE_LOADER_UVH5', true)
        dep_lst += [uvh5_dep]
    endif
endif

ldr_lst += {'FBH5': basic_deps_found and fbh5_dep_found}
ldr_lst += {'UVH5': basic_deps_found and uvh5_dep_found}
