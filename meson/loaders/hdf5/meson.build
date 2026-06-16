deps = [
    subproject('vfd-gds').get_variable('lib_hdf5_vfd_gds_dep'),
    subproject('h5dsc99').get_variable('lib_h5dsc99_dep'),
    subproject('filterbankc99').get_variable('lib_filterbankc99_dep'),
    subproject('uvh5c99').get_variable('lib_uvh5c99_dep'),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_LOADER_HDF5', true)
    cfg_lst.set('STELLINE_LOADER_FBH5', true)
    cfg_lst.set('STELLINE_LOADER_UVH5', true)
    dep_lst += deps
endif

ldr_lst += {'FBH5': all_deps_found}
ldr_lst += {'UVH5': all_deps_found}
