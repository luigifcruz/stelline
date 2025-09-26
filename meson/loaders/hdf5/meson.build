deps = [
    dependency('hdf5_vfd_gds', required: false),
    dependency('h5dsc99', required: false),
    dependency('filterbankc99', required: false),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_LOADER_HDF5', true)
    dep_lst += deps
endif

ldr_lst += {'HDF5': all_deps_found}
