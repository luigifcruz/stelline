# HDF5 upstream only supports CMake, so it cannot follow the plain
# subproject path used by the other wraps.
cmake = import('cmake')
hdf5_options = cmake.subproject_options()
hdf5_options.add_cmake_defines({
    'BUILD_SHARED_LIBS': false,
    'BUILD_STATIC_LIBS': true,
    'CMAKE_POSITION_INDEPENDENT_CODE': true,
    'HDF5_BUILD_CPP_LIB': false,
    'HDF5_BUILD_EXAMPLES': false,
    'HDF5_BUILD_FORTRAN': false,
    'HDF5_BUILD_HL_LIB': false,
    'HDF5_BUILD_JAVA': false,
    'HDF5_BUILD_TOOLS': false,
})

hdf5_dep = cmake.subproject('hdf5', options: hdf5_options).dependency('hdf5-static')
meson.override_dependency('hdf5', hdf5_dep)

deps = [
    subproject(
        'vfd-gds',
        default_options: [
            'default_library=static',
            'b_staticpic=true',
        ],
    ).get_variable('lib_hdf5_vfd_gds_dep'),
    subproject(
        'h5dsc99',
        default_options: [
            'default_library=static',
            'b_staticpic=true',
        ],
    ).get_variable('lib_h5dsc99_dep'),
    subproject(
        'filterbankc99',
        default_options: [
            'default_library=static',
            'b_staticpic=true',
        ],
    ).get_variable('lib_filterbankc99_dep'),
    subproject(
        'uvh5c99',
        default_options: [
            'default_library=static',
            'b_staticpic=true',
            'radiointerferometryc99:default_library=static',
            'radiointerferometryc99:b_staticpic=true',
        ],
    ).get_variable('lib_uvh5c99_dep'),
]

all_deps_found = true
foreach x_dep : deps
    all_deps_found = all_deps_found and x_dep.found()
endforeach

if all_deps_found
    cfg_lst.set('STELLINE_LOADER_HDF5_AVAILABLE', true)
    cfg_lst.set('STELLINE_LOADER_FBH5_AVAILABLE', true)
    cfg_lst.set('STELLINE_LOADER_UVH5_AVAILABLE', true)
    dep_lst += deps
endif

ldr_lst += {'FBH5': all_deps_found}
ldr_lst += {'UVH5': all_deps_found}
