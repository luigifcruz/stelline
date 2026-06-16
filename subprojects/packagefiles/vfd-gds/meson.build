project(
    'hdf5_vfd_gds',
    'c',
    version: '1.0.2',
    default_options: [
        'c_std=c99',
        'buildtype=release',
        'b_ndebug=if-release',
    ]
)

deps = [
    dependency('cuda', version: '>=11', modules: ['cuda', 'cudart', 'cufile']),
    dependency('hdf5', language: 'c'),
    dependency('threads'),
]

inc = include_directories('src')

lib_hdf5_vfd_gds = library(
    'hdf5_vfd_gds',
    'src/H5FDgds.c',
    include_directories: inc,
    dependencies: deps,
    install: true,
)

lib_hdf5_vfd_gds_dep = declare_dependency(
    include_directories: inc,
    dependencies: deps,
    link_with: lib_hdf5_vfd_gds,
)

install_headers('src/H5FDgds.h')

pkg = import('pkgconfig')
pkg.generate(
    lib_hdf5_vfd_gds,
    libraries: deps,
)
