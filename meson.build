project(
    'stelline',
    ['cpp', 'cuda'],
    version: '0.4.0',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',
        'cuda_std=c++20',
    ]
)

cfg_lst = configuration_data()
cfg_lst.set('version', meson.project_version())

src_lst = []
cuda_lst = []
module_test_lst = []
dep_lst = []
cuda_dep_lst = []
inc_lst = [
    include_directories('include'),
]

subdir('meson')
subdir('src')

lib_stelline_device = static_library(
    'stelline_device',
    sources: cuda_lst,
    include_directories: inc_lst,
    cuda_args: [
        '-Xcompiler=-fPIC',
        '--expt-relaxed-constexpr',
    ],
    build_by_default: cuda_lst.length() > 0,
)

lib_stelline = library(
    'stelline',
    src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_with: lib_stelline_device,
    gnu_symbol_visibility: 'hidden',
    link_args: ['-Wl,-Bsymbolic', '-Wl,--no-as-needed'],
    install: true,
)

lib_stelline_dep = declare_dependency(
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_with: lib_stelline,
)

pkg = import('pkgconfig')
pkg.generate(
    lib_stelline,
    libraries: dep_lst,
)
