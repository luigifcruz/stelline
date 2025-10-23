project(
    'stelline',
    ['cpp', 'cuda'],
    version: '0.1.0',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',
        'cuda_std=c++20',
    ]
)

is_static = get_option('default_library') == 'static'
cuda_args = ['-arch', 'compute_70', '--expt-relaxed-constexpr', '-diag-suppress', '815']
cdn_url = 'https://stelline.cdn.luigi.ltd/' + meson.project_version() + '/'

cfg_lst = configuration_data({
    'version': meson.project_version(),
    'build_type': get_option('buildtype'),
})
src_lst = []
dep_lst = [
    dependency('cuda', version: '>=11', modules: ['cuda', 'culibos', 'cufile']),
]
inc_lst = [
    include_directories('include'),
]

subdir('meson')
subdir('src')
subdir('resources')
subdir('include')

lib_stelline = library(
    'stelline',
    src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
    gnu_symbol_visibility: 'hidden',
    install: not is_static,
    cuda_args: cuda_args,
)

lib_stelline_dep = declare_dependency(
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_with: lib_stelline,
)

if not is_static
    pkg = import('pkgconfig')
    pkg.generate(
        lib_stelline,
        libraries: dep_lst,
    )
endif

subdir('python')
