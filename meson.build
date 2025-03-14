project(
    'stelline',
    ['cpp', 'cuda'],
    version: '0.0.1',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',
        'cuda_std=c++20',
    ]
)

is_static = get_option('default_library') == 'static'
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
