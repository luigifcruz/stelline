# Add source list to the stack.

src_lst = files([
    'bridge.cc',
    'render.cc',
    'source_module.cc',
    'tensor_to_holoscan.cc',
])

# Check if all dependencies are present.

dep_lst = [
    dependency('jetstream', required: false),
    dependency('cuda', modules: ['cudart', 'culibos'], required: false),
    dependency(
        'holoscan', 
        modules: [
            'holoscan::core', 
            'holoscan::logger', 
        ],
        required: false,
    ),
]

foreach dep : dep_lst
    if not dep.found()
        warning('Holoscan stack cannot be enabled because dependency `' + dep.name() + '` was not found.')
        subdir_done() 
    endif
endforeach

cfg_lst.set('CYBERBRIDGE_HOLOSCAN_STACK_AVAILABLE', true)

# Create stack library.

cyberbridge_holoscan_lib = library(
    'cyberbridge-holoscan',
    src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
    install: true
)

cyberbridge_holoscan_dep = declare_dependency(
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_with: cyberbridge_holoscan_lib
)

pkg = import('pkgconfig')
pkg.generate(
    cyberbridge_holoscan_lib,
    libraries: dep_lst,
)