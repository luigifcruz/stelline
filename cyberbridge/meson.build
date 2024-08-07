project(
    'CyberBridge',
    ['cpp'],
    version: '1.0.0-alpha3',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',
    ]
)

# Declare global variables.

cfg_lst = configuration_data()
inc_lst = [
    include_directories('include'),
]

# Declare global configuration variables.

cfg_lst.set('version', meson.project_version())

# Declare stacks.

subdir('stack_holoscan')

# Declare targets.

subdir('include')