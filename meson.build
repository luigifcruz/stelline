project(
    'stelline',
    ['cpp', 'cuda'],
    version: '0.4.3',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',
        'cuda_std=c++20',
    ]
)

#
# Global Variables
#

stl_is_linux = target_machine.system().startswith('linux')

if not stl_is_linux
    error('Stelline only supports Linux targets for now.')
endif

minimum_jetstream_version = '1.7.0'

#
# Load Python
#

pymod = import('python')
py = pymod.find_installation(required: false)

#
# Handle Options
#

cfg_lst = configuration_data()
src_lst = []
cuda_lst = []
dep_lst = []
inc_lst = [
    include_directories('include'),
]

#
# Handle Configuration Data
#

cfg_lst.set('version', meson.project_version())
version_parts = meson.project_version().split('.')
cfg_lst.set('version_major', version_parts[0])
cfg_lst.set('version_minor', version_parts[1])
cfg_lst.set('version_patch', version_parts[2])

#
# Call Subdirectories
#

subdir('meson')
subdir('src')

#
# Library
#

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

#
# Install Library
#

subdir('include')

pkg = import('pkgconfig')
pkg.generate(
    lib_stelline,
    libraries: dep_lst,
)

#
# Packaging
#

stelline_examples = files(
    'examples/ata-beamformer.yml',
    'examples/ata-correlator.yml',
    'examples/ata-spectrogram.yml',
)

if py.found()
    custom_target(
        'stelline_cep',
        input: [lib_stelline] + stelline_examples,
        output: 'stelline.cep',
        command: [
            py,
            files('tools/bundler.py'),
            '--output', '@OUTPUT@',
            '--name', meson.project_name(),
            '--version', meson.project_version(),
            '--minimum-jetstream-version', minimum_jetstream_version,
            '--target', 'path=@INPUT0@,system=' + host_machine.system() + ',device=cuda,arch=' + host_machine.cpu_family(),
            '--example', '@INPUT1@',
            '--example', '@INPUT2@',
            '--example', '@INPUT3@',
        ],
        build_by_default: true,
        install: false,
    )
endif

summary({'CEP bundle': py.found()}, section: 'Packaging', bool_yn: true)

#
# Print Summary
#

summary({
    'cpp_std': get_option('cpp_std'),
    'prefix': get_option('prefix'),
    'buildtype': get_option('buildtype'),
}, section: 'General', bool_yn: true)
