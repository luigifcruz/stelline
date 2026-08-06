project(
    'stelline',
    ['cpp', 'cuda'],
    version: '0.5.0',
    meson_version: '>=1.11.0',
    default_options: [
        'cpp_std=c++20',
        'warning_level=3',
        'buildtype=release',
        'b_ndebug=if-release',
        'cuda_std=c++20',
    ]
)

#
# Global Variables
#

plugin_is_linux = target_machine.system().startswith('linux')

if not plugin_is_linux
    error('Stelline only supports Linux targets for now.')
endif

minimum_jetstream_version = '1.8.2'

#
# Load Python
#

pymod = import('python')
py = pymod.find_installation(required: true)

#
# Handle Options
#

cfg_lst = configuration_data()
src_lst = []
cuda_lst = []
dep_lst = []
plugin_test_lst = []
inc_lst = [
    include_directories('include'),
]

plugin_requested_devices = get_option('devices')
if plugin_requested_devices.length() == 0
    error('The devices option must contain auto or cuda.')
endif

plugin_devices_auto = 'auto' in plugin_requested_devices
if plugin_devices_auto and plugin_requested_devices.length() != 1
    error('The auto device cannot be combined with explicit devices.')
endif

plugin_device_cuda_requested = plugin_devices_auto or 'cuda' in plugin_requested_devices
plugin_device_cuda_enabled = false

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

if not plugin_device_cuda_enabled
    error('No supported device implementations are available.')
endif

subdir('include')
subdir('src')

#
# Library
#

plugin_device_library = static_library(
    'stelline_device',
    sources: cuda_lst,
    include_directories: inc_lst,
    cuda_args: [
        '-Xcompiler=-fPIC',
        '--expt-relaxed-constexpr',
    ],
    build_by_default: cuda_lst.length() > 0,
    install: false,
)

plugin_library = static_library(
    'stelline_code',
    src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_with: plugin_device_library,
    gnu_symbol_visibility: 'hidden',
    pic: true,
    install: false,
)

plugin_code_dep = declare_dependency(
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_whole: plugin_library,
)

if get_option('tests')
    subdir('tests')
endif

#
# Packaging
#

plugin_examples = files(
    'examples/ata-beamformer.yml',
    'examples/ata-correlator.yml',
    'examples/ata-spectrogram.yml',
)

plugin_module = shared_module(
    'stelline',
    [],
    dependencies: plugin_code_dep,
    gnu_symbol_visibility: 'hidden',
    link_language: 'cpp',
    link_args: ['-Wl,-Bsymbolic', '-Wl,--no-as-needed'],
    name_prefix: '',
    build_by_default: false,
    install: false,
)

plugin_package_device = plugin_device_cuda_enabled ? 'cuda' : ''

custom_target(
    'stelline_cep',
    input: [plugin_module] + plugin_examples,
    output: 'stelline.cep',
    command: [
        py,
        files('tools/bundler.py'),
        '--output', '@OUTPUT@',
        '--name', meson.project_name(),
        '--version', meson.project_version(),
        '--minimum-jetstream-version', minimum_jetstream_version,
        '--target', 'path=@INPUT0@,system=' + host_machine.system() + ',device=' + plugin_package_device + ',arch=' + host_machine.cpu_family(),
        '--example', '@INPUT1@',
        '--example', '@INPUT2@',
        '--example', '@INPUT3@',
    ],
    build_by_default: true,
    install: false,
)

summary({'CEP bundle': true}, section: 'Packaging', bool_yn: true)

#
# Print Summary
#

summary({
    'cpp_std': get_option('cpp_std'),
    'prefix': get_option('prefix'),
    'buildtype': get_option('buildtype'),
}, section: 'General', bool_yn: true)

summary({
    'Selection': ','.join(plugin_requested_devices),
    'CUDA': plugin_device_cuda_enabled,
}, section: 'Devices', bool_yn: true)
