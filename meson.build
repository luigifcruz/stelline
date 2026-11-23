project(
    'stelline',
    ['cpp', 'cuda'],
    version: '0.7.0',
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
# Plugin Metadata
#

plugin_id = 'stelline'
plugin_title = 'Stelline'
plugin_summary = 'GPU-accelerated radio astronomy blocks for the Allen Telescope Array.'
plugin_author = 'Stelline'
plugin_license = 'MIT'
plugin_description = import('fs').read('description.md').strip()

#
# CyberEther
#

cyberether_dep = dependency(
    'jetstream',
    fallback: ['cyberether', 'libjetstream_dep'],
    required: true,
    default_options: [
        'tests=false',
        'examples=false',
        'remote=disabled',
    ],
)

cyberether_version = cyberether_dep.version()
cyberether_build_id = cyberether_dep.get_variable('build_id', default_value: '')
cyberether_backends = cyberether_dep.get_variable('backends', default_value: '').split(',')

if cyberether_build_id == ''
    error('This CyberEther SDK does not export a build ID. Build against a newer CyberEther.')
endif

if not cyberether_build_id.startswith('linux-')
    error('Stelline only supports Linux builds of CyberEther for now.')
endif

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

plugin_device_cuda_requested = 'cuda' in cyberether_backends
plugin_device_cuda_enabled = false

if not plugin_device_cuda_requested
    error('Stelline needs a CUDA build of CyberEther, such as linux-x86_64-cuda.')
endif

plugin_device_cpu_enabled = true
cfg_lst.set('STELLINE_DEVICE_CPU_AVAILABLE', true)

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
    plugin_id,
    [],
    dependencies: plugin_code_dep,
    gnu_symbol_visibility: 'hidden',
    link_language: 'cpp',
    link_args: ['-Wl,-Bsymbolic', '-Wl,--no-as-needed'],
    name_prefix: '',
    build_by_default: false,
    install: false,
)

plugin_bundle_args = [
    py,
    files('tools/bundler.py'),
    '--output', '@OUTPUT@',
    '--id', plugin_id,
    '--title', plugin_title,
    '--summary', plugin_summary,
    '--author', plugin_author,
    '--license', plugin_license,
    '--description', plugin_description,
    '--version', meson.project_version(),
    '--cyberether', cyberether_version,
    '--build', cyberether_build_id,
    '--library', '@INPUT0@',
]
foreach example : plugin_examples
    plugin_bundle_args += ['--example', example]
endforeach

custom_target(
    'stelline_cep',
    input: [plugin_module],
    depend_files: [plugin_examples, files('description.md')],
    output: '-'.join(plugin_id, meson.project_version(), 'cyberether', cyberether_version, cyberether_build_id) + '.cep',
    command: plugin_bundle_args,
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
    'Version': cyberether_version,
    'Build ID': cyberether_build_id,
    'Backends': ','.join(cyberether_backends),
}, section: 'CyberEther', bool_yn: true)

summary({
    'CPU': plugin_device_cpu_enabled,
    'CUDA': plugin_device_cuda_enabled,
}, section: 'Devices', bool_yn: true)
