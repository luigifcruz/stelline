project('daqiri', ['cpp', 'cuda'],
  version: '2026.7.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=0',
    'buildtype=release'
  ]
)

cpp = meson.get_compiler('cpp')

# Vendored yaml-cpp (git submodule). Public daqiri headers include yaml-cpp/yaml.h,
# so both the library and its consumers need these headers.
yaml_cpp_inc = include_directories('third_party/yaml-cpp/include')

yaml_cpp_lib = static_library('daqiri-yaml-cpp',
  files(
    'third_party/yaml-cpp/src/binary.cpp',
    'third_party/yaml-cpp/src/convert.cpp',
    'third_party/yaml-cpp/src/depthguard.cpp',
    'third_party/yaml-cpp/src/directives.cpp',
    'third_party/yaml-cpp/src/emit.cpp',
    'third_party/yaml-cpp/src/emitfromevents.cpp',
    'third_party/yaml-cpp/src/emitter.cpp',
    'third_party/yaml-cpp/src/emitterstate.cpp',
    'third_party/yaml-cpp/src/emitterutils.cpp',
    'third_party/yaml-cpp/src/exceptions.cpp',
    'third_party/yaml-cpp/src/exp.cpp',
    'third_party/yaml-cpp/src/fptostring.cpp',
    'third_party/yaml-cpp/src/memory.cpp',
    'third_party/yaml-cpp/src/node.cpp',
    'third_party/yaml-cpp/src/node_data.cpp',
    'third_party/yaml-cpp/src/nodebuilder.cpp',
    'third_party/yaml-cpp/src/nodeevents.cpp',
    'third_party/yaml-cpp/src/null.cpp',
    'third_party/yaml-cpp/src/ostream_wrapper.cpp',
    'third_party/yaml-cpp/src/parse.cpp',
    'third_party/yaml-cpp/src/parser.cpp',
    'third_party/yaml-cpp/src/regex_yaml.cpp',
    'third_party/yaml-cpp/src/scanner.cpp',
    'third_party/yaml-cpp/src/scanscalar.cpp',
    'third_party/yaml-cpp/src/scantag.cpp',
    'third_party/yaml-cpp/src/scantoken.cpp',
    'third_party/yaml-cpp/src/simplekey.cpp',
    'third_party/yaml-cpp/src/singledocparser.cpp',
    'third_party/yaml-cpp/src/stream.cpp',
    'third_party/yaml-cpp/src/tag.cpp',
  ),
  include_directories: yaml_cpp_inc,
  pic: true,
)

# Vendored spdlog (git submodule). Used header-only, matching the CMake build.
spdlog_inc = include_directories('third_party/spdlog/include')

# The overlay ships a pre-generated daqiri/version.h under gen/include.
gen_inc = include_directories('gen/include')

public_inc = [
  include_directories('include'),
  gen_inc,
  spdlog_inc,
  yaml_cpp_inc,
]

# Sources include their headers as "src/...", relative to the repository root.
private_inc = [include_directories('.')]

# The ibverbs raw-Ethernet engine is added below when libibverbs and libmlx5
# are available. The socket engine must always be compiled: the engine factory
# requires one of dpdk, socket, or rdma as the default engine type, and
# ibverbs is only reachable through explicit selection. The dpdk and rdma
# engines stay compiled out, so their DAQIRI_ENGINE_* macros are intentionally
# left undefined.
compile_defs = [
  '-DALLOW_EXPERIMENTAL_API',
  '-DDOCA_ALLOW_EXPERIMENTAL_API',
  '-DDAQIRI_ENABLE_OTEL_METRICS=0',
  '-DDAQIRI_ENABLE_GDS=0',
  '-DDAQIRI_ENABLE_S3=0',
  '-DDAQIRI_ENGINE_SOCKET=1',
]

numa_dep = cpp.find_library('numa', required: false)
if numa_dep.found() and cpp.has_header('numa.h')
  compile_defs += '-DDAQIRI_HAVE_NUMA=1'
else
  compile_defs += '-DDAQIRI_HAVE_NUMA=0'
endif

cuda_dep = dependency('cuda', modules: ['cudart', 'cuda'], required: true)

deps = [
  dependency('threads'),
  cuda_dep,
]

if numa_dep.found()
  deps += numa_dep
endif

# The ibverbs engine requires the Mellanox libraries plus the DevX headers.
mlx5_dep = cpp.find_library('mlx5', required: false)
ibverbs_dep = cpp.find_library('ibverbs', required: false)
dl_dep = cpp.find_library('dl', required: false)

foreach extra_dep : [mlx5_dep, ibverbs_dep, dl_dep]
  if extra_dep.found()
    deps += extra_dep
  endif
endforeach

sources = files(
  'src/common.cpp',
  'src/common_gds.cpp',
  'src/common_s3.cpp',
  'src/engine.cpp',
  'src/engines/socket/daqiri_socket_engine.cpp',
  'src/logging.cpp',
)

have_ibverbs_engine = mlx5_dep.found() and ibverbs_dep.found() and \
                      cpp.has_header('infiniband/verbs.h') and \
                      cpp.has_header('infiniband/mlx5dv.h')

if have_ibverbs_engine
  compile_defs += '-DDAQIRI_ENGINE_IBVERBS=1'
  sources += files('src/engines/ibverbs/daqiri_ibverbs_engine.cpp')
endif

cuda_sources = files(
  'src/kernels.cu',
)

daqiri_lib = library('daqiri',
  sources + cuda_sources,
  include_directories: private_inc + public_inc,
  dependencies: deps,
  cpp_args: compile_defs,
  cuda_args: compile_defs + [
    '--diag-suppress=1217',
    '-Xcompiler=-fPIC',
  ],
  link_with: yaml_cpp_lib,
  install: true,
  version: meson.project_version(),
  soversion: '0',
  gnu_symbol_visibility: 'default',
)

daqiri_dep = declare_dependency(
  link_with: [daqiri_lib, yaml_cpp_lib],
  include_directories: public_inc,
  dependencies: deps,
  compile_args: compile_defs,
)
