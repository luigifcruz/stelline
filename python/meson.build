py = import('python').find_installation('python3', required: false)

if not py.found()
    warning('Python installation not found. Skipping Python bindings.')
    subdir_done()
endif

py_install_path = py.get_install_dir()

pybind11_dep = dependency('pybind11', fallback: ['pybind11', 'pybind11_dep'])

extension_rpath = '$ORIGIN/../..:' + get_option('prefix') / get_option('libdir')

py.extension_module(
    '_types',
    sources: '../src/bindings/types.cc',
    include_directories: inc_lst,
    dependencies: [pybind11_dep, py.dependency()] + dep_lst,
    link_with: lib_stelline,
    install: true,
    install_dir: py_install_path / 'stelline',
    install_rpath: extension_rpath,
    gnu_symbol_visibility: 'hidden'
)

py.extension_module(
    '_logging',
    sources: '../src/bindings/logging.cc',
    include_directories: inc_lst,
    dependencies: [pybind11_dep, py.dependency()] + dep_lst,
    link_with: lib_stelline,
    install: true,
    install_dir: py_install_path / 'stelline',
    install_rpath: extension_rpath,
    gnu_symbol_visibility: 'hidden'
)

py.extension_module(
    '_blade_ops',
    sources: '../src/bindings/operators/blade_ops.cc',
    include_directories: inc_lst,
    dependencies: [pybind11_dep, py.dependency()] + dep_lst,
    link_with: lib_stelline,
    install: true,
    install_dir: py_install_path / 'stelline/operators',
    install_rpath: extension_rpath,
    gnu_symbol_visibility: 'hidden'
)

py.extension_module(
    '_filesystem_ops',
    sources: '../src/bindings/operators/filesystem_ops.cc',
    include_directories: inc_lst,
    dependencies: [pybind11_dep, py.dependency()] + dep_lst,
    link_with: lib_stelline,
    install: true,
    install_dir: py_install_path / 'stelline/operators',
    install_rpath: extension_rpath,
    gnu_symbol_visibility: 'hidden'
)

py.extension_module(
    '_transport_ops',
    sources: '../src/bindings/operators/transport_ops.cc',
    include_directories: inc_lst,
    dependencies: [pybind11_dep, py.dependency()] + dep_lst,
    link_with: lib_stelline,
    install: true,
    install_dir: py_install_path / 'stelline/operators',
    install_rpath: extension_rpath,
    gnu_symbol_visibility: 'hidden'
)

py.extension_module(
    '_socket_ops',
    sources: '../src/bindings/operators/socket_ops.cc',
    include_directories: inc_lst,
    dependencies: [pybind11_dep, py.dependency()] + dep_lst,
    link_with: lib_stelline,
    install: true,
    install_dir: py_install_path / 'stelline/operators',
    install_rpath: extension_rpath,
    gnu_symbol_visibility: 'hidden'
)

py.extension_module(
    '_frbnn_ops',
    sources: '../src/bindings/operators/frbnn_ops.cc',
    include_directories: inc_lst,
    dependencies: [pybind11_dep, py.dependency()] + dep_lst,
    link_with: lib_stelline,
    install: true,
    install_dir: py_install_path / 'stelline/operators',
    install_rpath: extension_rpath,
    gnu_symbol_visibility: 'hidden'
)

py.extension_module(
    '_advanced_network_ops',
    sources: '../src/bindings/operators/advanced_network_ops.cc',
    include_directories: inc_lst,
    dependencies: [pybind11_dep, py.dependency()] + dep_lst,
    link_with: lib_stelline,
    install: true,
    install_rpath: extension_rpath,
    gnu_symbol_visibility: 'hidden',
    subdir: 'stelline/operators'
)

install_subdir(
    'stelline',
    install_dir: py_install_path,
    exclude_files: ['__pycache__'],
    exclude_directories: ['__pycache__']
)
