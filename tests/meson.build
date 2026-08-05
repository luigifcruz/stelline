catch2_dep = dependency(
    'catch2',
    version: '>=3.0',
    fallback: ['catch2', 'catch2_dep'],
    default_options: [
        'default_library=static',
        'tests=false',
        'cpp_std=c++20',
    ],
)

stelline_test_dependencies = [plugin_code_dep, catch2_dep]
stelline_test_main = static_library(
    'stelline-test-main',
    'main.cc',
    dependencies: stelline_test_dependencies,
)

stelline_architecture_tests = executable(
    'stelline-architecture-tests',
    'architecture.cc',
    dependencies: stelline_test_dependencies,
    link_with: stelline_test_main,
    link_language: 'cpp',
    link_args: ['-Wl,--no-as-needed'],
)

test(
    'architecture',
    stelline_architecture_tests,
    suite: 'stelline',
    is_parallel: false,
    timeout: 60,
)
