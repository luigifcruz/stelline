custom_target(
    'default.yaml',
    input: 'default.yaml',
    output:  'default.yaml',
    command: ['cp', '@INPUT@', '@OUTPUT@'],
    install: false,
    build_by_default: true
)

app_src_lst = [
    'main.cc',
]

# TODO: Remove once CDN is up.
# subdir('data')

executable(
    'frbnn-test-offline',
    app_src_lst,
    dependencies: lib_stelline_dep,
    install: false,
)
