project('h5dsc99', 'c',
  version: '0.4.1',
  default_options: [
    'c_std=c99',
    'buildtype=release',
    'b_ndebug=if-release'
  ]
)

src_lst = []
inc_lst = [
	include_directories('include'),
]
hdf5_dep = dependency('hdf5', language: 'c')
dep_lst = [
  hdf5_dep
]

subdir('include')
subdir('src')

lib_h5dsc99 = library(
  'h5dsc99',
  src_lst,
  include_directories: inc_lst,
  dependencies: dep_lst,
  install: true,
)

lib_h5dsc99_dep = declare_dependency(
  include_directories: inc_lst,
  dependencies: dep_lst,
  link_with: lib_h5dsc99,
)

pkg = import('pkgconfig')
pkg.generate(
  lib_h5dsc99,
  libraries: dep_lst,
)

subdir('tests')
