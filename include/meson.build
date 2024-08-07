subdir('jetstream')

if not is_static
   install_subdir(
      'stelline',
      install_dir: get_option('includedir'),
      exclude_files: [
         'config.hh.in',
         'meson.build',
      ],
   )
endif
