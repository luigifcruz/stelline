assets = [
    'frbnn/test/offline/input.bin',
    'frbnn/test/offline/output.bin',
]

foreach asset : assets
    url = cdn_url + asset
    filename = asset.split('/')[-1]

    run_command(curl, '-L', '-o', filename, url, check: true)

    configure_file(
        input: filename,
        output: filename,
        copy: true
    )
endforeach
