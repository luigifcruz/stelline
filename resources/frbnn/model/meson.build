models = [
    {'asset': 'frbnn/model/frbnn_preprocessor.onnx', 'shape': '192x2048'},
    {'asset': 'frbnn/model/frbnn.onnx', 'shape': '24x192x256'},
]
batch_sizes = ['32', '64']

foreach model : models
    url = cdn_url + model['asset']
    shape = model['shape']

    onnx_filename = url.split('/')[-1]

    if not fs.exists(onnx_filename)
        run_command(curl, '-L', '-o', onnx_filename, url, check: true)
    else
        message('Model file `@0@` already exists. Skipping download.'.format(onnx_filename))
    endif

    foreach batch_size : batch_sizes
        engine_filename = onnx_filename.split('.')[0] + '_' + batch_size + '.trt'
        engine_shape = batch_size + 'x' + shape

        trtexec_cmd = [
            trtexec,
            '--onnx=' + onnx_filename,
            '--saveEngine=' + engine_filename,
            '--shapes=modelInput:' + engine_shape
        ]

        if not fs.exists(engine_filename)
            run_command(trtexec_cmd, check: true)
        else
            message('File `@0@` already exists. Skipping engine generation.'.format(engine_filename))
        endif

        configure_file(
            input: engine_filename,
            output: engine_filename,
            copy: true,
            # TODO: Add install.
        )
    endforeach
endforeach
