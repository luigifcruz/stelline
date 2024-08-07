models = [
    {'asset': 'frbnn/frbnn_preprocessor.onnx', 'shape': '192x2048'},
    {'asset': 'frbnn/frbnn.onnx', 'shape': '3x192x2048'},
]
batch_sizes = ['32', '64']

foreach model : models
    url = cdn_url + model['asset']
    shape = model['shape']

    filename = url.split('/')[-1]
    run_command(curl, '-L', '-o', filename, url, check: true)

    foreach batch_size : batch_sizes
        engine_name = filename.split('.')[0] + '_' + batch_size + '.trt'
        engine_shape = batch_size + 'x' + shape

        trtexec_cmd = [
            trtexec,
            '--onnx=' + filename,
            '--saveEngine=' + engine_name,
            '--shapes=modelInput:' + engine_shape
        ]

        run_command(trtexec_cmd, check: true)

        configure_file(
            input: engine_name,
            output: engine_name,
            copy: true
        )
    endforeach
endforeach
