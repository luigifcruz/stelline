# Models

No manual action is necessary. Meson will download and compile the latest weights.

Download the model ONNX files from the following links:
- Pre-processor: TBD
- Model: TBD

## Compile the ONNX into TensorRT Engine

Run these inside the docker container:
```
$ trtexec --onnx=frbnn.onnx \
          --saveEngine=frbnn.trt \
          --shapes=modelInput:32x3x192x2048

$ trtexec --onnx=frbnn_preprocessor.onnx \
          --saveEngine=frbnn_preprocessor.trt \
          --shapes=modelInput:32x192x2048
```
