"""
FRBNN Bits Module
"""

from typing import Tuple, Any

from holoscan.core import Application
from holoscan.operators import InferenceOp

from ..operators import (
    ModelPreprocessorOp,
    ModelAdapterOp,
    ModelPostprocessorOp,
    SimpleDetectionOp,
)
from ..utils import logger


def FrbnnInferenceBit(
    app: Application, pool: Any, id: int, config: str
) -> Tuple[Any, Any]:
    """
    Create an FRBNN inference bit from configuration.

    This function creates FRBNN inference operators for detecting Fast Radio Bursts
    using trained neural network models. The operator processes DSP blocks and
    outputs detection results for further processing.

    Parameters
    ----------
    app : holoscan.Application
        The Holoscan application instance
    pool : Any
        Memory pool resource
    id : int
        Unique identifier for this bit instance
    config : str
        Configuration key to fetch from the application's YAML config

    Returns
    -------
    Tuple[Any, Any]
        A tuple of (input_operator, output_operator) representing the bit interface
    """

    cfg = app.kwargs(config)

    frbnn_preprocessor_path = cfg["frbnn_preprocessor_path"]
    frbnn_path = cfg["frbnn_path"]

    logger.info("FRBNN Inference Configuration:")
    logger.info(f"  Preprocessor Path: {frbnn_preprocessor_path}")
    logger.info(f"  Model Path: {frbnn_path}")

    # Configure app for metadata
    app.is_metadata_enabled(True)

    # Build FRBNN Preprocessor configuration
    frbnn_preprocessor_path_map = {"frbnn_preprocessor": frbnn_preprocessor_path}
    frbnn_preprocessor_input_map = {"frbnn_preprocessor": ["input"]}
    frbnn_preprocessor_output_map = {"frbnn_preprocessor": ["output"]}

    # Build FRBNN configuration
    frbnn_path_map = {"frbnn": frbnn_path}
    frbnn_input_map = {"frbnn": ["input"]}
    frbnn_output_map = {"frbnn": ["output"]}

    # Create operators with exact naming
    model_preprocessor_name = f"frbnn-model-preprocessor-{id}"
    model_preprocessor_op = ModelPreprocessorOp(
        fragment=app,
        name=model_preprocessor_name,
    )

    frbnn_preprocessor_inference_name = f"frbnn-preprocessor-inference-{id}"
    frbnn_preprocessor_inference_op = InferenceOp(
        fragment=app,
        name=frbnn_preprocessor_inference_name,
        backend="trt",
        model_path_map=frbnn_preprocessor_path_map,
        pre_processor_map=frbnn_preprocessor_input_map,
        inference_map=frbnn_preprocessor_output_map,
        infer_on_cpu=False,
        input_on_cuda=True,
        output_on_cuda=True,
        transmit_on_cuda=True,
        is_engine_path=True,
        allocator=pool,
    )

    model_adapter_name = f"frbnn-model-adapter-{id}"
    model_adapter_op = ModelAdapterOp(
        fragment=app,
        name=model_adapter_name,
    )

    frbnn_inference_name = f"frbnn-inference-{id}"
    frbnn_inference_op = InferenceOp(
        fragment=app,
        name=frbnn_inference_name,
        backend="trt",
        model_path_map=frbnn_path_map,
        pre_processor_map=frbnn_input_map,
        inference_map=frbnn_output_map,
        infer_on_cpu=False,
        input_on_cuda=True,
        output_on_cuda=True,
        transmit_on_cuda=True,
        is_engine_path=True,
        allocator=pool,
    )

    model_postprocessor_name = f"frbnn-model-postprocessor-{id}"
    model_postprocessor_op = ModelPostprocessorOp(
        fragment=app,
        name=model_postprocessor_name,
    )

    # Connect operators
    app.add_flow(
        model_preprocessor_op, frbnn_preprocessor_inference_op, {("out", "receivers")}
    )
    app.add_flow(
        frbnn_preprocessor_inference_op, model_adapter_op, {("transmitter", "in")}
    )
    app.add_flow(model_adapter_op, frbnn_inference_op, {("out", "receivers")})
    app.add_flow(frbnn_inference_op, model_postprocessor_op, {("transmitter", "in")})

    return (model_preprocessor_op, model_postprocessor_op)


def FrbnnDetectionBit(
    app: Application, pool: Any, id: int, config: str
) -> Tuple[Any, Any]:
    """
    Create an FRBNN detection bit from configuration.

    This function creates FRBNN detection operators for post-processing
    inference results, applying additional filtering, and saving detection
    results to files.

    Parameters
    ----------
    app : holoscan.Application
        The Holoscan application instance
    pool : Any
        Memory pool resource
    id : int
        Unique identifier for this bit instance
    config : str
        Configuration key to fetch from the application's YAML config

    Returns
    -------
    Tuple[Any, Any]
        A tuple of (detection_op, detection_op) representing the bit interface
    """

    cfg = app.kwargs(config)

    csv_file_path = cfg["csv_file_path"]
    hits_directory = cfg["hits_directory"]

    logger.info("FRBNN Detection Configuration:")
    logger.info(f"  CSV File Path: {csv_file_path}")
    logger.info(f"  Hits Directory: {hits_directory}")

    # Configure app for metadata
    app.is_metadata_enabled(True)

    # Create detection operator
    frbnn_simple_detection_name = f"frbnn-simple-detection-{id}"
    frbnn_simple_detection_op = SimpleDetectionOp(
        fragment=app,
        name=frbnn_simple_detection_name,
        csv_file_path=csv_file_path,
        hits_directory=hits_directory,
    )

    return (frbnn_simple_detection_op, frbnn_simple_detection_op)
