from typing import Any, Dict, Optional, Tuple

from holoscan.core import Application, Resource

from ..operators import BeamformerOp, CorrelatorOp, FrbnnOp
from ..registry import register_bit
from ..types import BlockShape
from ..utils import logger


@register_bit("blade_bit")
def BladeBit(
    app: Application,
    pool: Resource,
    id: int,
    config: str,
) -> Tuple[Any, Any]:
    cfg = app.kwargs(config)

    input_shape_config = cfg.get("input_shape")
    output_shape_config = cfg.get("output_shape")
    mode = cfg.get("mode")
    number_of_buffers = cfg.get("number_of_buffers") or 4
    options = cfg.get("options") or {}

    logger.info("Blade Configuration:")
    logger.info(f"  Input Shape: {input_shape_config}")
    logger.info(f"  Output Shape: {output_shape_config}")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Number of Buffers: {number_of_buffers}")
    logger.info(f"  Options: {options}")

    input_block_shape = BlockShape(
        number_of_antennas=input_shape_config["number_of_antennas"],
        number_of_channels=input_shape_config["number_of_channels"],
        number_of_samples=input_shape_config["number_of_samples"],
        number_of_polarizations=input_shape_config["number_of_polarizations"],
    )
    output_block_shape = BlockShape(
        number_of_antennas=output_shape_config["number_of_antennas"],
        number_of_channels=output_shape_config["number_of_channels"],
        number_of_samples=output_shape_config["number_of_samples"],
        number_of_polarizations=output_shape_config["number_of_polarizations"],
    )

    if mode == "correlator":
        logger.info("Creating Correlator operator.")
        blade_name = f"blade-correlator-{id}"
        blade_op = CorrelatorOp(
            fragment=app,
            number_of_buffers=number_of_buffers,
            input_shape=input_block_shape,
            output_shape=output_block_shape,
            options={k: str(v) for k, v in options.items()},
            name=blade_name,
        )
    elif mode == "beamformer":
        logger.info("Creating Beamformer operator.")
        blade_name = f"blade-beamformer-{id}"
        blade_op = BeamformerOp(
            fragment=app,
            number_of_buffers=number_of_buffers,
            input_shape=input_block_shape,
            output_shape=output_block_shape,
            options={k: str(v) for k, v in options.items()},
            name=blade_name,
        )
    elif mode == "frbnn":
        logger.info("Creating FRBNN operator.")
        blade_name = f"blade-frbnn-{id}"
        blade_op = FrbnnOp(
            fragment=app,
            number_of_buffers=number_of_buffers,
            input_shape=input_block_shape,
            output_shape=output_block_shape,
            options={k: str(v) for k, v in options.items()},
            name=blade_name,
        )
    else:
        raise ValueError(f"Unsupported blade mode: {mode}")

    return (blade_op, blade_op)


__all__ = ["BladeBit"]
