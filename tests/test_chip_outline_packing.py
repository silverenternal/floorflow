import numpy as np

from floorflow.chip_flow.outline_packing import pack_fixed_outline


def test_pack_fixed_outline_keeps_shapes_and_fits() -> None:
    boxes = np.asarray(
        [
            [20.0, 20.0, 3.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )
    result = pack_fixed_outline(boxes, (6.0, 4.0), time_limit_ms=2000)

    assert result is not None
    assert np.allclose(result.xywh[:, 2:], boxes[:, 2:])
    assert result.xywh[:, 0].min() >= 0.0
    assert result.xywh[:, 1].min() >= 0.0
    assert (result.xywh[:, 0] + result.xywh[:, 2]).max() <= 6.0 + 1.0e-7
    assert (result.xywh[:, 1] + result.xywh[:, 3]).max() <= 4.0 + 1.0e-7


def test_pack_fixed_outline_rejects_impossible_block() -> None:
    boxes = np.asarray([[0.0, 0.0, 7.0, 2.0]], dtype=np.float64)
    assert pack_fixed_outline(boxes, (6.0, 4.0)) is None


def test_pack_fixed_outline_can_rotate_a_fixed_rectangle() -> None:
    boxes = np.asarray([[0.0, 0.0, 5.0, 2.0]], dtype=np.float64)
    result = pack_fixed_outline(
        boxes,
        (3.0, 6.0),
        rotatable=np.asarray([True]),
    )

    assert result is not None
    assert np.allclose(result.xywh[0, 2:], [2.0, 5.0])
