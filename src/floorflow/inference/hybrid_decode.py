"""Class-group routing between multiple generated floorplans."""

from __future__ import annotations

import numpy as np

from floorflow.data.taxonomy import FloorplanTaxonomy


def hybrid_decode(
    primary: np.ndarray,
    secondary: np.ndarray,
    taxonomy: FloorplanTaxonomy,
    *,
    secondary_classes: tuple[str, ...] = ("furniture",),
) -> np.ndarray:
    output = primary.copy()
    if "furniture" in secondary_classes:
        output[np.isin(secondary, taxonomy.furniture_ids)] = secondary[np.isin(secondary, taxonomy.furniture_ids)]
    if "opening" in secondary_classes:
        opening = np.isin(secondary, taxonomy.opening_ids)
        output[opening] = secondary[opening]
    if "wall" in secondary_classes:
        wall = secondary == taxonomy.class_id("wall")
        output[wall] = secondary[wall]
    return output
