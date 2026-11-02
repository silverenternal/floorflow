# Strict-Locality Stress Tests

| Seed | Suite | N | Raw outside | Gated outside | Gated room | Validity | Watertight | Inside change |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 19 | wall_adjacent | 2230 | 0.988875 | 1.000000 | 1.000000 | 1.000000 | 0.967265 | 0.315042 |
| 19 | opening_covering | 2130 | 0.988421 | 1.000000 | 1.000000 | 1.000000 | 0.965728 | 0.319092 |
| 19 | exterior_boundary | 2073 | 0.988231 | 1.000000 | 1.000000 | 1.000000 | 0.964785 | 0.330987 |
| 20 | wall_adjacent | 2230 | 0.990256 | 1.000000 | 1.000000 | 1.000000 | 0.969955 | 0.314688 |
| 20 | opening_covering | 2130 | 0.989968 | 1.000000 | 1.000000 | 1.000000 | 0.966667 | 0.318789 |
| 20 | exterior_boundary | 2073 | 0.989682 | 1.000000 | 1.000000 | 1.000000 | 0.967197 | 0.330757 |
| 21 | wall_adjacent | 2230 | 0.968311 | 1.000000 | 1.000000 | 1.000000 | 0.972197 | 0.314734 |
| 21 | opening_covering | 2130 | 0.967116 | 1.000000 | 1.000000 | 1.000000 | 0.968075 | 0.318741 |
| 21 | exterior_boundary | 2073 | 0.966377 | 1.000000 | 1.000000 | 1.000000 | 0.969609 | 0.330695 |
| 22 | wall_adjacent | 2230 | 0.991062 | 1.000000 | 1.000000 | 1.000000 | 0.975336 | 0.314759 |
| 22 | opening_covering | 2130 | 0.990773 | 1.000000 | 1.000000 | 1.000000 | 0.970892 | 0.318833 |
| 22 | exterior_boundary | 2073 | 0.990521 | 1.000000 | 1.000000 | 1.000000 | 0.972021 | 0.330760 |
| 23 | wall_adjacent | 2230 | 0.984353 | 1.000000 | 1.000000 | 1.000000 | 0.975785 | 0.314998 |
| 23 | opening_covering | 2130 | 0.983728 | 1.000000 | 1.000000 | 1.000000 | 0.973239 | 0.319045 |
| 23 | exterior_boundary | 2073 | 0.983412 | 1.000000 | 1.000000 | 1.000000 | 0.975398 | 0.331003 |

Wall-adjacent means the edit neighborhood contains a wall and at least two target room types. Opening-covering means a one-pixel edit dilation intersects a door/window. Exterior-boundary means a five-pixel dilation crosses the footprint.
