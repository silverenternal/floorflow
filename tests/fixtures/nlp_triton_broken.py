import json

print(
    json.dumps(
        {
            "compiled": True,
            "correct": False,
            "max_abs_error": 1.0,
            "shape": [257],
        }
    )
)
