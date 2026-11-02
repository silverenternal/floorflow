"""Shared semantic label space for CubiCasa5K floorplans."""

from __future__ import annotations

BACKGROUND = 0
WALL = 1
DOOR = 2
WINDOW = 3
LIVING = 4
BEDROOM = 5
KITCHEN = 6
BATHROOM = 7
CIRCULATION = 8
STORAGE = 9
OTHER_ROOM = 10
TOILET = 11
SINK = 12
SHOWER = 13
CLOSET = 14
APPLIANCE = 15
SAUNA = 16
FIREPLACE = 17
FURNITURE_OTHER = 18
STRUCTURE = 19

CLASS_NAMES = (
    "background",
    "wall",
    "door",
    "window",
    "living",
    "bedroom",
    "kitchen",
    "bathroom",
    "circulation",
    "storage",
    "other_room",
    "toilet",
    "sink",
    "shower",
    "closet",
    "appliance",
    "sauna",
    "fireplace",
    "furniture_other",
    "structure",
)
NUM_CLASSES = len(CLASS_NAMES)
NUM_TARGET_CHANNELS = NUM_CLASSES - 1

# RGB preview colors. Class 0 is white so generated drawings remain readable.
CLASS_COLORS = (
    (255, 255, 255),
    (20, 20, 20),
    (220, 40, 40),
    (40, 120, 220),
    (248, 228, 176),
    (184, 210, 245),
    (246, 188, 120),
    (124, 205, 205),
    (218, 218, 218),
    (184, 170, 220),
    (220, 220, 160),
    (118, 164, 196),
    (96, 170, 230),
    (110, 210, 230),
    (168, 142, 112),
    (150, 150, 170),
    (210, 150, 90),
    (200, 90, 60),
    (180, 180, 180),
    (100, 100, 100),
)

ROOM_CLASS_BY_TYPE = {
    "LivingRoom": LIVING,
    "Dining": LIVING,
    "FamilyRoom": LIVING,
    "Office": LIVING,
    "Bedroom": BEDROOM,
    "Kitchen": KITCHEN,
    "Bath": BATHROOM,
    "Bathroom": BATHROOM,
    "Sauna": BATHROOM,
    "WC": BATHROOM,
    "Toilet": BATHROOM,
    "Hall": CIRCULATION,
    "Lobby": CIRCULATION,
    "Entry": CIRCULATION,
    "Corridor": CIRCULATION,
    "DraughtLobby": CIRCULATION,
    "Stairs": CIRCULATION,
    "Closet": STORAGE,
    "Storage": STORAGE,
    "WalkIn": STORAGE,
    "Pantry": STORAGE,
    "Utility": STORAGE,
}

STRUCTURE_IDS = {
    "Wall",
    "Railing",
    "Column",
    "Beam",
    "Elevator",
    "Stair",
    "Stairs",
}

ROOM_CLASS_IDS = tuple(range(LIVING, OTHER_ROOM + 1))
FURNITURE_CLASS_IDS = tuple(range(TOILET, FURNITURE_OTHER + 1))
THIN_STRUCTURE_CLASS_IDS = (DOOR, WINDOW, SHOWER, SAUNA, STRUCTURE)

FURNITURE_CLASS_BY_TYPE = {
    "Toilet": TOILET,
    "Urinal": TOILET,
    "Sink": SINK,
    "DoubleSink": SINK,
    "DoubleSinkRight": SINK,
    "RoundSink": SINK,
    "SideSink": SINK,
    "CornerSink": SINK,
    "WaterTap": SINK,
    "Shower": SHOWER,
    "ShowerScreen": SHOWER,
    "ShowerScreenRoundRight": SHOWER,
    "ShowerScreenRoundLeft": SHOWER,
    "ShowerCab": SHOWER,
    "ShowerPlatform": SHOWER,
    "Bathtub": SHOWER,
    "BathtubRound": SHOWER,
    "Jacuzzi": SHOWER,
    "Closet": CLOSET,
    "ClosetTriangle": CLOSET,
    "ClosetRound": CLOSET,
    "CoatCloset": CLOSET,
    "WallCabinet": CLOSET,
    "BaseCabinet": CLOSET,
    "BaseCabinetRound": CLOSET,
    "BaseCabinetTriangle": CLOSET,
    "CounterTop": CLOSET,
    "CoatRack": CLOSET,
    "ElectricalAppliance": APPLIANCE,
    "Refrigerator": APPLIANCE,
    "WashingMachine": APPLIANCE,
    "Dishwasher": APPLIANCE,
    "IntegratedStove": APPLIANCE,
    "GasStove": APPLIANCE,
    "Housing": APPLIANCE,
    "SaunaBench": SAUNA,
    "SaunaBenchMid": SAUNA,
    "SaunaBenchHigh": SAUNA,
    "SaunaBenchLow": SAUNA,
    "SaunaStove": SAUNA,
    "Fireplace": FIREPLACE,
    "FireplaceCorner": FIREPLACE,
    "FireplaceRound": FIREPLACE,
    "PlaceForFireplace": FIREPLACE,
    "PlaceForFireplaceCorner": FIREPLACE,
    "PlaceForFireplaceRound": FIREPLACE,
    "Chimney": FIREPLACE,
    "WoodStove": FIREPLACE,
}

SEMANTIC_CHANNEL_RELATIONS = (
    (WALL, DOOR, 0.9),
    (WALL, WINDOW, 0.9),
    (WALL, STRUCTURE, 0.6),
    (BATHROOM, TOILET, 1.0),
    (BATHROOM, SINK, 0.8),
    (BATHROOM, SHOWER, 1.0),
    (BATHROOM, SAUNA, 0.8),
    (KITCHEN, SINK, 0.8),
    (KITCHEN, APPLIANCE, 1.0),
    (KITCHEN, CLOSET, 0.6),
    (STORAGE, CLOSET, 1.0),
    (BEDROOM, CLOSET, 0.7),
    (LIVING, FIREPLACE, 0.8),
    (LIVING, FURNITURE_OTHER, 0.4),
)


def room_class(class_name: str) -> int:
    """Map a CubiCasa SVG class string to a stable room category."""

    tokens = class_name.split()
    if "Outdoor" in tokens:
        return BACKGROUND
    for token in tokens[1:]:
        if token in ROOM_CLASS_BY_TYPE:
            return ROOM_CLASS_BY_TYPE[token]
    return OTHER_ROOM


def furniture_class(class_name: str) -> int:
    tokens = class_name.split()
    for token in tokens[1:]:
        if token in FURNITURE_CLASS_BY_TYPE:
            return FURNITURE_CLASS_BY_TYPE[token]
    return FURNITURE_OTHER


def class_ids_for_target() -> tuple[int, ...]:
    return tuple(range(1, NUM_CLASSES))


def semantic_channel_prior() -> tuple[tuple[float, ...], ...]:
    """Relation biases over foreground channels (class IDs 1..N-1)."""

    matrix = [[0.0] * NUM_TARGET_CHANNELS for _ in range(NUM_TARGET_CHANNELS)]
    for source, target, strength in SEMANTIC_CHANNEL_RELATIONS:
        source_index = source - 1
        target_index = target - 1
        matrix[source_index][target_index] = float(strength)
        matrix[target_index][source_index] = float(strength)
    return tuple(tuple(row) for row in matrix)
