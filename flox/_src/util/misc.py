from dataclasses import astuple


def unpack(obj):
    assert hasattr(obj, "__dataclass_fields__"), f"{obj} is not a dataclass"
    for field in obj.__dataclass_fields__:
        yield getattr(obj, field)
