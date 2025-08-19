import numpy as np
from pint import UnitRegistry


ureg = UnitRegistry()
def convert_unit(value:float|np.ndarray, from_unit:str, to_unit:str):
    return (value * ureg(from_unit)).to(to_unit).magnitude  # Compute conversion factor. Magnitude here preserves the sign of the value.
def get_unit_conversion(from_unit:str, to_unit:str):
    return convert_unit(value=1, from_unit=from_unit, to_unit=to_unit)
def pretty_unit(unit:str):
    return f'{ureg(unit).u:~P}'

class KeyAwareDefaultDict(dict):
    def __init__(self, factory, **kwargs):
        self.factory = factory
        self.update(**kwargs)
    def __missing__(self,key:str):
        self[key] = self.factory(key)
        return self[key]