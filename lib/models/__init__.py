from .PSalsaNext import UnpNet
from .JunNet13 import JunNet13
from .CRS_JUNet import CRS_JUNet
from .SJunNet4 import SJunNet4
from .SJunNet8 import SJunNet8


def get_model(model):
    return eval(model)
