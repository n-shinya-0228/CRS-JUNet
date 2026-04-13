from .PSalsaNext import UnpNet
from .JunNet13 import JunNet13
from .CRS_JUNet import CRS_JUNet
from .SJunNet11 import SJunNet11
from .SJunNet12 import SJunNet12
from .SJNet import SJNet
from .SJNetl import SJNetl

def get_model(model):
    return eval(model)
