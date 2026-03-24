from .PSalsaNext import UnpNet
from .JunNet import JunNet
from .JunNet2 import JunNet2
from .JunNet3 import JunNet3
from .JunNet4 import JunNet4
from .JunNet5 import JunNet5
from .JunNet6 import JunNet6
from .JunNet7 import JunNet7
from .JunNet8 import JunNet8
from .JunNet9 import JunNet9
from .JunNet10 import JunNet10
from .JunNet11 import JunNet11
from .JunNet12 import JunNet12
from .JunNet13 import JunNet13
from .CRS_JUNet import CRS_JUNet
from .SJunNet import SJunNet
from .SJunNet2 import SJunNet2
from .SJunNet3 import SJunNet3
from .SJunNet4 import SJunNet4
from .SJunNet5 import SJunNet5
from .SJunNet6 import SJunNet6
from .SJunNet7 import SJunNet7


def get_model(model):
    return eval(model)
