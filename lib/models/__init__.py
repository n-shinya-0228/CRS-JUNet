from .PSalsaNext import UnpNet
# from .JunNet import JunNet
# from .GemiNet import GemiNet
# from .JunNet2 import JunNet2
# from .JunNet3 import JunNet3
# from .JunNet4 import JunNet4
# from .JunNet5 import JunNet5
# from .ChatNet import ChatNet
# from .ChatNet2 import ChatNet2
# from .JunNet6 import JunNet6
# from .JunNet7 import JunNet7
# from .ChatNet3 import ChatNet3
# from .ChatNet4 import ChatNet4
# from .JunNet8 import JunNet8
# from .JunNet9 import JunNet9
# from .JunNet10 import JunNet10
# from .JunNet11 import JunNet11
# from .JunNet12 import JunNet12
from .JunNet13 import JunNet13


def get_model(model):
    return eval(model)
