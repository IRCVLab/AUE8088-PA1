# Python packages
from termcolor import colored
from types import ModuleType


def dict_from_module(module):
    context = {}
    for setting in dir(module):
        if not setting.startswith("_"):
            v = getattr(module, setting)
            if not isinstance(v, ModuleType):
                context[setting] = getattr(module, setting)
    return context

def blue(msg):
    return colored(msg, color='blue', attrs=('bold',))

def show_setting(setting):
    for k, v in dict_from_module(setting).items():
        print(blue(k), end=': ')
        print(v)
