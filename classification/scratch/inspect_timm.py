import timm
print("timm version:", timm.__version__)

try:
    from timm.utils import ModelEmaV2
    print("Found ModelEmaV2 in timm.utils")
    import inspect
    print(inspect.getsource(ModelEmaV2))
except ImportError:
    print("ModelEmaV2 not found in timm.utils")

try:
    from timm.utils import ModelEmaV3
    print("Found ModelEmaV3 in timm.utils")
except ImportError:
    print("ModelEmaV3 not found in timm.utils")
