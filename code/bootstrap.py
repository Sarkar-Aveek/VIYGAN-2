import os
import sys
import types
import importlib.abc
import importlib.util

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_env(path=os.path.join(ROOT, ".env")):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


class _FunctionalTensorShim(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    NAME = "torchvision.transforms.functional_tensor"

    def find_spec(self, name, path=None, target=None):
        return importlib.util.spec_from_loader(name, self) if name == self.NAME else None

    def create_module(self, spec):
        from torchvision.transforms.functional import rgb_to_grayscale
        module = types.ModuleType(spec.name)
        module.rgb_to_grayscale = rgb_to_grayscale
        return module

    def exec_module(self, module):
        pass


load_env()
if not any(isinstance(f, _FunctionalTensorShim) for f in sys.meta_path):
    sys.meta_path.append(_FunctionalTensorShim())   # last, so a real module (older torchvision) still wins
