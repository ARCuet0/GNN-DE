"""Import select MetaBox classes from the local clone WITHOUT triggering the
heavy ``src/__init__.py`` chain (which pulls Trainer -> learnable optimizers ->
tianshou/ray/evox, none of which are installed in the tersq venv).

Trick: register synthetic skeleton packages (empty module objects whose
``__path__`` points at the real source dirs) under a private root name, then let
the normal import machinery locate the leaf submodules via those paths. The real
``__init__.py`` files are never executed because the package names already exist
in ``sys.modules``. Relative imports inside the leaves resolve through the same
skeletons.

If ``metaevobox`` is pip-installed (the Magerit / full-run case), callers should
prefer the real package; this module is the no-install local fallback.
"""
import importlib
import importlib.util
import os
import sys
import types

_THIS = os.path.dirname(os.path.abspath(__file__))
_TERSQ_ROOT = os.path.dirname(_THIS)
_MBX_SRC = os.path.join(_TERSQ_ROOT, 'literature', 'rl_metabbo', 'MetaBox', 'src')
_ROOT = '_mbx'  # synthetic package root mapped onto MetaBox/src


def _ensure_skeleton(dotted):
    """Register empty skeleton packages for every prefix of ``dotted`` so the
    import machinery can find real submodules via ``__path__`` without running
    any real ``__init__.py``."""
    parts = dotted.split('.')
    for i in range(1, len(parts) + 1):
        name = '.'.join(parts[:i])
        if name in sys.modules:
            continue
        path = _MBX_SRC if i == 1 else os.path.join(_MBX_SRC, *parts[1:i])
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        mod.__package__ = name
        if i > 1:
            setattr(sys.modules['.'.join(parts[:i - 1])], parts[i - 1], mod)
        sys.modules[name] = mod


def _import(real_dotted_under_src):
    """Import ``environment.optimizer.basic_optimizer`` style dotted path
    (relative to MetaBox/src) via the skeleton, returning the leaf module.
    Handles a top-level module under src (e.g. ``logger``) whose parent is the
    root skeleton itself."""
    if '.' in real_dotted_under_src:
        pkg = _ROOT + '.' + real_dotted_under_src.rsplit('.', 1)[0]
    else:
        pkg = _ROOT
    _ensure_skeleton(pkg)
    full = _ROOT + '.' + real_dotted_under_src
    return importlib.import_module(full)


def get_basic_optimizer():
    try:
        from metaevobox.environment.optimizer.basic_optimizer import Basic_Optimizer
        return Basic_Optimizer
    except Exception:
        pass
    mod = _import('environment.optimizer.basic_optimizer')
    return mod.Basic_Optimizer


def get_protein_dataset():
    try:
        from metaevobox.environment.problem.SOO.PROTEIN_DOCKING.protein_docking_dataset import (
            Protein_Docking_Dataset,
        )
        return Protein_Docking_Dataset
    except Exception:
        pass
    mod = _import('environment.problem.SOO.PROTEIN_DOCKING.protein_docking_dataset')
    return mod.Protein_Docking_Dataset


def load_protein_problem(version='numpy', difficulty='all', index=0):
    """Return one MetaBox protein-docking problem instance (default: first of
    the 280-instance 'all' test set)."""
    ds = get_protein_dataset()
    _, test_set = ds.get_datasets(version=version, difficulty=difficulty)
    return test_set.data[index]


def load_protein_testset(version='numpy', difficulty='all'):
    """Return the full MetaBox protein-docking test set (280 instances under
    difficulty='all')."""
    ds = get_protein_dataset()
    _, test_set = ds.get_datasets(version=version, difficulty=difficulty)
    return test_set.data


def get_random_search():
    """MetaBox Random_search (the AEI normalization reference baseline)."""
    try:
        from metaevobox.baseline.bbo.random_search import Random_search
        return Random_search
    except Exception:
        pass
    # Ensure basic_optimizer (and its environment.* skeletons) are loaded first
    # so random_search's `from ...environment.optimizer.basic_optimizer` import
    # resolves from cache rather than traversing the heavy environment/__init__.
    get_basic_optimizer()
    mod = _import('baseline.bbo.random_search')
    return mod.Random_search


def get_basic_logger():
    """MetaBox Basic_Logger (carries the real AEI machinery: get_random_baseline,
    aei_cost, cal_aei). __init__ only stores config, so it is safe to build."""
    try:
        from metaevobox.logger import Basic_Logger
        return Basic_Logger
    except Exception:
        pass
    # logger.py lives directly under src/ (not a subpackage), import as leaf.
    return _import('logger').Basic_Logger
