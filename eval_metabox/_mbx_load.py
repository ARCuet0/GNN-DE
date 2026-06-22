import io, pickle, sys, os, types, torch
_MBX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'literature', 'rl_metabbo', 'MetaBox', 'src')

def install_stubs():
    if 'ray' not in sys.modules:
        ray_stub = types.ModuleType('ray')
        def _remote(*a, **k):
            if len(a)==1 and callable(a[0]) and not k: return a[0]
            def deco(fn): return fn
            return deco
        ray_stub.remote=_remote; ray_stub.get=lambda x,**k:x
        ray_stub.wait=lambda x,**k:(x,[]); ray_stub.init=lambda *a,**k:None
        ray_stub.put=lambda x:x; ray_stub.is_initialized=lambda:False
        sys.modules['ray']=ray_stub

def _ensure(dotted):
    parts=dotted.split('.')
    for i in range(1,len(parts)+1):
        name='.'.join(parts[:i])
        if name in sys.modules: continue
        path=_MBX if i==1 else os.path.join(_MBX,*parts[1:i])
        m=types.ModuleType(name); m.__path__=[path]; m.__package__=name
        sys.modules[name]=m

def prep():
    install_stubs()
    for pkg in ['metaevobox','metaevobox.baseline','metaevobox.baseline.metabbo',
                'metaevobox.baseline.bbo','metaevobox.rl','metaevobox.environment',
                'metaevobox.environment.optimizer','metaevobox.environment.problem',
                'metaevobox.environment.parallelenv']:
        _ensure(pkg)
    if _MBX not in sys.path: sys.path.insert(0,_MBX)

class _U(pickle.Unpickler):
    def find_class(self, module, name):
        if module=='torch.storage' and name=='_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b),map_location='cpu',weights_only=False)
        if module.startswith('metaevobox') and '.' in module:
            _ensure(module.rsplit('.',1)[0])
        return super().find_class(module,name)

def load_ckpt(path):
    prep()
    with open(path,'rb') as f:
        return _U(f).load()

import importlib
def imp(rel):
    """Import metaevobox.<rel>, ensuring the container-package skeleton first so
    the leaf .py loads without executing the heavy metaevobox __init__ chain."""
    prep()
    dotted = 'metaevobox.' + rel
    _ensure(dotted.rsplit('.', 1)[0])
    return importlib.import_module(dotted)
