"""Let an NVTX-instrumented vLLM build compile (hardware validation on Oxford HTC).

Why. The vLLM the HTC pilot venv runs (Jason's trace build, commit c124fac
"Add canonical NVTX trace labels") wraps model-forward code in

    with record_function_or_nullcontext(<label>):

where <label> is an f-string of layer names, tensor shapes and, for the
tensor-parallel collectives, a ContextVar read (comm_nvtx_label). Those
statements sit inside the region torch.compile traces with fullgraph=True;
Dynamo refuses the ContextVar read ("Unsupported method call: ContextVar.get")
and the model only runs with --enforce-eager: no CUDA graphs, so every decode
step pays per-kernel launch overhead that neither the profiler's kernel
timings nor the simulator carry.

What. An import hook, two parts. (1) For modules under vllm.model_executor and
vllm.distributed whose source calls record_function_or_nullcontext, each
``with record_function_or_nullcontext(ANY):`` item becomes
``with contextlib.nullcontext():`` before the module is compiled, so the label
is never evaluated. With the profiling-scope variables unset (as in the bench)
the original helper returns a nullcontext as well: no computation changes.
The rewritten modules are compiled from source (their cached bytecode is
ignored) and nothing is written anywhere. (2) The same build reads an
undeclared ObservabilityConfig field (track_gpu_coll_op_timings) in its
collective wrapper (and track_moe_stats in its MoE-stats tracer); the class
gets False defaults when it lacks them. (3) BaseRouter._select_experts syncs
the host (.item()) on every call to count invalid expert ids, which breaks
CUDA-graph capture; the count becomes 0 (only the disabled stats path reads it).

How. Put this directory first on PYTHONPATH; every interpreter, including
vLLM's spawned workers, imports sitecustomize at start-up. Set
HWVAL_STRIP_NVTX_LABELS=0 to disable. Rewritten modules and counts are listed
on stderr when HWVAL_STRIP_NVTX_VERBOSE=1.
"""
import ast
import importlib.abc
import importlib.machinery
import os
import sys

_PREFIXES = ("vllm.model_executor.", "vllm.distributed.")
_HELPER = "record_function_or_nullcontext"
_ALIAS = "_hwval_contextlib"
REWRITTEN = []  # (module path, with-items rewritten)


def _is_helper_call(node):
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Name) and f.id == _HELPER) or (isinstance(f, ast.Attribute) and f.attr == _HELPER)


class _Strip(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def _items(self, node):
        self.generic_visit(node)
        for item in node.items:
            if _is_helper_call(item.context_expr):
                old = item.context_expr
                new = ast.Call(func=ast.Attribute(value=ast.Name(id=_ALIAS, ctx=ast.Load()),
                                                  attr="nullcontext", ctx=ast.Load()),
                               args=[], keywords=[])
                for n in ast.walk(new):
                    ast.copy_location(n, old)
                item.context_expr = new
                self.count += 1
        return node

    visit_With = _items
    visit_AsyncWith = _items


def _fix_router(tree):
    """BaseRouter._select_experts counts out-of-range expert ids on every call
    with a ``.item()`` host sync: illegal while a CUDA graph is being captured
    (cudaErrorStreamCaptureUnsupported) and a per-MoE-layer sync outside it,
    neither in upstream vLLM. Only the (disabled) MoE-stats event reads the
    count; it becomes the constant 0."""
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_select_experts":
            for st in node.body:
                if isinstance(st, ast.Assign) and len(st.targets) == 1 \
                        and isinstance(st.targets[0], ast.Attribute) \
                        and st.targets[0].attr == "last_invalid_expert_id_count":
                    st.value = ast.copy_location(ast.Constant(value=0), st.value)
                    n += 1
    return n


_MODULE_FIXES = {"vllm.model_executor.layers.fused_moe.router.base_router": _fix_router}


def rewrite(source, path, fullname=None):
    """Return (code object, n rewritten) for a module source."""
    tree = ast.parse(source, filename=path)
    t = _Strip()
    tree = t.visit(tree)
    fix = _MODULE_FIXES.get(fullname)
    extra = fix(tree) if fix is not None else 0
    if extra:
        ast.fix_missing_locations(tree)
    if t.count:
        body = tree.body
        i = 0
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            i = 1  # keep the docstring first
        while i < len(body) and isinstance(body[i], ast.ImportFrom) and body[i].module == "__future__":
            i += 1
        imp = ast.Import(names=[ast.alias(name="contextlib", asname=_ALIAS)])
        ast.copy_location(imp, body[i] if i < len(body) else body[-1])
        body.insert(i, imp)
        ast.fix_missing_locations(tree)
    return compile(tree, path, "exec", dont_inherit=True), t.count + extra


class _Loader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        path = self.get_filename(fullname)
        data = self.get_data(path)
        text = data.decode("utf-8")
        if _HELPER + "(" not in text and fullname not in _MODULE_FIXES:
            return super().get_code(fullname)
        code, n = rewrite(text, path, fullname)
        if n:
            REWRITTEN.append((path, n))
            if os.environ.get("HWVAL_STRIP_NVTX_VERBOSE") == "1":
                print(f"[hwval] {fullname}: {n} NVTX label scope(s) -> nullcontext", file=sys.stderr)
        return code


def _observability_defaults(module):
    """The same build's GroupCoordinator._run_collective_with_timing reads
    ``observability_config.track_gpu_coll_op_timings``, a field its
    ObservabilityConfig never declares. At run time no vLLM config is
    current, so eager runs never reach the read; compile passes that trace a
    collective (the all-reduce + RMSNorm fusion pattern) run with the config
    current and die on AttributeError. Default it to False: timing off,
    which is what the eager runs did."""
    cls = getattr(module, "ObservabilityConfig", None)
    for field in ("track_gpu_coll_op_timings", "track_moe_stats"):
        if cls is not None and not hasattr(cls, field):
            setattr(cls, field, False)
            PATCHED.append(f"ObservabilityConfig.{field}=False")


_POST_IMPORT = {"vllm.config.observability": _observability_defaults}
PATCHED = []


class _PostLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        super().exec_module(module)
        fix = _POST_IMPORT.get(module.__name__)
        if fix is not None:
            fix(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in _POST_IMPORT:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and type(spec.loader) is importlib.machinery.SourceFileLoader:
                spec.loader = _PostLoader(fullname, spec.origin)
            return spec
        if not fullname.startswith(_PREFIXES):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or type(spec.loader) is not importlib.machinery.SourceFileLoader:
            return spec
        spec.loader = _Loader(fullname, spec.origin)
        return spec


def _chain_shadowed_sitecustomize():
    """This file shadows any other sitecustomize on sys.path (a base Python's,
    e.g. EasyBuild's EBPYTHONPREFIXES support); run that one too."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in sys.path:
        if not d or os.path.abspath(d) == here:
            continue
        cand = os.path.join(d, "sitecustomize.py")
        if os.path.isfile(cand):
            import importlib.util
            spec = importlib.util.spec_from_file_location("_hwval_shadowed_sitecustomize", cand)
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception as exc:  # never let a site hook break start-up
                print(f"[hwval] shadowed sitecustomize {cand} failed: {exc!r}", file=sys.stderr)
            return cand
    return None


if os.environ.get("HWVAL_STRIP_NVTX_LABELS", "1") != "0" and not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
if __name__ == "sitecustomize":
    _chain_shadowed_sitecustomize()
