from __future__ import annotations
from dataclasses import dataclass
import itertools
from functools import lru_cache

from tinygrad.helpers import WINO, prod
from tinygrad.uop.ops import Ops, PatternMatcher, UOp, UPat
from tinygrad.dtype import DType

WINOGRAD_TAG = "WINOGRAD_META"

@dataclass(frozen=True)
class WinogradMeta:
  input_uop: UOp
  weight_uop: UOp
  bias_uop: UOp|None
  groups: int
  stride: tuple[int, ...]
  dilation: tuple[int, ...]
  padding: tuple[int, ...]
  dtype: DType|None
  output_shape: tuple[int, ...]
  input_shape: tuple[int, ...]
  weight_shape: tuple[int, ...]


@lru_cache(maxsize=None)
def _winograd_constants(_:int) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
  winograd_G = ((1/4, 0, 0), (-1/6, -1/6, -1/6), (-1/6, 1/6, -1/6), (1/24, 1/12, 1/6), (1/24, -1/12, 1/6), (0, 0, 1))
  winograd_Bt = ((4, 0, -5, 0, 1, 0), (0, -4, -4, 1, 1, 0), (0, 4, -4, -1, 1, 0), (0, -2, -1, 2, 1, 0), (0, 2, -1, -2, 1, 0), (0, 4, 0, -5, 0, 1))
  winograd_At = ((1, 1, 1, 1, 1, 0), (0, 1, -1, 2, -2, 0), (0, 1, 1, 4, 4, 0), (0, 1, -1, 8, -8, 1))
  return winograd_G, winograd_Bt, winograd_At


def _get_winograd_matcols(mat, dims:int, shp:tuple[int, ...], device, dtype) -> list[list]:
  from tinygrad import Tensor
  return [[Tensor.cat(*[Tensor.full(shp[:dim] + (1,) + shp[dim+1:], float(m[k]), device=device, dtype=dtype) for m in mat], dim=dim)
           for k in range(len(mat[0]))] for dim in range(dims)]


def _apply_winograd_matrix(mat, t, dims:int):
  from tinygrad import Tensor
  t_ = t.reshape(t.shape[:dims] + (1,) * dims + t.shape[dims:]).expand(t.shape[:dims] + (len(mat),) * dims + t.shape[dims:])
  matcols = _get_winograd_matcols(mat, dims, t_.shape[dims:], t_.device, t_.dtype)
  ret = sum(prod(col[idx] for col, idx in zip(matcols, mat_is)) * t_[mat_is] for mat_is in itertools.product(range(len(mat[0])), repeat=dims))
  assert isinstance(ret, Tensor)
  return ret


def _winograd_padding(input_shape:tuple[int, ...], padding:tuple[int, ...], tile:int) -> tuple[int, ...]:
  pad = []
  spatial_dims = input_shape[-len(padding)//2:]
  for i, dim in enumerate(spatial_dims):
    left, right = padding[i*2], padding[i*2+1]
    extra = (-(dim + left + right - 2) % tile)
    pad.extend((left, right + extra))
  return tuple(pad)


def _winograd_conv(meta:WinogradMeta):
  from tinygrad import Tensor
  if WINO.value == 0: return None
  dims = len(meta.weight_shape) - 2
  if dims != 2: return None
  if any(k != 3 for k in meta.weight_shape[2:]): return None
  if any(s != 1 for s in meta.stride) or any(d != 1 for d in meta.dilation): return None
  if meta.groups != 1: return None

  inp = Tensor(meta.input_uop)
  weight = Tensor(meta.weight_uop)
  bias = Tensor(meta.bias_uop) if meta.bias_uop is not None else None
  dtype = meta.dtype

  HWI = (6,) * dims
  HWO = (4,) * dims
  winograd_G, winograd_Bt, winograd_At = _winograd_constants(dims)

  pad = _winograd_padding(meta.input_shape, meta.padding, HWO[0])
  d = inp.pad(pad)._pool(HWI, HWO)
  d = d.permute(*range(len(d.shape)-dims, len(d.shape)), *range(len(d.shape)-dims))
  tyx = d.shape[-dims:]

  g = weight.permute(*range(len(weight.shape)-dims, len(weight.shape)), *range(len(weight.shape)-dims))
  bs, cout = inp.shape[0], weight.shape[0]
  cin = weight.shape[1]
  groups = meta.groups
  rcout = cout // groups

  gfactors = _apply_winograd_matrix(winograd_G, g, dims).reshape(*HWI, 1, groups, rcout, cin, *([1]*len(tyx)))
  dfactors = _apply_winograd_matrix(winograd_Bt, d, dims).reshape(*HWI, bs, groups, 1, cin, *tyx)
  ret = _apply_winograd_matrix(winograd_At, (gfactors * dfactors).sum(axis=-1-dims, dtype=dtype), dims)

  base = len(ret.shape) - dims
  perm = [*range(dims, base), *[i+o for i in range(dims) for o in [base, 0]]]
  ret = ret.permute(perm)
  ret = ret.reshape(bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]).shrink(tuple((0, s) for s in meta.output_shape))
  ret = ret if bias is None else ret.add(bias.reshape(1, -1, *[1 for _ in range(dims)]))
  return ret.contiguous().contiguous_backward()


def _winograd_replacer(reduce:UOp, mul:UOp, xpatch:UOp, wview:UOp):
  tag = reduce.tag
  if not (isinstance(tag, tuple) and len(tag) == 2 and tag[0] == WINOGRAD_TAG and isinstance(tag[1], WinogradMeta)):
    return None
  meta = tag[1]
  ret = _winograd_conv(meta)
  if ret is None: return None
  return ret.uop


winograd_pm = PatternMatcher([
  (UPat(Ops.REDUCE_AXIS, src=(UPat(Ops.MUL, src=(UPat.var("xpatch"), UPat.var("wview")), name="mul"),), name="reduce"), _winograd_replacer),
])

__all__ = ["WinogradMeta", "WINOGRAD_TAG", "winograd_pm"]
