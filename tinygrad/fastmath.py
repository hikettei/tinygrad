
import math
from typing import Tuple, List
from tinygrad.dtype import dtypes, DType
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.lazy import LazyBuffer

def dtype_of(d: LazyBuffer) -> DType:
  from test.helpers import is_dtype_supported
  if is_dtype_supported(dtypes.bfloat16, d.device):
    return dtypes.bfloat16 if d.dtype == dtypes.bfloat16 else d.dtype
  return d.dtype

def is_dtype_fastmath_supported(d: DType):
  return d in [dtypes.float16, dtypes.float32, dtypes.float64]

def _lazy_map_numbers(x: LazyBuffer, inf: LazyBuffer, _inf: LazyBuffer, nan: LazyBuffer, ratio: LazyBuffer):
  """replace inf -> inf, -inf -> _inf, nan -> nan, otherwise -> ratio"""
  return x.e(BinaryOps.CMPNE, x.const(math.inf)).e(TernaryOps.WHERE, x.e(BinaryOps.CMPNE, x).e(TernaryOps.WHERE, nan, x.e(BinaryOps.CMPNE, x.const(-math.inf)).e(TernaryOps.WHERE, ratio, _inf)), inf) # noqa: E501

# *** helper functions for double/quad precision arithmetics ***
def dfadd2_f2_f2_f2(xx: LazyBuffer, xy: LazyBuffer, yx: LazyBuffer, yy: LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  rx = xx.e(BinaryOps.ADD, yx)
  v = rx.e(BinaryOps.ADD, xx.e(UnaryOps.NEG))
  ry = xx.e(BinaryOps.ADD, rx.e(BinaryOps.ADD, v.e(UnaryOps.NEG)).e(UnaryOps.NEG)).e(BinaryOps.ADD, yx.e(BinaryOps.ADD, v.e(UnaryOps.NEG)))
  ry = xy.e(BinaryOps.ADD, yy)
  return rx, ry

def dfmul2_f2_f2_f2(xx: LazyBuffer, xy: LazyBuffer, yx: LazyBuffer, yy: LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  rx = xx.e(BinaryOps.MUL, yx)
  ry = xx.e(BinaryOps.MUL, yx).e(BinaryOps.ADD, rx.e(UnaryOps.NEG)).e(BinaryOps.ADD, xx.e(BinaryOps.MUL, yy)).e(BinaryOps.ADD, xy.e(BinaryOps.MUL, yx)) # noqa: E501
  return rx, ry

def dfdiv2_f2_f2_f2(nx: LazyBuffer, ny: LazyBuffer, dx: LazyBuffer, dy: LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  t = dx.e(UnaryOps.RECIP)
  qx = nx.e(BinaryOps.MUL, t)
  u = qx.e(UnaryOps.NEG).e(BinaryOps.ADD, nx.e(BinaryOps.MUL, t)).e(BinaryOps.ADD, qx.e(BinaryOps.MUL, dx.const(1).e(BinaryOps.ADD, dx.e(BinaryOps.MUL, t).e(UnaryOps.NEG)))) # noqa: E501
  qy = t.e(BinaryOps.MUL, ny.e(BinaryOps.ADD, qx.e(BinaryOps.MUL, dy).e(UnaryOps.NEG))).e(BinaryOps.ADD, u)
  return qx, qy

# *** helper functions for bit manipulation ***
def significand_bits(d: DType) -> int:
  assert is_dtype_fastmath_supported(d)
  return {dtypes.float64: 52, dtypes.float32: 23, dtypes.float16: 10, dtypes.bfloat16: 7}[d]

def exponent_bias(d: DType) -> int:
  return {dtypes.float64: 1022, dtypes.float32: 126, dtypes.float16: 14, dtypes.bfloat16: 126}[d]

def exponent_mask(d: DType) -> int:
  assert is_dtype_fastmath_supported(d)
  return {dtypes.float64: 0x7FF, dtypes.float32: 0xFF, dtypes.float16: 0x1F, dtypes.bfloat16: 0x7f80}[d]

def float_to_bits(d: LazyBuffer) -> LazyBuffer:
  cast_to = {dtypes.float64: dtypes.uint64, dtypes.float32: dtypes.uint32, dtypes.float16: dtypes.uint16}[d.dtype]
  return d.cast(cast_to, True, True)

def bits_to_float(d: LazyBuffer, float_dtype: DType) -> LazyBuffer:
  cast_to = {dtypes.uint64: dtypes.float64, dtypes.uint32: dtypes.float32, dtypes.uint16: float_dtype}[d.dtype]
  return d.cast(cast_to, True, True)

# **** utils ****
def rintk(d: LazyBuffer) -> LazyBuffer:
  assert is_dtype_fastmath_supported(d.dtype)
  return_t = {dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}[d.dtype]
  return d.e(BinaryOps.ADD, d.e(BinaryOps.CMPLT, d.const(0.0)).e(TernaryOps.WHERE, d.const(-0.5), d.const(0.5))).cast(return_t)

def mla(x: LazyBuffer, y: LazyBuffer, z: LazyBuffer) -> LazyBuffer:
  return x.e(BinaryOps.MUL, y).e(BinaryOps.ADD, z)

def polyN(u: LazyBuffer, s: LazyBuffer, coeffs: List[float]) -> LazyBuffer:
  for c in coeffs:
    u = mla(u, s, u.const(c))
  return u

def ilogb2k(d:LazyBuffer) -> LazyBuffer:
  assert is_dtype_fastmath_supported(d.dtype)
  dint = d.cast({dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}[d.dtype], True, True)
  # ((float_to_bits(d) >> significand_bits(dtype)) & exponent_mask(dtype)) - exponent_bias(dtype)
  return dint.e(BinaryOps.SHR, dint.const(significand_bits(dtype_of(d)))).e(BinaryOps.AND, dint.const(exponent_mask(dtype_of(d)))).e(BinaryOps.ADD, dint.const(-(exponent_bias(dtype_of(d))+1))) # noqa: E501

def ldexp3k(d:LazyBuffer, e:LazyBuffer) -> LazyBuffer:
  assert is_dtype_fastmath_supported(d.dtype) and is_dtype_fastmath_supported(e.dtype)
  dtype = d.dtype
  d = d.cast(dtypes.float64) if d.device != "METAL" else d
  cast_map = {dtypes.float64: dtypes.int64, dtypes.float32: dtypes.int32, dtypes.float16: dtypes.int16}
  e = e.cast(cast_map[d.dtype])
  m1 = d.cast(cast_map[d.dtype], True, True)
  m2 = e.e(BinaryOps.SHL, e.const(significand_bits(dtype_of(d))))
  return m1.e(BinaryOps.ADD, m2).cast(d.dtype, True, True).cast(dtype)

def pow2if(q: LazyBuffer, float_dtype: DType):
  final_dtype = {dtypes.int64: dtypes.float64, dtypes.int32: dtypes.float32, dtypes.int16: float_dtype}[q.dtype]
  return q.e(BinaryOps.ADD, q.const(exponent_bias(final_dtype)+1)).e(BinaryOps.SHL, q.const(significand_bits(final_dtype))).cast(dtypes.float16 if final_dtype == dtypes.bfloat16 else final_dtype, True, True) # noqa: E501

def ldexp2kf(d: LazyBuffer, e: LazyBuffer) -> LazyBuffer:
  assert is_dtype_fastmath_supported(d.dtype) and e.dtype in (dtypes.int16, dtypes.int32, dtypes.int64)
  return d.e(BinaryOps.MUL, pow2if(e.e(BinaryOps.SHR, e.const(1)), dtype_of(d))).e(BinaryOps.MUL, pow2if(e.e(BinaryOps.ADD, e.e(BinaryOps.SHR, e.const(1)).e(UnaryOps.NEG)), dtype_of(d))) # noqa: E501

def frexp(v: LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  m1 = {dtypes.float64: 0x800FFFFF, dtypes.float32: 0x807FFFFF, dtypes.float16: 0x83FF, dtypes.bfloat16: 0x80FF}[dtype_of(v)] # noqa: E501
  m2 = {dtypes.float64: 0x3FE0000000000000, dtypes.float32: 0x3F000000, dtypes.float16: 0x3C00, dtypes.bfloat16: 0x3F00}[dtype_of(v)] # noqa: E501
  bits = float_to_bits(v)
  exponent = bits.e(BinaryOps.SHR, bits.const(significand_bits(dtype_of(v)))).e(BinaryOps.AND, bits.const(exponent_mask(dtype_of(v))))
  exponent_zero = exponent.e(BinaryOps.CMPNE, exponent.const(0.0))
  result_f = bits_to_float(bits.e(BinaryOps.AND, bits.const(m1)).e(BinaryOps.OR, bits.const(m2)), v.dtype)
  value = exponent_zero.e(TernaryOps.WHERE, result_f, v)
  exp = exponent.e(BinaryOps.ADD, exponent.const(-exponent_bias(dtype_of(v))))
  exp = exponent_zero.e(TernaryOps.WHERE, exp, exp.const(0))
  return value, exp

# *** helper algorithm for sin ***
# d = abs(d_base)
def payne_hanek_reduction(d: LazyBuffer, d_base: LazyBuffer) -> LazyBuffer:
  assert is_dtype_fastmath_supported(d.dtype)
  two_over_pi_f = [0x00000000,0x28be60db,0x9391054a,0x7f09d5f4,0x7d4d3770,0x36d8a566,0x4f10e410] # noqa: E501

  input_dtype: DType = d.dtype
  dtype_via = dtypes.float32 if d.dtype == dtypes.float16 else d.dtype

  f, e = frexp(d)
  ia = (k := f.cast(dtype_via)).e(BinaryOps.MUL, k.const(4.294967296e9)).cast(dtypes.uint64)
  i = (k := e.cast(dtypes.uint64)).e(BinaryOps.SHR, k.const(5))
  e = (k := e.cast(dtypes.uint64)).e(BinaryOps.AND, k.const(31))

  def _eq(arr: LazyBuffer, eq_to: int) -> LazyBuffer: return arr.e(BinaryOps.CMPNE, arr.const(eq_to))

  # a_n = (two_over_pi_f[Int(i) + n] << e) | (two_over_pi_f[Int(i) + n+1] >> (nbits - e))
  a1 = i.const(0).cast(dtypes.uint32)
  a2 = i.const(0).cast(dtypes.uint32)
  a3 = i.const(0).cast(dtypes.uint32)
  for n in range(len(two_over_pi_f[:-2])):
    a1 = _eq(i, n).e(TernaryOps.WHERE, a1, a1.const(two_over_pi_f[n+0]))
    a2 = _eq(i, n).e(TernaryOps.WHERE, a2, a2.const(two_over_pi_f[n+1]))
    a3 = _eq(i, n).e(TernaryOps.WHERE, a3, a3.const(two_over_pi_f[n+2]))
  a1p1 = a1.const(0)
  a2p1 = a2.const(0)
  a3p1 = a3.const(0)
  for n in range(len(two_over_pi_f[0:-3])):
    a1p1 = _eq(i, n).e(TernaryOps.WHERE, a1p1, a1p1.const(two_over_pi_f[n+1]))
    a2p1 = _eq(i, n).e(TernaryOps.WHERE, a2p1, a2p1.const(two_over_pi_f[n+2]))
    a3p1 = _eq(i, n).e(TernaryOps.WHERE, a3p1, a3p1.const(two_over_pi_f[n+3]))

  e = e.cast(dtypes.uint32)
  offset = e.const(32).e(BinaryOps.ADD, e.e(UnaryOps.NEG))

  hi = _eq(e, 0).e(TernaryOps.WHERE, a1.e(BinaryOps.SHL, e).e(BinaryOps.OR, a1p1.e(BinaryOps.SHR, offset)), a1)
  mi = _eq(e, 0).e(TernaryOps.WHERE, a2.e(BinaryOps.SHL, e).e(BinaryOps.OR, a2p1.e(BinaryOps.SHR, offset)), a2)
  lo = _eq(e, 0).e(TernaryOps.WHERE, a3.e(BinaryOps.SHL, e).e(BinaryOps.OR, a3p1.e(BinaryOps.SHR, offset)), a3)

  def _hp_mul(x: LazyBuffer, y: LazyBuffer) -> LazyBuffer: return x.cast(dtypes.uint64).e(BinaryOps.MUL, y.cast(dtypes.uint64))
  p = _hp_mul(ia, lo)
  p = _hp_mul(ia, mi).e(BinaryOps.ADD, p.e(BinaryOps.SHR, p.const(32)))
  p = _hp_mul(ia, hi).e(BinaryOps.SHL, p.const(32)).e(BinaryOps.ADD, p)

  q = p.e(BinaryOps.SHR, p.const(62)).cast(dtypes.int32)
  p = p.e(BinaryOps.AND, p.const(0x3fffffffffffffff))

  fr_map = p.e(BinaryOps.AND, p.const(0x2000000000000000)).e(BinaryOps.CMPNE, p.const(0))

  p = fr_map.e(TernaryOps.WHERE, p.e(BinaryOps.ADD, p.const(-0x4000000000000000)), p)
  q = fr_map.e(TernaryOps.WHERE, q.e(BinaryOps.ADD, q.const(1)), q)

  d = p.cast(dtype_via)
  d = d.e(BinaryOps.MUL, d.const(3.4061215800865545e-19))
  r = d.cast(input_dtype)

  lt_zero_map = d_base.e(BinaryOps.CMPLT, d_base.const(0.0))

  q = q.e(BinaryOps.MOD, q.const(4))
  q_mod_2 = q.e(BinaryOps.MOD, q.const(2))
  rotate_map = q_mod_2.e(BinaryOps.CMPNE, q_mod_2.const(1))
  rotations = rotate_map.e(TernaryOps.WHERE, r.const(0), r.const(math.pi / 2))
  r = r.e(BinaryOps.ADD, rotations)

  add_map = q.e(BinaryOps.CMPLT, q.const(2))

  r = add_map.e(TernaryOps.WHERE, r, r.e(UnaryOps.NEG))
  r = lt_zero_map.e(TernaryOps.WHERE, r.e(UnaryOps.NEG), r)
  return r.cast(input_dtype)

# *** base implementation for xsin/xlog2/xexp2 ***
# Set fast=True to skip Payne-Hanek reduction.
def _xsin_base(d: LazyBuffer, fast:bool=False) -> LazyBuffer:
  assert is_dtype_fastmath_supported(d.dtype)
  d = _lazy_map_numbers(d, d.const(0.0), d.const(0.0), d.const(0.0), d)
  fp64_p = d.dtype == dtypes.float64
  trig_range_lv1 = d.const(15.0 if fp64_p else 125.0)
  trig_range_lv2 = d.const(1e+14 if fp64_p else 39000)
  m_1_pi = 0.318309886183790671537767526745028724

  # di = abs(d)
  di = d.e(BinaryOps.MUL, d.e(BinaryOps.CMPNE, d.const(0)).e(TernaryOps.WHERE, d.e(BinaryOps.CMPLT, d.const(0)).e(TernaryOps.WHERE, d.const(-1), d.const(1)), d.const(0))) # noqa: E501

  qdh = None
  if fp64_p:
    qdh = d.e(BinaryOps.MUL, d.const(m_1_pi / 16777216)).cast(dtypes.int64).cast(d.dtype).e(BinaryOps.MUL, d.const(16777216.0))

  def __lv1q(x: LazyBuffer) -> LazyBuffer:
    return rintk(x.e(BinaryOps.MUL, d.const(m_1_pi))).cast(d.dtype)

  def __lv2q(x: LazyBuffer) -> LazyBuffer:
    if fp64_p:
      assert qdh is not None
      return rintk(mla(d, d.const(m_1_pi), qdh.e(UnaryOps.NEG))).cast(d.dtype)
    return __lv1q(x)

  lv3_reduced_d = payne_hanek_reduction(di, d)
  lv3_q = __lv2q(d) if fast else __lv2q(lv3_reduced_d) # skip the payne_hanek_reduction
  q: LazyBuffer = di.e(BinaryOps.CMPLT, trig_range_lv1).e(TernaryOps.WHERE, __lv1q(d), di.e(BinaryOps.CMPLT, trig_range_lv2).e(TernaryOps.WHERE, __lv2q(d), lv3_q)) # noqa: E501
  def __lv1(x: LazyBuffer) -> LazyBuffer:
    if fp64_p:
      d = mla(q, x.const(-3.141592653589793116), x)
      d = mla(q, x.const(1.2246467991473532072e-16), d)
    else:
      d = mla(q, x.const(-3.1414794921875), x)
      d = mla(q, x.const(-0.00011315941810607910156), d)
      d = mla(q, x.const(-1.9841872589410058936e-09), d)
    return d

  def __lv2(x: LazyBuffer) -> LazyBuffer:
    if fp64_p:
      assert qdh is not None
      d = mla(qdh, x.const(-3.1415926218032836914), x)
      d = mla(q, x.const(-3.1415926218032836914), d)
      d = mla(qdh, x.const(-3.1786509424591713469e-08), d)
      d = mla(q, x.const(-3.1786509424591713469e-08), d)
      d = mla(qdh, x.const(-1.2246467864107188502e-16), d)
      d = mla(q, x.const(-1.2246467864107188502e-16), d)
      d = mla(qdh.e(BinaryOps.ADD, q), x.const(-1.2736634327021899816e-24), d)
    else:
      d = mla(q, x.const(-3.1414794921875), x)
      d = mla(q, x.const(-0.00011315941810607910156), d)
      d = mla(q, x.const(-1.9841872589410058936e-09), d)
      d = mla(q, x.const(-1.2154201256553420762e-10), d)
    return d

  lv3_d = __lv2(d) if fast else __lv2(lv3_reduced_d)
  d = di.e(BinaryOps.CMPLT, trig_range_lv1).e(TernaryOps.WHERE, __lv1(d), di.e(BinaryOps.CMPLT, trig_range_lv2).e(TernaryOps.WHERE, __lv2(d), lv3_d))
  s = d.e(BinaryOps.MUL, d)
  a = q.cast(dtypes.int64).e(BinaryOps.MOD, d.const(2).cast(dtypes.int64)).cast(d.dtype)
  d = d.e(BinaryOps.MUL, a.e(BinaryOps.CMPNE, d.const(0)).e(TernaryOps.WHERE, d.const(-1), d.const(1)))

  u = None
  if fp64_p:
    s2 = s.e(BinaryOps.MUL, s)
    s4 = s2.e(BinaryOps.MUL, s2)
    def __poly4(x: LazyBuffer, x2: LazyBuffer, c3, c2, c1, c0) -> LazyBuffer: return mla(x2, mla(x, d.const(c3), d.const(c2)), mla(x, d.const(c1), d.const(c0))) # noqa: E501
    def __poly8(x, x2, x4, c7, c6, c5, c4, c3, c2, c1, c0) -> LazyBuffer: return mla(x4, __poly4(x, x2, c7, c6, c5, c4), __poly4(x, x2, c3, c2, c1, c0)) # noqa: E501
    u = __poly8(s, s2, s4, -7.97255955009037868891952e-18, 2.81009972710863200091251e-15, -7.64712219118158833288484e-13, 1.60590430605664501629054e-10, -2.50521083763502045810755e-08, 2.75573192239198747630416e-06, -0.000198412698412696162806809, 0.00833333333333332974823815) # noqa: E501
    u = mla(u, s, d.const(-0.166666666666666657414808))
    u = mla(s, u.e(BinaryOps.MUL, d), d)
  else:
    u = polyN(s.const(2.6083159809786593541503e-06), s, [-0.0001981069071916863322258, 0.00833307858556509017944336, -0.166666597127914428710938]) # noqa: E501
    u = mla(s, u.e(BinaryOps.MUL, d), d)
  return u

def _xexp2_base(d: LazyBuffer) -> LazyBuffer:
  fp64_p = d.dtype == dtypes.float64
  q = rintk(d)
  s = d.e(BinaryOps.ADD, q.cast(d.dtype).e(UnaryOps.NEG))
  # a polynomial approximation with 13 non-zero terms in the range of [−(log 2)/2,(log 2)/2].
  if fp64_p:
    u = polyN(s.const(0.4434359082926529454e-9), s, [0.7073164598085707425e-8, 0.1017819260921760451e-6, 0.1321543872511327615e-5, 0.1525273353517584730e-4, 0.1540353045101147808e-3, 0.1333355814670499073e-2, 0.9618129107597600536e-2, 0.5550410866482046596e-1, 0.2402265069591012214e+0, 0.6931471805599452862e+0, 0.1000000000000000000e+1]) # noqa: E501
  else:
    u = polyN(s.const(0.1535920892e-3), s, [0.1339262701e-2, 0.9618384764e-2, 0.5550347269e-1, 0.2402264476e+0, 0.6931471825e+0, 0.1000000000e+1])
  u = ldexp2kf(u, q)
  upper = {dtypes.float64: 1024, dtypes.float32: 128, dtypes.float16: 23.0}[d.dtype]
  lower = {dtypes.float64: -2000, dtypes.float32: -150, dtypes.float16: -22}[d.dtype]
  u = d.e(BinaryOps.CMPNE, d.const(upper)).e(TernaryOps.WHERE, u, d.const(math.inf))
  u = d.e(BinaryOps.CMPLT, d.const(upper)).e(TernaryOps.WHERE, u, d.const(math.inf))
  u = d.e(BinaryOps.CMPLT, d.const(lower)).e(TernaryOps.WHERE, d.const(0.0), u)
  return u

# when denormal=True, dedicated to x < FLT_MIN, when False, dedicated to x >= FLT_MIN
def _xlog2_base(d: LazyBuffer, denormal: bool) -> LazyBuffer:
  if 0 in d.shape: return d
  fp64_p = d.dtype == dtypes.float64

  # d *= 2**32 * 2**32
  for _ in range(2):
    d = d.e(BinaryOps.MUL, d.const(2 ** 32)) if denormal else d

  e = ilogb2k(d.e(BinaryOps.MUL, d.const(1.0 / 0.75))).cast(d.dtype)
  m = ldexp3k(d, e.e(UnaryOps.NEG))
  e = e.e(BinaryOps.ADD, e.const(-64)) if denormal else e

  if fp64_p:
    x = m.e(BinaryOps.ADD, m.const(-1.0)).e(BinaryOps.MUL, m.e(BinaryOps.ADD, m.const(1.0)).e(UnaryOps.RECIP))
    x2 = x.e(BinaryOps.MUL, x)
    t = polyN(x.const(0.2211941750456081490e+0), x2, [0.2200768693152277689e+0, 0.2623708057488514656e+0, 0.3205977477944495502e+0, 0.4121985945485324709e+0, 0.5770780162997058982e+0, 0.96179669392608091449]) # noqa: E501
    s_hi, s_lo = dfadd2_f2_f2_f2(e, e.const(0), *dfmul2_f2_f2_f2(t.const(2.885390081777926774), t.const(0), x, x.const(0)))
    r = mla(t, x.e(BinaryOps.MUL, x2), s_hi.e(BinaryOps.ADD, s_lo))
  else:
    xx, xy = dfdiv2_f2_f2_f2(*dfadd2_f2_f2_f2(m.const(-1), m.const(0), m, m.const(0)), *dfadd2_f2_f2_f2(m.const(1), m.const(0), m, m.const(0)))
    x2 = xx.e(BinaryOps.MUL, xx)
    t = polyN(d.const(0.4374550283e+0), x2, [0.5764790177e+0, 0.9618012905120])
    sx, sy = dfadd2_f2_f2_f2(e, e.const(0), *dfmul2_f2_f2_f2(xx, xy, xx.const(2.8853900432586669922), xy.const(3.2734474483568488616e-08)))
    sx, sy = dfadd2_f2_f2_f2(sx, sy, x2.const(0), x2.e(BinaryOps.MUL, xx).e(BinaryOps.MUL, t))
    r = sx.e(BinaryOps.ADD, sy)

  isinf_map = d.e(BinaryOps.CMPNE, d.const(math.inf))
  nan_map1 = d.e(BinaryOps.CMPLT, d.const(0.0))
  nan_map2 = d.e(BinaryOps.CMPNE, d)
  zero_map = d.e(BinaryOps.CMPNE, d.const(0.0))

  r = isinf_map.e(TernaryOps.WHERE, r, r.const(math.inf))
  r = nan_map1.e(TernaryOps.WHERE, r.const(math.nan), r)
  r = nan_map2.e(TernaryOps.WHERE, r.const(math.nan), r)
  r = zero_map.e(TernaryOps.WHERE, r, r.const(-math.inf))
  return r

# ****** toplevels for fastmath *****
def xsin(x: LazyBuffer, fast: bool=False) -> LazyBuffer:
  assert is_dtype_fastmath_supported(x.dtype)
  if 0 in x.shape: return x
  return _lazy_map_numbers(x, x.const(math.nan), x.const(math.nan), x.const(math.nan), _xsin_base(x, fast=fast))

def xlog2(d: LazyBuffer) -> LazyBuffer:
  assert is_dtype_fastmath_supported(d.dtype)
  FLT_MIN = d.const(1e-6 if d.dtype == dtypes.float16 else 1e-4)
  out = d.e(BinaryOps.CMPLT, FLT_MIN).e(TernaryOps.WHERE, _xlog2_base(d, True), _xlog2_base(d, False))
  return d.e(BinaryOps.CMPNE, d.const(0.0)).e(TernaryOps.WHERE, out, d.const(-math.inf))

def xexp2(d: LazyBuffer) -> LazyBuffer:
  assert is_dtype_fastmath_supported(d.dtype)
  if 0 in d.shape: return d
  x = _lazy_map_numbers(d, d.const(0.0), d.const(0.0), d.const(0.0), d)
  d = _lazy_map_numbers(d, d.const(math.inf), d.const(0.0), d.const(math.nan), _xexp2_base(x))
  return d

