import numpy as np
import pytest

from tinygrad import Tensor, Context
from tinygrad.codegen.winograd import WINOGRAD_TAG, winograd_pm
from tinygrad.uop.ops import Ops, UOp, graph_rewrite


def _base_inputs() -> tuple[np.ndarray, np.ndarray]:
  data = (np.arange(36, dtype=np.float32).reshape(1, 1, 6, 6) / 10).copy()
  weight = (np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) / 10).copy()
  return data, weight


@pytest.mark.parametrize("bias", [None, np.array([0.25], dtype=np.float32)])
def test_winograd_forward_matches_direct(bias):
  data_np, weight_np = _base_inputs()
  with Context(WINO=0):
    data = Tensor(data_np)
    weight = Tensor(weight_np)
    bias_tensor = None if bias is None else Tensor(bias)
    ref = data.conv2d(weight, padding=1, bias=bias_tensor).realize()
  with Context(WINO=1):
    data = Tensor(data_np)
    weight = Tensor(weight_np)
    bias_tensor = None if bias is None else Tensor(bias)
    out = data.conv2d(weight, padding=1, bias=bias_tensor).realize()
  np.testing.assert_allclose(ref.numpy(), out.numpy(), atol=1e-5, rtol=1e-5)


def test_winograd_backward_matches_direct():
  data_np, weight_np = _base_inputs()
  with Context(WINO=0):
    x0 = Tensor(data_np).requires_grad_(True)
    w0 = Tensor(weight_np).requires_grad_(True)
    loss0 = x0.conv2d(w0, padding=1).sum()
    loss0.backward()
    grad_x_ref, grad_w_ref = x0.grad.numpy().copy(), w0.grad.numpy().copy()
  with Context(WINO=1):
    x1 = Tensor(data_np).requires_grad_(True)
    w1 = Tensor(weight_np).requires_grad_(True)
    loss1 = x1.conv2d(w1, padding=1).sum()
    loss1.backward()
    grad_x_new, grad_w_new = x1.grad.numpy(), w1.grad.numpy()
  np.testing.assert_allclose(grad_x_ref, grad_x_new, atol=1e-5, rtol=1e-5)
  np.testing.assert_allclose(grad_w_ref, grad_w_new, atol=1e-5, rtol=1e-5)


def test_winograd_metadata_and_rewrite_trigger():
  data_np, weight_np = _base_inputs()
  with Context(WINO=1):
    conv = Tensor(data_np).conv2d(Tensor(weight_np), padding=1)
    reduce_nodes = [u for u in conv.uop.toposort() if u.op is Ops.REDUCE_AXIS]
    assert any(isinstance(u.tag, tuple) and u.tag[0] == WINOGRAD_TAG for u in reduce_nodes)
    sink = UOp.sink(conv.uop)
    rewritten = graph_rewrite(sink, winograd_pm)
    assert rewritten is not sink
