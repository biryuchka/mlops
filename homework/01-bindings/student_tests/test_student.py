"""Student tests. Run against an installed wheel, never by adding src to sys.path."""
import numpy as np
import pytest

from tensor_ops import mac


def test_hand_calculated_unusual_shape():
    # Shape (2, 1, 3): values chosen so a * b + c is easy to verify by hand.
    a = np.array([[[1.0, 2.0, 3.0]], [[-1.0, 0.5, 10.0]]])
    b = np.array([[[2.0, 2.0, 2.0]], [[4.0, -4.0, 0.1]]])
    c = np.array([[[0.0, 1.0, -6.0]], [[4.0, 2.0, -1.0]]])
    expected = np.array([[[2.0, 5.0, 0.0]], [[0.0, 0.0, 0.0]]])
    out = mac(a, b, c)
    assert out.shape == (2, 1, 3)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_fortran_layout_is_rejected(position):
    args = [np.ones((2, 3, 4)) for _ in range(3)]
    args[position] = np.asfortranarray(args[position])
    assert not args[position].flags.c_contiguous
    with pytest.raises(ValueError):
        mac(*args)


@pytest.mark.parametrize("position", [0, 1, 2])
def test_transposed_view_is_rejected(position):
    args = [np.ones((2, 3, 4)) for _ in range(3)]
    # Same shape (2, 3, 4) as the others, but a non-C-contiguous view.
    args[position] = np.ones((4, 3, 2)).transpose(2, 1, 0)
    assert args[position].shape == (2, 3, 4)
    with pytest.raises(ValueError):
        mac(*args)


def test_same_object_for_all_inputs():
    x = np.array([[[1.0, 2.0], [3.0, -4.0]]])
    snapshot = x.copy()
    out = mac(x, x, x)
    # x * x + x computed by hand: 1+1, 4+2, 9+3, 16-4.
    np.testing.assert_allclose(out, [[[2.0, 6.0], [12.0, 12.0]]], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(x, snapshot)
    assert not np.shares_memory(out, x)


def test_empty_input_still_validates_dtype():
    a = np.empty((0, 3, 4))
    b = np.empty((0, 3, 4), dtype=np.float32)
    with pytest.raises(TypeError):
        mac(a, b, a)


def test_empty_input_still_validates_shape():
    a = np.empty((0, 3, 4))
    with pytest.raises(ValueError):
        mac(a, a, np.empty((0, 4, 3)))


def test_zero_dimensional_and_4d_are_rejected():
    ok = np.ones((1, 1, 1))
    with pytest.raises(ValueError):
        mac(np.ones(()), ok, ok)
    with pytest.raises(ValueError):
        mac(np.ones((1, 1, 1, 1)), ok, ok)
