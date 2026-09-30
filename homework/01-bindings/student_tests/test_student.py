"""Student tests. Run against an installed wheel, never by adding src to sys.path."""
import numpy as np
import pytest

from tensor_ops import mac


# --- numerical ---------------------------------------------------------------

def test_index_formula_is_exact_on_prime_shape():
    # a = i + 1, b = j - 2, c = k / 4: every value and result is a small dyadic
    # number, so the expected answer from plain Python arithmetic is exact and
    # any mix-up of axes or flat index shows up as a wrong element.
    shape = (3, 5, 7)
    a, b, c = (np.empty(shape) for _ in range(3))
    expected = np.empty(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                a[i, j, k], b[i, j, k], c[i, j, k] = i + 1, j - 2, k / 4
                expected[i, j, k] = (i + 1) * (j - 2) + k / 4
    np.testing.assert_array_equal(mac(a, b, c), expected)


def test_algebraic_identities():
    rng = np.random.default_rng(417)
    shape = (2, 5, 3)
    a, b, c = (rng.uniform(-8.0, 8.0, size=shape) for _ in range(3))
    ones, zeros = np.ones(shape), np.zeros(shape)
    np.testing.assert_array_equal(mac(a, ones, zeros), a)   # a * 1 + 0
    np.testing.assert_array_equal(mac(a, zeros, c), c)      # a * 0 + c
    np.testing.assert_array_equal(mac(a, b, c), mac(b, a, c))  # a * b == b * a


# --- negative ----------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["int64", "uint64", "M8[ns]", "m8[s]", "complex64", "S8"])
def test_eight_byte_lookalike_dtypes_are_rejected(dtype):
    # All of these have itemsize 8, so a size-only check would let them through.
    ok = np.ones((2, 2, 2))
    impostor = np.zeros((2, 2, 2), dtype=dtype)
    assert impostor.itemsize == 8
    with pytest.raises(TypeError):
        mac(ok, impostor, ok)


def test_broadcast_view_with_matching_shape_is_rejected():
    ok = np.ones((2, 3, 4))
    stretched = np.broadcast_to(np.ones((1, 3, 4)), (2, 3, 4))  # stride 0 on axis 0
    assert stretched.shape == ok.shape and stretched.strides[0] == 0
    with pytest.raises(ValueError):
        mac(ok, ok, stretched)


def test_reversed_view_is_rejected():
    ok = np.ones((2, 3, 4))
    backwards = np.arange(24.0).reshape(2, 3, 4)[:, :, ::-1]  # negative stride
    with pytest.raises(ValueError):
        mac(backwards, ok, ok)


def test_wrong_argument_count_and_names():
    x = np.ones((1, 1, 1))
    with pytest.raises(TypeError):
        mac(x, x)
    with pytest.raises(TypeError):
        mac(x, x, x, x)
    with pytest.raises(TypeError):
        mac(x, x, d=x)


# --- memory / boundary ---------------------------------------------------------

def test_overlapping_windows_of_one_buffer():
    # a, b, c are C-contiguous slices of the same buffer that overlap each other.
    base = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3)
    snapshot = base.copy()
    a, b, c = base[0:2], base[1:3], base[2:4]
    assert np.shares_memory(a, b) and np.shares_memory(b, c)
    out = mac(a, b, c)
    np.testing.assert_array_equal(out, snapshot[0:2] * snapshot[1:3] + snapshot[2:4])
    np.testing.assert_array_equal(base, snapshot)
    assert not np.shares_memory(out, base)
    assert out.flags.owndata and out.flags.writeable


def test_readonly_file_backed_input(tmp_path):
    path = tmp_path / "a.bin"
    np.arange(12.0).tofile(path)
    a = np.memmap(path, dtype=np.float64, mode="r", shape=(2, 2, 3))
    assert not a.flags.writeable
    b = np.full((2, 2, 3), 2.0)
    c = np.full((2, 2, 3), -1.0)
    out = mac(a, b, c)
    np.testing.assert_array_equal(out, np.arange(12.0).reshape(2, 2, 3) * 2.0 - 1.0)
    assert type(out) is np.ndarray  # a plain new array, not a memmap onto the file
    np.testing.assert_array_equal(np.fromfile(path), np.arange(12.0))


def test_immutable_bytes_buffer_input():
    raw = np.array([1.5, -2.0, 4.0, 0.25]).tobytes()
    a = np.frombuffer(raw, dtype=np.float64).reshape(1, 2, 2)
    out = mac(a, a, a)  # x * x + x
    np.testing.assert_array_equal(out, [[[3.75, 2.0], [20.0, 0.3125]]])
    assert np.frombuffer(raw, dtype=np.float64).tolist() == [1.5, -2.0, 4.0, 0.25]


def test_array_that_is_both_c_and_f_contiguous_is_accepted():
    # With a single non-unit axis the Fortran copy is C-contiguous as well.
    a = np.asfortranarray(np.arange(5.0).reshape(1, 1, 5))
    assert a.flags.c_contiguous and a.flags.f_contiguous
    np.testing.assert_array_equal(mac(a, a, a), a * a + a)


def test_equal_but_not_identical_dtype_is_accepted():
    tagged = np.dtype("float64", metadata={"unit": "m"})
    assert tagged is not np.dtype("float64") and tagged == np.dtype("float64")
    a = np.full((1, 2, 1), 3.0, dtype=tagged)
    np.testing.assert_array_equal(mac(a, a, a), [[[12.0], [12.0]]])


def test_empty_inputs_are_still_validated():
    empty = np.empty((0, 3, 4))
    assert mac(empty, empty, empty).shape == (0, 3, 4)
    with pytest.raises(TypeError):
        mac(empty, np.empty((0, 3, 4), dtype=np.float32), empty)
    with pytest.raises(ValueError):
        mac(empty, empty, np.empty((0, 4, 3)))
    with pytest.raises(ValueError):
        mac(empty, empty, np.empty((3, 0, 4)))
