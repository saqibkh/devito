import numpy as np
import pytest

from devito import Eq, Function, Grid, Operator, SubGrid, switchconfig
from devito.ir.iet import Expression, FindNodes, HaloSpot, Iteration
from devito.symbolics import retrieve_indexed

COEFFS = (3., 2., 0.5)


def linear(shape):
    """`3*i + 2*j + 0.5*k` over `shape`, in index units."""
    idx = np.meshgrid(*[np.arange(n) for n in shape], indexing='ij')
    return sum(c*i for c, i in zip(COEFFS[:len(shape)], idx, strict=True))


def restricted(factor, n=25, ndim=2):
    """Read a linear SubGrid field at its parent's points."""
    grid = Grid(shape=(n,)*ndim, extent=(float(n - 1),)*ndim)
    sub = SubGrid(grid, factor=factor)

    u = Function(name='u', grid=grid, space_order=2)
    vp = Function(name='vp', grid=sub, space_order=2)
    vp.data[:] = linear(sub.shape).astype(vp.dtype)
    u.data[:] = 0.

    Operator(Eq(u, vp)).apply()

    return sub, u, vp


def expected_at_parent(sub, idx):
    """
    `linear` evaluated at the parent points `idx`, in SubGrid index units.

    A parent point `x` sits at SubGrid coordinate `x/factor`, so this is what
    an exact read of a `linear`-valued SubGrid Function must produce.
    """
    return sum(c*(i/float(f)) for c, i, f in
               zip(COEFFS[:len(idx)], idx, sub.factor, strict=True))


def interior(sub, n, ndim):
    """Mask of parent points whose stencil stays inside the SubGrid."""
    idx = np.meshgrid(*[np.arange(n)]*ndim, indexing='ij')
    mask = np.ones_like(idx[0], dtype=bool)
    for i, (ns, f) in enumerate(zip(sub.shape, sub.factor, strict=True)):
        mask &= idx[i] <= int((ns - 1)*float(f)) - 1
    return mask, idx


class TestApi:

    """
    A SubGrid Function is read at the LHS's location, with no wrapper -- the
    same convention a staggered Function follows.
    """

    def test_plain_equation(self):
        grid = Grid(shape=(13, 13), extent=(12., 12.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        u.data[:] = 1.
        vp.data[:] = 2.5

        Operator(Eq(u, vp*u)).apply()

        mask, _ = interior(sub, 13, 2)
        assert np.allclose(np.asarray(u.data)[mask], 2.5, atol=1e-5)

    def test_iterates_parent_dimensions_only(self):
        """
        The SubGrid Dimensions are not iterated: `vp` is read through a
        stencil indexed by the parent's Dimensions.
        """
        grid = Grid(shape=(13, 13), extent=(12., 12.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        op = Operator(Eq(u, vp))

        dims = [i.dim.name for i in FindNodes(Iteration).visit(op)]
        assert dims == [d.name for d in grid.dimensions]

    def test_every_axis_relocated_under_blocking(self):
        """
        In 3D, blocking has already run when the subgrid pass sees the Cluster,
        so a parent Dimension appears in the IterationSpace only through its
        block Dimensions. Every axis must still be relocated -- the symptom of
        getting this wrong is a read left indexed by a SubGrid Dimension.
        """
        grid = Grid(shape=(17, 17, 17), extent=(16.,)*3)
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        op = Operator(Eq(u, vp))

        e, = [i.expr for i in FindNodes(Expression).visit(op) if i.write is u]
        reads = [i for i in retrieve_indexed(e) if i.function is vp]
        assert reads

        gone = set(sub.dimensions)
        for i in reads:
            assert not (set(i.free_symbols) & gone)

        # And the loop nest is the parent's, not a deeper one
        dims = [i.dim for i in FindNodes(Iteration).visit(op)]
        assert not (set(dims) & gone)

    def test_shared_factor_one_dimension_still_iterates(self):
        """
        An axis of factor 1 shares the parent's own Dimension, so it must stay
        in the IterationSpace -- dropping it leaves a loop bound with no
        runtime value.
        """
        grid = Grid(shape=(13, 13, 13), extent=(12.,)*3)
        sub = SubGrid(grid, factor=(2, 1.5, 1.0))
        assert sub.dimensions[2] is grid.dimensions[2]

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        vp.data[:] = 2.5

        op = Operator(Eq(u, vp))

        dims = [i.dim.name for i in FindNodes(Iteration).visit(op)]
        assert dims == [d.name for d in grid.dimensions]

        op.apply()      # must not raise: `z` needs its bounds
        assert np.allclose(np.asarray(u.data)[:-1, :-1, :], 2.5, atol=1e-5)

    def test_is_a_plain_stencil(self):
        """
        No accumulator and no per-tap branch: the taps are a flat weighted
        sum, so the inner loop stays vectorizable.
        """
        grid = Grid(shape=(25, 25), extent=(24., 24.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        with switchconfig(language='openmp'):
            ccode = str(Operator(Eq(u, vp)).ccode)

        assert 'for (int r' not in ccode
        assert 'if (' not in ccode
        assert 'floor' not in ccode
        assert '#pragma omp simd' in ccode

    def test_weights_are_compile_time(self):
        """The per-phase weights are a stack table, not computed per point."""
        grid = Grid(shape=(13, 13), extent=(12., 12.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        ccode = str(Operator(Eq(u, vp)).ccode)

        # `factor = 3/2` reading at the parent's points gives 3 phases of 2
        # taps, i.e. a 6-entry table
        assert '[6]' in ccode

    @pytest.mark.parametrize('ndim,taps', [(1, 2), (2, 4), (3, 8)])
    def test_tap_count(self, ndim, taps):
        """Linear interpolation reads `2**ndim` SubGrid points."""
        n = 13
        grid = Grid(shape=(n,)*ndim, extent=(float(n - 1),)*ndim)
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        op = Operator(Eq(u, vp))
        expr, = [i.expr for i in FindNodes(Expression).visit(op) if i.write is u]

        assert len([i for i in retrieve_indexed(expr.rhs)
                    if i.function is vp]) == taps


class TestExactness:

    """
    Linear interpolation is exact on a linear field -- the sharpest check, as
    any error in a base index or a weight shows up immediately.
    """

    @pytest.mark.parametrize('factor', [1.5, 2.5, 0.5, 0.75, 1.25])
    def test_1d(self, factor):
        n = 25
        sub, u, _ = restricted(factor, n=n, ndim=1)

        mask, idx = interior(sub, n, 1)
        expected = expected_at_parent(sub, idx)

        assert np.allclose(np.asarray(u.data)[mask], expected[mask], atol=2e-4)

    @pytest.mark.parametrize('factor', [1.5, 2.5, 0.5, (2, 1.5), (0.5, 1.5)])
    def test_2d(self, factor):
        n = 25
        sub, u, _ = restricted(factor, n=n, ndim=2)

        mask, idx = interior(sub, n, 2)
        expected = expected_at_parent(sub, idx)

        assert np.allclose(np.asarray(u.data)[mask], expected[mask], atol=2e-4)

    @pytest.mark.parametrize('factor', [1.5, 2.5, (2, 1.5, 1.)])
    def test_3d(self, factor):
        """
        3D is the case where blocking has already run by the time the subgrid
        pass sees the Cluster, so a parent Dimension shows up only through its
        block Dimensions. Every axis must still be relocated.
        """
        n = 17
        sub, u, _ = restricted(factor, n=n, ndim=3)

        mask, idx = interior(sub, n, 3)
        expected = expected_at_parent(sub, idx)

        assert np.allclose(np.asarray(u.data)[mask], expected[mask], atol=2e-4)

    def test_constant_field(self):
        """The weights must sum to one at every phase."""
        n = 25
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        vp.data[:] = -3.25
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        mask, _ = interior(sub, n, 2)
        assert np.allclose(np.asarray(u.data)[mask], -3.25, atol=1e-5)

    def test_refinement(self):
        """A factor below 1 refines: every parent point is an exact hit."""
        n = 25
        sub, u, vp = restricted(0.5, n=n, ndim=1)

        assert sub.shape[0] > n
        mask, idx = interior(sub, n, 1)
        assert np.allclose(np.asarray(u.data)[mask], (3.*idx[0]/0.5)[mask],
                           atol=1e-4)


class TestIntegerFactor:

    """
    An integer factor needs no interpolation when *writing* the SubGrid -- its
    points coincide with the parent's, so `Eq(g, u)` is a strided copy and
    keeps Part 1's lowering.

    *Reading* one on the parent grid is a different matter: `Eq(u, g)` is
    evaluated at every parent point, and the points in between two SubGrid
    points do need interpolating.
    """

    def test_not_interpolated(self):
        grid = Grid(shape=(12, 12))

        assert not SubGrid(grid, factor=3).is_interpolated
        assert SubGrid(grid, factor=1.5).is_interpolated

    def test_strided_copy(self):
        n = 24
        grid = Grid(shape=(n, n))
        sub = SubGrid(grid, factor=3)

        u = Function(name='u', grid=grid)
        g = Function(name='g', grid=sub, space_order=0)
        data = np.arange(n*n, dtype=u.dtype).reshape(n, n)
        u.data[:] = data

        Operator(Eq(g, u)).apply()

        assert np.array_equal(np.asarray(g.data), data[::3, ::3])

    def test_read_writes_every_parent_point(self):
        """
        `Eq(u, vp*u)` writes `u`, so every parent point must be computed. The
        Cluster still carries the SubGrid's GuardFactors at this stage; leaving
        them behind lets `subsampling` strengthen them into strided iteration
        and only every `factor`-th point of `u` gets written -- silently wrong
        results, not a crash.
        """
        n = 9
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=2)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        u.data[:] = 1.
        vp.data[:] = 3.

        op = Operator(Eq(u, vp*u))

        # The loop nest is the parent's, not the SubGrid's
        dims = [i.dim.name for i in FindNodes(Iteration).visit(op)]
        assert dims == [d.name for d in grid.dimensions]

        op.apply()

        assert not np.any(np.asarray(u.data) == 1.)
        assert np.allclose(np.asarray(u.data), 3., atol=1e-5)

    def test_read_is_exact_on_a_linear_field(self):
        """
        Reading an integer-factor SubGrid at *every* parent point interpolates
        at the points falling between two SubGrid points, and linear
        interpolation is exact there.
        """
        n = 17
        grid = Grid(shape=(n,), extent=(float(n - 1),))
        sub = SubGrid(grid, factor=2)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        vp.data[:] = (3.*np.arange(sub.shape[0])).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        idx = np.arange(n)
        inside = idx <= (sub.shape[0] - 1)*2 - 1
        expected = 3.*(idx/2.)

        assert np.allclose(np.asarray(u.data)[inside], expected[inside],
                           atol=1e-5)

    def test_write_stays_a_strided_copy(self):
        """Writing the SubGrid must not acquire interpolation weights."""
        grid = Grid(shape=(13, 13), extent=(12., 12.))
        sub = SubGrid(grid, factor=2)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        op = Operator(Eq(vp, u))

        dims = [i.dim.name for i in FindNodes(Iteration).visit(op)]
        assert dims == [d.name for d in sub.dimensions]
        assert 'float w' not in str(op)

    @pytest.mark.parametrize('factor', [3, (3, 1), (2, 3)])
    def test_scatter_per_axis(self, factor):
        """Each axis scatters on its own factor, including a factor-1 axis."""
        n = 13
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=factor)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        u.data[:] = -1.
        vp.data[:] = 7.

        Operator(Eq(u, vp, subdomain=sub)).apply()

        a = np.asarray(u.data)
        f0, f1 = [int(v) for v in sub.factor]
        assert np.all(a[::f0, ::f1] == 7.)
        # Nothing outside the aligned points was touched
        assert int((a != -1.).sum()) == a[::f0, ::f1].size

    @pytest.mark.parametrize('factor', [3, (3, 1), (2, 3)])
    def test_gather_per_axis_is_exact(self, factor):
        """
        Reading at every parent point is exact on a linear field for per-axis
        integer factors too, where the aligned and interpolated axes mix.
        """
        n = 17
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=factor)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        mask, idx = interior(sub, n, 2)
        expected = expected_at_parent(sub, idx)

        assert np.allclose(np.asarray(u.data)[mask], expected[mask], atol=2e-4)

    def test_explicit_scatter(self):
        """
        `subdomain=sub` iterates the SubGrid's points alone, so writing the
        parent from it scatters: every `factor`-th point is written and the
        rest keep their previous values.
        """
        n = 9
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=2)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        u.data[:] = -1.
        vp.data[:] = 7.

        op = Operator(Eq(u, vp, subdomain=sub))

        # The strided loop over the SubGrid, not the parent's
        dims = [i.dim.name for i in FindNodes(Iteration).visit(op)]
        assert dims == [d.name for d in sub.dimensions]
        assert 'float w' not in str(op)

        op.apply()

        a = np.asarray(u.data)
        assert np.all(a[::2, ::2] == 7.)
        assert np.all(a[1::2, :] == -1.)
        assert np.all(a[:, 1::2] == -1.)

    def test_scatter_rejected_when_interpolated(self):
        """
        An interpolated SubGrid has no points on the parent grid, so there is
        nothing to scatter onto -- it must raise, not silently gather.
        """
        grid = Grid(shape=(9, 9), extent=(8., 8.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        # Rejected where the mistake is made, at Eq construction
        with pytest.raises(ValueError, match='fall \\*between\\*'):
            Eq(u, vp, subdomain=sub)

    def test_self_reference(self):
        """
        A SubGrid Function read at a shift of its own Dimensions is a
        recurrence, not a relocation -- it must not be interpolated.
        """
        grid = Grid(shape=(12, 12))
        sub = SubGrid(grid, factor=(3, 1))
        xs = sub.dimensions[0]

        f = Function(name='f', grid=grid)
        g = Function(name='g', grid=sub, space_order=0)
        f.data[:] = 1.
        g.data[:] = 0.

        Operator(Eq(g, g.subs(xs, xs - 1) + f)).apply()

        assert np.allclose(np.asarray(g.data)[:, 0], [0., 1., 2., 3.])


class TestSchemes:

    """
    `interpolation=` selects the scheme: `linear` (2 taps), `cubic` (4,
    Catmull-Rom) or `sinc` (8, Kaiser windowed).
    """

    @pytest.mark.parametrize('scheme,ntaps', [('linear', 2), ('cubic', 4),
                                              ('sinc', 8)])
    def test_tap_count(self, scheme, ntaps):
        grid = Grid(shape=(33,), extent=(32.,))
        sub = SubGrid(grid, factor=1.5, interpolation=scheme)

        u = Function(name='u', grid=grid, space_order=8)
        vp = Function(name='vp', grid=sub, space_order=8)

        op = Operator(Eq(u, vp))

        e, = [i.expr for i in FindNodes(Expression).visit(op) if i.write is u]
        reads = {i for i in retrieve_indexed(e) if i.function is vp}

        assert len(reads) == ntaps

    def test_default_is_linear(self):
        grid = Grid(shape=(12,))

        assert SubGrid(grid, factor=1.5).interpolation == 'linear'

    def test_unknown_scheme_rejected(self):
        grid = Grid(shape=(12,))

        with pytest.raises(ValueError, match='Unknown interpolation scheme'):
            SubGrid(grid, factor=1.5, interpolation='quintic')

    @pytest.mark.parametrize('scheme', ['linear', 'cubic', 'sinc'])
    def test_weights_sum_to_one(self, scheme):
        """
        Partition of unity, at every phase -- otherwise a constant field picks
        up a bias. The windowed sinc needs explicit normalisation for this.
        """
        from devito.passes.clusters.subgrid import _taps

        for frac in np.linspace(0., 0.99, 25):
            assert np.isclose(_taps(scheme, frac).sum(), 1., atol=1e-12)

    @pytest.mark.parametrize('scheme,atol', [('linear', 2e-4), ('cubic', 2e-4),
                                             ('sinc', 5e-3)])
    def test_linear_field(self, scheme, atol):
        """
        `linear` and `cubic` reproduce a linear field *exactly* -- their taps
        match its first moment, `sum(w*pos) == frac`. Windowed `sinc` does not:
        truncating the window perturbs that moment by ~2e-4, so it is accurate
        but not exact. That is inherent to the scheme, not a defect.
        """
        n = 33
        grid = Grid(shape=(n,), extent=(float(n - 1),))
        sub = SubGrid(grid, factor=1.5, interpolation=scheme)

        u = Function(name='u', grid=grid, space_order=8)
        vp = Function(name='vp', grid=sub, space_order=8)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        from devito.passes.clusters.subgrid import SCHEMES
        pad = int(np.ceil(SCHEMES[scheme]*1.5)) + 1
        idx = np.arange(n)
        mask = (idx >= pad) & (idx <= int((sub.shape[0] - 1)*1.5) - pad)
        expected = 3.*(idx/1.5)

        assert np.allclose(np.asarray(u.data)[mask], expected[mask], atol=atol)

    @pytest.mark.parametrize('scheme', ['linear', 'cubic'])
    def test_first_moment_is_exact(self, scheme):
        """
        `sum(w*pos) == frac` is what makes a scheme exact on linear fields.
        Excludes `sinc`, which trades this for its spectral accuracy.
        """
        from devito.passes.clusters.subgrid import SCHEMES, _taps

        r = SCHEMES[scheme]
        pos = np.arange(-r + 1, r + 1)
        for frac in np.linspace(0., 0.99, 25):
            assert np.isclose((_taps(scheme, frac)*pos).sum(), frac,
                              atol=1e-12)

    def test_higher_order_is_more_accurate(self):
        """
        On a field the SubGrid only marginally resolves, the wider stencils win
        -- which is the whole point of offering them. `sinc` is the best here;
        on a very smooth field `cubic` edges it, so this uses a short
        wavelength where the ordering is unambiguous.
        """
        n = 129
        fac = 1.5
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        k = 2*np.pi*8/(n - 1)        # ~10 SubGrid points per cycle

        errs = {}
        for scheme in ('linear', 'cubic', 'sinc'):
            sub = SubGrid(grid, factor=fac, interpolation=scheme,
                          name=scheme[0])
            u = Function(name=f'u{scheme[0]}', grid=grid, space_order=8)
            vp = Function(name=f'v{scheme[0]}', grid=sub, space_order=8)

            i, j = np.meshgrid(*[np.arange(s) for s in sub.shape],
                               indexing='ij')
            vp.data[:] = np.sin(k*i*fac)*np.cos(k*j*fac)
            u.data[:] = 0.

            Operator(Eq(u, vp)).apply()

            from devito.passes.clusters.subgrid import SCHEMES
            pad = int(np.ceil(SCHEMES[scheme]*fac)) + 1
            px, py = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            exact = np.sin(k*px)*np.cos(k*py)
            mask = np.ones_like(px, dtype=bool)
            for q, c in enumerate((px, py)):
                mask &= (c >= pad) & (c <= int((sub.shape[q] - 1)*fac) - pad)

            errs[scheme] = np.abs(np.asarray(u.data)[mask]
                                  - exact[mask]).max()

        assert errs['cubic'] < errs['linear']/5.
        assert errs['sinc'] < errs['cubic']/5.


class TestMPI:

    """
    Interpolated reads under domain decomposition.

    The parent Grid and the SubGrid are decomposed independently, so a rank's
    parent points generally need SubGrid points its neighbour owns. Two things
    have to be right for this to work: the halo exchange must happen at all,
    and the base index must be computed on the *global* index.
    """

    @pytest.mark.parallel(mode=[2, 4])
    @pytest.mark.parametrize('factor', [1.5, 2.5, 0.75])
    def test_read_matches_serial(self, mode, factor):
        """
        Linear interpolation is exact on a linear field, on every rank. Getting
        the index space wrong leaves rank 0 correct and the others wrong, so
        this must assert on all of them.
        """
        n = 25
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=factor)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        mask, idx = interior(sub, n, 2)
        expected = expected_at_parent(sub, idx)

        # Compare on this rank's share of the global domain
        loc = u.local_indices
        assert np.allclose(np.asarray(u.data)[mask[loc]],
                           expected[loc][mask[loc]], atol=2e-4)

    @pytest.mark.parallel(mode=[2, 4])
    def test_base_index_is_global(self, mode):
        """
        `IntDiv`/`Mod` are non-linear, so they must be evaluated on the global
        index. Computing them on the local one is correct only on the rank
        whose offsets are zero, which is why `test_read_matches_serial` has to
        check every rank.
        """
        grid = Grid(shape=(25, 25), extent=(24., 24.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        op = Operator(Eq(u, vp))

        e, = [i.expr for i in FindNodes(Expression).visit(op) if i.write is u]
        syms = {s.name for s in e.free_symbols}
        for d, ds in zip(grid.dimensions, sub.dimensions, strict=True):
            assert f'oi_{d.name}' in syms
            assert f'oi_{ds.name}' in syms

    @pytest.mark.parallel(mode=[2, 4])
    def test_read_matches_serial_3d(self, mode):
        """
        3D is where blocking has already run when the subgrid pass sees the
        Cluster, so it exercises the block-promotion and the global index
        together.
        """
        n = 17
        grid = Grid(shape=(n,)*3, extent=(float(n - 1),)*3)
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        mask, idx = interior(sub, n, 3)
        expected = expected_at_parent(sub, idx)
        loc = u.local_indices

        assert np.allclose(np.asarray(u.data)[mask[loc]],
                           expected[loc][mask[loc]], atol=2e-4)

    @pytest.mark.parallel(mode=2)
    def test_needs_a_halo_to_interpolate(self, mode):
        """
        The second tap reads `base + 1`, which on a decomposed SubGrid falls in
        the halo. With `space_order=0` there is none, so the read would run out
        of bounds -- silently, and only on the ranks that aren't first.
        """
        grid = Grid(shape=(25, 25), extent=(24., 24.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=0)

        with pytest.raises(ValueError, match='out of bounds'):
            Operator(Eq(u, vp))

        # One layer is enough
        vp1 = Function(name='vp1', grid=sub, space_order=1)
        vp1.data[:] = linear(sub.shape).astype(vp1.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp1)).apply()

        mask, idx = interior(sub, 25, 2)
        expected = expected_at_parent(sub, idx)
        loc = u.local_indices
        assert np.allclose(np.asarray(u.data)[mask[loc]],
                           expected[loc][mask[loc]], atol=2e-4)

    @pytest.mark.parallel(mode=[2, 4])
    @pytest.mark.parametrize('scheme', ['linear', 'cubic', 'sinc'])
    def test_every_scheme_matches_serial(self, mode, scheme):
        """The wider stencils reach further into the halo, but still agree."""
        from devito.passes.clusters.subgrid import SCHEMES

        n, fac = 65, 1.5
        grid = Grid(shape=(n, n), extent=(float(n - 1),)*2)
        sub = SubGrid(grid, factor=fac, interpolation=scheme)

        u = Function(name='u', grid=grid, space_order=8)
        vp = Function(name='vp', grid=sub, space_order=8)
        vp.data[:] = linear(sub.shape).astype(vp.dtype)
        u.data[:] = 0.

        Operator(Eq(u, vp)).apply()

        pad = int(np.ceil(SCHEMES[scheme]*fac)) + 1
        px, py = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        expected = 3.*(px/fac) + 2.*(py/fac)
        mask = np.ones_like(px, dtype=bool)
        for q, c in enumerate((px, py)):
            mask &= (c >= pad) & (c <= int((sub.shape[q] - 1)*fac) - pad)

        loc = u.local_indices
        assert np.allclose(np.asarray(u.data)[mask[loc]],
                           expected[loc][mask[loc]], atol=5e-3)

    @pytest.mark.parallel(mode=2)
    @pytest.mark.parametrize('scheme,ok_so', [('cubic', 2), ('sinc', 4)])
    def test_halo_depth_scales_with_the_scheme(self, mode, scheme, ok_so):
        """A wider stencil needs a deeper halo, and says so."""
        grid = Grid(shape=(65, 65), extent=(64., 64.))
        sub = SubGrid(grid, factor=1.5, interpolation=scheme)

        u = Function(name='u', grid=grid, space_order=8)

        too_thin = Function(name='thin', grid=sub, space_order=ok_so - 1)
        with pytest.raises(ValueError, match='out of bounds'):
            Operator(Eq(u, too_thin))

        enough = Function(name='thick', grid=sub, space_order=ok_so)
        Operator(Eq(u, enough))      # must not raise

    @pytest.mark.parallel(mode=2)
    def test_halo_exchange_is_emitted(self, mode):
        """
        `vp` is never written, but its halo is stale on entry -- `.data` only
        fills the local domain -- so the exchange must not be dropped.
        """
        grid = Grid(shape=(25, 25), extent=(24., 24.))
        sub = SubGrid(grid, factor=1.5)

        u = Function(name='u', grid=grid, space_order=2)
        vp = Function(name='vp', grid=sub, space_order=2)

        op = Operator(Eq(u, vp))

        assert len(FindNodes(HaloSpot).visit(op)) == 0      # already lowered
        assert 'haloupdate' in str(op)
