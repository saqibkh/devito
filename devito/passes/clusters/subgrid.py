from fractions import Fraction

import numpy as np
import sympy

try:
    from scipy.special import i0
except ImportError:
    from numpy import i0

from devito.finite_differences.interpolation import SubGridInterp
from devito.ir.clusters import Queue
from devito.ir.support.guards import GuardFactor
from devito.symbolics import search, uxreplace
from devito.tools import timed_pass
from devito.types import Array, CustomDimension

__all__ = ['subgrid_interp']


@timed_pass()
def subgrid_interp(clusters, sregistry=None, **kwargs):
    """
    Lower a Cluster reading an interpolated SubGrid Function into a plain
    stencil.

    A SubGrid point `i` sits at parent coordinate `i*factor`, so reading it at
    the parent point `x` means evaluating at `x/factor`. With `factor = p/q`
    that coordinate is `x*q/p`, hence

        base = (x*q) // p        frac = ((x*q) % p) / p

    Both are plain integer arithmetic, and `frac` depends on `x` only through
    `x % p` -- so there are `p` phases, and within a phase the tap weights are
    *constants*. Only the weights vary between phases, so they go in a small
    compile-time table indexed by `Mod(x, p)`.

    No phase loop is therefore needed, and nothing is added to the iteration
    space: the SubGrid Dimensions are simply dropped and the inner loops stay
    unit-stride and branch-free.
    """
    return SubGridInterpolation(sregistry).process(clusters)


class SubGridInterpolation(Queue):

    def __init__(self, sregistry=None, **kwargs):
        super().__init__(**kwargs)
        self.sregistry = sregistry

    def callback(self, clusters, prefix):
        if prefix:
            return clusters

        processed = []
        for c in clusters:
            processed.append(_lower(c, self.sregistry))

        return processed


def _lower(cluster, sregistry):
    """Expand the SubGridInterp markers in one Cluster, or return it as is."""
    markers = set()
    for e in cluster.exprs:
        markers.update(search(e, SubGridInterp))

    if not markers:
        return cluster

    ispace = cluster.ispace
    weights = {}       # (factor, dtype) -> (Array, ntaps, nphases)
    subs = {}

    # Blocking may already have run, in which case a parent Dimension appears
    # in the IterationSpace only through its block Dimensions (`y -> y0_blk0`).
    # Promote those back so the membership test below sees the roots
    itdims = set(ispace.promote(lambda d: d.is_Block).itdims)

    for m in markers:
        grid = m.subgrid
        func = m.func_.function
        scheme = grid.interpolation

        terms = []
        for ds, d, fac in zip(grid.dimensions, grid.parent.dimensions,
                              grid.factor, strict=True):
            if d not in itdims:
                continue

            frac = Fraction(float(fac)).limit_denominator(10**6)
            key = (float(fac), func.dtype, scheme)
            if key not in weights:
                name = sregistry.make_name(prefix='w') if sregistry else f'w{d.name}'
                weights[key] = _phase_weights(fac, func.dtype, name, scheme)
            w, ntaps, nphases, lo = weights[key]

            # The taps span `base + lo .. base + lo + ntaps - 1`, which on a
            # decomposed SubGrid reaches into the halo on both sides. Without
            # enough of one the read runs out of bounds -- silently, and only
            # on the ranks that aren't at the boundary along `d`
            need = max(-lo, lo + ntaps - 1)
            if nphases > 1 and grid.distributor.is_parallel and \
               min(func._size_halo[ds]) < need:
                raise ValueError(
                    f"`{func.name}` is defined on the interpolated SubGrid "
                    f"`{grid}` with a halo of "
                    f"{tuple(func._size_halo[ds])} along `{ds}`, but "
                    f"`{scheme}` interpolation reads {need} points past the "
                    "base index, so interpolating it onto its parent Grid "
                    "under MPI would read out of bounds. Give "
                    f"`{func.name}` a `space_order` of at least {need}"
                )

            # `IntDiv` and `Mod` are non-linear, so they must be evaluated
            # on the *global* index; under MPI the local one differs by this
            # rank's offset. `oglb` is the global parent index, `soff` brings
            # the resulting SubGrid index back into local terms
            poff, soff = grid.ioffsets[d]
            oglb = d + poff

            # `nphases == 1` means every parent point lands exactly on a
            # SubGrid point: a single tap of unit weight, no interpolation
            ph = sympy.Mod(oglb, nphases) if nphases > 1 else None
            terms.append((ds, oglb, soff, ph, w, ntaps, lo, frac))

        if not terms:
            continue

        # The relocated read: a weighted sum over the taps of every relocated
        # Dimension, with the weights taken from each Dimension's phase row
        subs[m] = _stencil(func, terms)

    exprs = [uxreplace(e, subs) for e in cluster.exprs]

    # The SubGrid Dimensions are no longer iterated: the relocated reads are
    # indexed by the parent's Dimensions alone. An axis of factor 1 is the
    # exception -- there the SubGrid shares the parent's own Dimension, which
    # stays in the IterationSpace
    gone = {ds for m in markers
            for ds, d in zip(m.subgrid.dimensions, m.subgrid.parent.dimensions,
                             strict=True)
            if ds is not d}
    ispace = ispace.project(lambda d: d not in gone)

    # An integer-factor SubGrid has ConditionalDimensions, so the Cluster also
    # carries their GuardFactors -- keyed on the *parent* Dimension. Those
    # guards described the dropped Dimensions, and leaving them behind would
    # let `subsampling` strengthen them into strided iteration, writing only
    # every `factor`-th point of the parent-grid LHS
    guards = cluster.guards
    for d, v in cluster.guards.items():
        if any(g.d in gone for g in v.find(GuardFactor)):
            guards = guards.popany(d)

    return cluster.rebuild(exprs=exprs, ispace=ispace, guards=guards)


def _stencil(func, terms):
    """
    `func` read as a weighted sum over the taps of every relocated Dimension,
    with the weights taken from each Dimension's phase row.
    """
    from itertools import product

    retval = []
    ranges = [range(t[5]) if t[3] is not None else range(1) for t in terms]
    for offsets in product(*ranges):
        weight = sympy.S.One
        indices = {}
        for (ds, oglb, soff, ph, w, ntaps, lo, frac), o in zip(terms, offsets,
                                                               strict=True):
            if ph is not None:
                weight = weight*w[ntaps*ph + o]
            indices[ds] = _base(oglb, frac) - soff + lo + o

        # `func` is indexed directly here, which bypasses the halo shift that
        # index lowering applies to ordinary reads -- so add it explicitly
        idxs = [indices.get(d, d) + h for d, h in
                zip(func.dimensions, func._size_nodomain.left, strict=True)]
        retval.append(weight*func.indexed[tuple(idxs)])

    return sympy.Add(*retval)


def _base(d, frac):
    """
    The SubGrid base index for parent index `d`: `(d*q)//p` for a
    `factor = p/q`, in integer arithmetic.

    `d` is the raw iteration Dimension: index lowering adds each Function's
    halo afterwards, so neither halo takes part in the division.
    """
    from devito.symbolics import IntDiv
    return IntDiv(frac.denominator*d, frac.numerator)


# The interpolation schemes, as `name -> radius`. A scheme of radius `r` has
# `2*r` taps, sitting at `base - r + 1 .. base + r` so that the stencil is
# centred on the interval `[base, base + 1]` the position falls in
SCHEMES = {'linear': 1, 'cubic': 2, 'sinc': 4}

# Kaiser window parameter from Hicks (2002), Table 1 -- the same table
# `SincInterpolator` uses, keyed by the *number of taps*. (Fitting `b`
# numerically for the convention here recovers 3.8/6.5/8.3 for 4/6/8 taps,
# against the table's 4.14/6.40/8.56, so the table is the right one; it is
# `SincInterpolator` that indexes it by radius.)
_B_TABLE = {2: 2.94, 3: 4.53, 4: 4.14, 5: 5.26, 6: 6.40,
            7: 7.51, 8: 8.56, 9: 9.56, 10: 10.64}


def _taps(scheme, frac):
    """
    The interpolation weights at fractional position `frac` in `[0, 1)`.

    Returns the `ntaps` weights for the taps at offsets
    `-radius + 1 .. radius` from the base index.
    """
    r = SCHEMES[scheme]
    pos = np.arange(-r + 1, r + 1) - frac

    if scheme == 'linear':
        return np.array([1. - frac, frac])

    if scheme == 'cubic':
        # Catmull-Rom: C1-continuous, interpolating, and exact on linears
        t = np.abs(pos)
        w = np.where(
            t <= 1., (1.5*t - 2.5)*t*t + 1.,
            np.where(t < 2., ((-0.5*t + 2.5)*t - 4.)*t + 2., 0.)
        )
        return w

    # Kaiser windowed sinc, as in `_sinc_weights`
    b = _B_TABLE[2*r]
    w = i0(b*np.sqrt(np.maximum(0., 1. - (pos/r)**2)))/i0(b)*np.sinc(pos)

    # Truncating the window costs ~0.4% of the unit sum, which otherwise
    # appears as a constant bias large enough to swamp the scheme's accuracy.
    # `linear` and `cubic` sum to 1 exactly and need no such correction
    return w/w.sum()


def _phase_weights(factor, dtype, name, scheme):
    """
    The per-phase tap weights for *reading* a SubGrid of the given `factor` at
    its parent's points, as a compile-time table.

    A SubGrid point sits at parent coordinate `i*factor`, so conversely the
    parent point `x` sits at SubGrid coordinate `x/factor`. With
    `factor = p/q` that is `x*q/p`, hence

        base = (x*q)//p        frac = ((x*q) % p)/p

    `base` is the integer part, and `frac` depends on `x` only through
    `x % p`. So there are `p` phases, and within a phase the weights are
    constants -- an ordinary constant-offset stencil.

    Returns `(weights, ntaps, nphases, lo)`: the `Array` holding
    `nphases*ntaps` values, the taps per phase, the number of phases, and the
    offset of the first tap from the base index.
    """
    frac = Fraction(float(factor)).limit_denominator(10**6)
    p, q = frac.numerator, frac.denominator

    r = SCHEMES[scheme]
    # Reading the SubGrid at the parent's points inverts the factor, so the
    # phase period is `p`
    nphases, ntaps, lo = p, 2*r, -r + 1

    values = []
    for ph in range(nphases):
        f = float(Fraction((ph*q) % p, p))
        values.extend(sympy.Float(v) for v in _taps(scheme, f))

    d = CustomDimension(name=f'{name}d', symbolic_min=0,
                        symbolic_max=nphases*ntaps - 1,
                        symbolic_size=nphases*ntaps)
    weights = Array(name=name, dimensions=(d,), shape=(nphases*ntaps,),
                    dtype=dtype, scope='stack', initvalue=values)

    return weights, ntaps, nphases, lo
