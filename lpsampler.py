#!/usr/bin/python
"""
lpsampler is a tool for point sampling on multi-dimensional spheres in Lp space.
In particular lpsampler allows for the generation of non-dominated
pareto fronts in arbitrary dimension and number of points.
Lp sampler Supports random sampling and Lp distance based sampling (Convex, Planar and
Concave shapes of an arbitrary curvature coefficient) in the positive subspace.

Usage:
  lpsampler generate (gauss <k> | random) <dimension> <n_points> [options]
  lpsampler generate Sample2014

Options:
  -m <epsilon>, --mirror=<epsilon>  Mirrors the front by the origin and
                                    adds <epsilon> to each objective
  -o <file>, --output=<file>        Writes the front to
                                    file [default: PaGMO_Front.txt]

Examples:
  Create a random front in 10 dimension and 100 points.

    PHT.py generate random 10 100 -o PaGMO.Random.10d.100n.txt

  Create a spherical (k=2.0) front in 5 dimensions and 30 points.
  Afterwards, mirror the front by origin and add 1.0 epsilon:

    PHT.py generate gauss 2.0 5 30 -m 1.0

  When k=1 the resulting shape is a planar surface.
  For k < 1.0 we obtain the concave shape.
  Providing a "-m=1.0" maps each point [x_1, x_2, ..., x_n]
  to [1.0 - x_1, 1.0 - x_2, ..., 1.0 - x_n].

  Generate the Sample2014 dataset in current directory:

    PHT.py generate Sample2014
"""

import os
import numpy as np
from itertools import product


def a_dominates_b(a, b):
    return (all(a[i] <= b[i] for i in range(len(a)))
            and any(a[i] < b[i] for i in range(len(a))))


def a_equals_b(a, b):
    return all(a[i] == b[i] for i in range(len(a)))


def a_incomparable_b(a, b):
    if a_equals_b(a, b) or a_dominates_b(a, b) or a_dominates_b(b, a):
        return False
    else:
        return True


def a_incomparable_S(a, S):
    """Point 'a' is a valid addition to the set 'S':
        1) 'a' is not dominated
        2) 'a' does not dominate any of the points in 'S'
        3) 'a' is not equal to any of the points in 'S'
    """
    return all(a_incomparable_b(a, b) for b in S)


def sample_random(dimension, n_points, mirror_f=lambda x: x):
    """Generates a random front. """

    points = []

    while len(points) < n_points:
        new = [mirror_f(x_i) for x_i in np.random.uniform(0.0, 1.0, dimension)]
        if a_incomparable_S(new, points):
            points.append([mirror_f(x_i) for x_i in new])
    return points


def sample_gauss(dimension, n_points, k, mirror_f=lambda x: x):
    """Samples the front using the normal variate method."""

    points = []

    def make_pow(k):
        return lambda x: x ** k

    _pow = make_pow(k)
    _sqrt = make_pow(1.0 / k)

    while len(points) < n_points:
        X = [abs(xi) for xi in np.random.normal(0.0, 1.0, dimension)]
        norm = _sqrt(sum(_pow(xi) for xi in X))
        X = [mirror_f(xi / norm) for xi in X]

        # Check for dominance just to be sure
        # if a_incomparable_S(X, points):
        points.append(X)
    return points


def generate_Sample2014():
    set_name = "Sample2014"

    # Setup a fixed seed to numpy:
    np.random.seed(42478512)

    # ns = [10, 20,... , 90, 100, 200,... , 900, 1000]
    ns = list(range(10, 100, 10)) + list(range(100, 1000 + 1, 100))

    # ds = [2, 3,... , 19, 20, 30,... , 90, 100]
    ds = list(range(2, 20)) + list(range(20, 100 + 1, 10))

    identity_f = lambda x: x
    mirror_f = lambda x: 1.0 - x

    gauss_shapes = (
        ("Convex", 2, identity_f),
        ("Planar", 1, identity_f),
        ("Concave", 0.5, identity_f),
        ("InvertedConvex", 2, mirror_f),
        ("InvertedPlanar", 1, mirror_f),
        ("InvertedConcave", 0.5, mirror_f),
    )

    def make_dirpath(set_name, shape_name, dimension, n_points):
        """Returns like 'PaGMO/Planar/10d/150n'."""
        return os.path.join(set_name, shape_name,
                            "%dd" % (dimension, ), "%dn" % (n_points, ))

    def make_filepath(set_name, shape_name, dimension, n_points, i):
        """Returns like 'PaGMO/Planar/10d/150n/PaGMO.Planar.10d.150n.10'."""
        dir_p = make_dirpath(set_name, shape_name, dimension, n_points)
        f_name = "%s.%s.%sd.%sn.%s" % (set_name, shape_name,
                                       dimension, n_points, i)
        return os.path.join(dir_p, f_name)

    # number of fronts to generate per combination of dimension and n_points
    n_sets = 10

    # Create directories:
    # Sample2014/{Planar,... , Random}/{2, 3, ..., 90, 100}d/{10, ..., 1000}n/
    dir_vars = product([n for n, _, _ in gauss_shapes] + ['Random', ], ds, ns)
    for shape_name, dimension, n_points in dir_vars:
        path = make_dirpath(set_name, shape_name, dimension, n_points)
        if not os.path.exists(path):
            os.makedirs(path)

    # Generate gaussian fronts
    for (shape_name, k, inv_f), dimension, n_points in product(gauss_shapes, ds, ns):
        print("Generating", shape_name, dimension, n_points)
        for i in range(1, n_sets + 1):
            filename = make_filepath(set_name, shape_name, dimension, n_points, i)
            front = sample_gauss(dimension, n_points, k, inv_f)
            np.savetxt(filename, front)

    # Generate random fronts
    for dimension, n_points in product(ds, ns):
        print("Generating", dimension, n_points, "Random")
        for i in range(1, n_sets + 1):
            filename = make_filepath(set_name, "Random", dimension, n_points, i)
            front = sample_random(dimension, n_points)
            np.savetxt(filename, front)


def docopt_handle():
    args = docopt(__doc__, version=1.0)
    if args['Sample2014']:
        generate_Sample2014()
    else:
        # If --mirror option is provided, prepare the mirror_f function.
        mirror_f = lambda x: x
        if args['--invert']:
            inv_eps = float(args['--invert'])
            mirror_f = lambda x: inv_eps - x
        dimension = int(args['<dimension>'])
        n_points = int(args['<n_points>'])

        if args['random']:
            points = sample_random(dimension, n_points, mirror_f=mirror_f)
        elif args['gauss']:
            k = float(args['<k>'])
            points = sample_gauss(dimension, n_points, k, mirror_f=mirror_f)

        # Save the front as text file
        np.savetxt(args['--output'], points)

if __name__ == "__main__":
    try:
        from docopt import docopt
        docopt_handle()
    except ImportError:
        generate_Sample2014()
