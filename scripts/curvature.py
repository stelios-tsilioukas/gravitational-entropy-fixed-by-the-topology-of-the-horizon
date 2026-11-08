"""Shared helper: curvature invariants of a metric given in coordinates.

curvature(g, x) returns (R, Ric_ab Ric^ab, R_abcd R^abcd, GB) for the metric
matrix g (sympy Matrix) in the coordinates x, with
GB = R^2 - 4 Ric_ab Ric^ab + R_abcd R^abcd the Gauss-Bonnet density.
"""
import sympy as sp


def christoffel(g, x):
    n = len(x)
    ginv = g.inv()
    return [[[sp.simplify(sum(ginv[l, s] * (sp.diff(g[s, i], x[j]) + sp.diff(g[s, j], x[i])
                                            - sp.diff(g[i, j], x[s])) for s in range(n)) / 2)
              for j in range(n)] for i in range(n)] for l in range(n)]


def riemann_up(g, x):
    """R^l_{k i j} with the convention R^l_{kij} = d_i Gam^l_{jk} - d_j Gam^l_{ik} + ..."""
    n = len(x)
    Gam = christoffel(g, x)
    return [[[[sp.simplify(sp.diff(Gam[l][j][k], x[i]) - sp.diff(Gam[l][i][k], x[j])
                           + sum(Gam[l][i][s] * Gam[s][j][k] - Gam[l][j][s] * Gam[s][i][k]
                                 for s in range(n)))
               for k in range(n)] for j in range(n)] for i in range(n)] for l in range(n)]


def curvature(g, x):
    n = len(x)
    ginv = g.inv()
    Riem = riemann_up(g, x)
    Rd = [[[[sp.simplify(sum(g[l, m] * Riem[m][i][j][k] for m in range(n)))
             for k in range(n)] for j in range(n)] for i in range(n)] for l in range(n)]
    Ric = sp.Matrix(n, n, lambda i, j: sp.simplify(sum(Riem[l][l][i][j] for l in range(n))))
    Rs = sp.simplify(sum(ginv[i, j] * Ric[i, j] for i in range(n) for j in range(n)))
    Ric2 = sp.simplify(sum(ginv[i, k] * ginv[j, l] * Ric[i, j] * Ric[k, l]
                           for i in range(n) for j in range(n) for k in range(n) for l in range(n)))
    Riem2 = 0
    idx = range(n)
    for A in idx:
        for B in idx:
            for C in idx:
                for D in idx:
                    if Rd[A][B][C][D] == 0:
                        continue
                    for E in idx:
                        for F in idx:
                            for G in idx:
                                for H in idx:
                                    if Rd[E][F][G][H] == 0:
                                        continue
                                    Riem2 += (ginv[A, E] * ginv[B, F] * ginv[C, G] * ginv[D, H]
                                              * Rd[A][B][C][D] * Rd[E][F][G][H])
    Riem2 = sp.simplify(Riem2)
    GB = sp.simplify(Rs ** 2 - 4 * Ric2 + Riem2)
    return Rs, Ric2, Riem2, GB
