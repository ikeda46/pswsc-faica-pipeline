"""Factor analysis + JADE (cumulant-based ICA) — "faica".

Ported from `tau0/empirical_tau0.py` (originally developed and validated
there on synthetic mixed-signal data and on daisy-mode DESHIMA
observations). Two stages, matching Ikeda & Toyama (Neural Networks, 2000):

1. `whiten_fa`: factor analysis (2nd-order statistics only) gives a
   pseudo-inverse whitening transform Q such that Q @ Lambda = I_m exactly
   (Lambda = fa.components_.T), so the signal part of z = Q(x - mean) has
   unit covariance while sensor-noise residual Q @ Sigma @ Q.T is
   deliberately left in place (it cannot be removed by any linear
   transform).
2. `jade`: joint diagonalization of 4th-order cumulant matrices fixes the
   rotation that 2nd-order statistics alone cannot determine, giving
   genuinely independent (not just decorrelated) components.
"""

import numpy as np


def whiten_fa(fa, X):
    """Ikeda & Toyama (Neural Networks, 2000) eq. (8) whitening.

    fa: a fitted `sklearn.decomposition.FactorAnalysis`.
    X: (T, n_channels) data the factor analysis was fit on.

    Returns (Z, Q): Z is (m, T) whitened signal, Q is the (m, n_channels)
    pseudo-inverse whitening matrix.
    """
    Lambda = fa.components_.T  # (n_channels, m)
    inv_psi = 1.0 / fa.noise_variance_  # (n_channels,)
    LtPsiInv = Lambda.T * inv_psi  # (m, n_channels)
    Q = np.linalg.solve(LtPsiInv @ Lambda, LtPsiInv)  # (m, n_channels)
    Xc = X - fa.mean_
    return Q @ Xc.T, Q  # (m, T), (m, n_channels)


def jade(Z, max_sweeps=50, tol=1e-6):
    """ICA via joint diagonalization of 4th-order cumulant matrices.

    Z: (m, T) whitened (decorrelated, unit-variance) signal.
    Returns the separating orthogonal matrix V (m, m); independent
    components are S = V @ Z.
    """
    m, T = Z.shape
    idx_pairs = [(p, q) for p in range(m) for q in range(p, m)]
    I = np.eye(m)
    CM = np.empty((len(idx_pairs), m, m))
    for k, (p, q) in enumerate(idx_pairs):
        M = np.zeros((m, m))
        if p == q:
            M[p, p] = 1.0
        else:
            M[p, q] = M[q, p] = 1.0 / np.sqrt(2)
        quad = np.sum(Z * (M @ Z), axis=0)  # z_t^T M z_t for each t
        Q = (Z * quad) @ Z.T / T
        CM[k] = Q - M - M.T - np.trace(M) * I

    V = np.eye(m)
    for _ in range(max_sweeps):
        max_theta = 0.0
        for p in range(m - 1):
            for q in range(p + 1, m):
                g = np.stack(
                    [CM[:, p, p] - CM[:, q, q], CM[:, p, q] + CM[:, q, p]], axis=0
                )
                gg = g @ g.T
                ton, toff = gg[0, 0] - gg[1, 1], gg[0, 1] + gg[1, 0]
                theta = -0.5 * np.arctan2(toff, ton + np.sqrt(ton**2 + toff**2))
                max_theta = max(max_theta, abs(theta))
                if abs(theta) < tol:
                    continue
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s], [s, c]])
                CM[:, [p, q], :] = np.einsum("ab,kbc->kac", R, CM[:, [p, q], :])
                CM[:, :, [p, q]] = np.einsum("kab,bc->kac", CM[:, :, [p, q]], R.T)
                V[[p, q], :] = R @ V[[p, q], :]
        if max_theta < tol:
            break
    return V
