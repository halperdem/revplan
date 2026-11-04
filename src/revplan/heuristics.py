"""Greedy rule, assembly allocation decoder and genetic algorithms (Section 4)."""
from __future__ import annotations
import time
from collections import Counter, defaultdict
import numpy as np
from .data import Instance


def greedy(inst: Instance):
    """Add Q units one at a time; each unit takes fitting assemblies by value, then stock."""
    n = len(inst.pool)
    x = np.zeros(n, dtype=int)
    w = Counter()
    rem = inst.s.copy()
    yrem = dict(inst.Ys)
    fitj = defaultdict(list)
    for j, y in inst.pairs:
        fitj[j].append(y)
    order = {j: sorted(fitj[j], key=lambda y: -inst.V[y]) for j in fitj}
    for _ in range(inst.Q):
        best = None
        for j in range(n):
            dem = inst.A[:, j].copy()
            used, gain = [], 0.0
            for y in order.get(j, []):
                if yrem[y] >= 1 and np.all(dem >= inst.Bm[y] - 1e-9):
                    dem -= inst.Bm[y]; used.append(y); gain += inst.V[y]
            u = np.minimum(dem, rem)
            g = gain + float(inst.v @ (2 * u - dem))
            if best is None or g > best[0]:
                best = (g, j, used, u)
        _, j, used, u = best
        x[j] += 1
        rem -= u
        for y in used:
            yrem[y] -= 1; w[(j, y)] += 1
    return x, dict(w)


class Decoder:
    """Value of a plan x when assemblies are allocated by the greedy unit rule (cached)."""

    def __init__(self, inst: Instance):
        self.inst = inst
        self.ys = inst.ys
        self.yi = {y: k for k, y in enumerate(inst.ys)}
        self.Bmat = np.array([inst.Bm[y] for y in inst.ys]).T if inst.ys else np.zeros((len(inst.components), 0))
        self.V = np.array([inst.V[y] for y in inst.ys])
        self.stock0 = np.array([inst.Ys[y] for y in inst.ys])
        fitj = defaultdict(list)
        for j, y in inst.pairs:
            fitj[j].append(self.yi[y])
        self.fit = {}
        for j, lst in fitj.items():
            lst = sorted(lst, key=lambda k: -self.V[k])
            sup = np.unique(np.concatenate([np.flatnonzero(self.Bmat[:, k]) for k in lst]))
            self.fit[j] = (lst, self.Bmat[sup][:, lst], inst.A[sup, j])
        self.order = sorted(self.fit)
        self.cache = {}
        self.evals = 0

    def allocate(self, x):
        stock = self.stock0.copy()
        W = np.zeros(len(self.ys))
        for j in self.order:
            left = int(x[j])
            if not left:
                continue
            lst, Bs, a = self.fit[j]
            while left > 0:
                rem = a.copy(); S = []
                for t, k in enumerate(lst):
                    if stock[k] >= 1 and np.all(rem >= Bs[:, t] - 1e-9):
                        rem = rem - Bs[:, t]; S.append(k)
                if not S:
                    break
                take = int(min(left, min(stock[k] for k in S)))
                for k in S:
                    stock[k] -= take; W[k] += take
                left -= take
        return W

    def score(self, X):
        X = np.atleast_2d(X)
        out = np.empty(len(X))
        for r in range(len(X)):
            key = X[r].tobytes()
            if key in self.cache:
                out[r] = self.cache[key]; continue
            self.evals += 1
            W = self.allocate(X[r])
            d = self.inst.A @ X[r] - self.Bmat @ W
            u = np.minimum(d, self.inst.s)
            z = float(self.inst.v @ (2 * u - d) + self.V @ W)
            self.cache[key] = z
            out[r] = z
        if len(self.cache) > 200000:
            self.cache.clear()
        return out


def ga(inst: Instance, seed: int, seed_x=None, population: int = 100, max_gen: int = 2000,
       stall: int = 200, crossover: float = 0.8, mutation: float = 0.05, elites: int = 2):
    """Integer GA with exact-Q repair; seed_x inserts the greedy plan (greedy-seeded GA)."""
    dec = Decoder(inst)
    n, Q = len(inst.pool), inst.Q
    rng = np.random.default_rng(seed)
    X = np.array([rng.multinomial(Q, rng.dirichlet(np.ones(n))) for _ in range(population)])
    if seed_x is not None:
        X[0] = seed_x
    hist, best, last = [], -np.inf, 0
    t0 = time.perf_counter()
    for gen in range(max_gen + 1):
        f = dec.score(X)
        order = np.argsort(-f, kind="stable")
        b = float(f[order[0]])
        hist.append(b)
        if gen == 0 or b > best + 1e-9 * max(1.0, abs(best)):
            best, last = b, gen
        if gen == max_gen or gen - last >= stall:
            break
        wts = f - f.min()
        wts += max(float(wts.max()), 1.0) * 1e-12
        par = rng.choice(population, size=(population - elites, 2), p=wts / wts.sum())
        ch = X[par[:, 0]].copy()
        nc = len(ch)
        if n > 1:
            cuts = rng.integers(1, n, size=nc)
            cr = rng.random(nc) < crossover
            mask = (np.arange(n)[None, :] >= cuts[:, None]) & cr[:, None]
            ch = np.where(mask, X[par[:, 1]], ch)
        for r in np.flatnonzero(ch.sum(1) != Q):
            tot = int(ch[r].sum())
            if tot == 0:
                ch[r] = rng.multinomial(Q, np.ones(n) / n); continue
            sc = ch[r] * Q / tot
            fx = np.floor(sc).astype(int)
            left = Q - int(fx.sum())
            fr = sc - fx
            if left:
                fx[rng.choice(n, left, replace=False, p=fr / fr.sum())] += 1
            ch[r] = fx
        for j in range(n):
            sel = np.flatnonzero((rng.random(nc) < mutation) & (ch[:, j] > 0))
            if not len(sel):
                continue
            dest = rng.integers(n, size=len(sel))
            lim = np.minimum(ch[sel, j], max(1, Q // 10))
            amt = (rng.random(len(sel)) * lim).astype(int) + 1
            ch[sel, j] -= amt
            ch[sel, dest] += amt
        assert np.all(ch.sum(1) == Q)
        X = np.vstack([X[order[:elites]], ch])
    x = X[order[0]].copy()
    return dict(x=x, Z=hist[-1], generations=gen, seconds=time.perf_counter() - t0,
                evaluations=dec.evals, history=hist)
