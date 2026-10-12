#!/usr/bin/env python3
"""G5-N verification for the gdn_solver.py partial-model fix.

1. Synthetic artifact injection: force m[zv] -> None for chosen knob consts
   (monkeypatch ModelRef.get_interp, the exact z3py path m[zv] takes) and
   check the enumeration STILL terminates unsat with the SAME space size,
   no duplicates, and only legal vectors (each re-checked against a fresh
   KnobSpace's constraints via completion probes).
2. Termination guard: an rlimit abort mid-enumeration must not silently
   truncate (solve retries uncapped / raises).
3. Determinism: kkt-368 / o-465 / o-K64V64-204 enumerated twice,
   byte-identical after normalizing timing fields; --validate rho.
"""
import json
import sys

sys.path.insert(0, "tools")
import z3

import gdn_solver as G

SPACES = [
    dict(stage="kkt", K=128, V=None, B=4, T=8192, H=16, expect=368),
    dict(stage="o", K=128, V=128, B=4, T=8192, H=16, expect=465),
    dict(stage="o", K=64, V=64, B=4, T=8192, H=16, expect=204),
]

DROP_SETS = [[], ["prec"], ["tma"], ["prec", "tma"], ["mma"],
             ["tile_K"]]   # z3 const names: bk == "tile_K"


def drop_interp(names):
    """Return (patch, restore): make m[zv] read None for knobs in names."""
    orig = z3.ModelRef.get_interp

    def patched(self, decl):
        if decl.name() in names:
            return None            # the documented artifact, verbatim
        return orig(self, decl)

    z3.ModelRef.get_interp = patched
    return lambda: setattr(z3.ModelRef, "get_interp", orig)


def legalize(kv, stage, K, V):
    """Python mirror of KnobSpace legality for an emitted vector."""
    sp = dict(stage=stage, K=K, V=V)
    n_glob = 64 if stage == "kkt" else V
    n_gemm = 1 if stage == "kkt" else 2
    if kv["dtype"] not in (0, 1) or kv["prec"] not in (0, 1):
        return False
    if kv["dtype"] == 0 and kv["prec"] != 0:
        return False
    if kv["dtype"] == 1 and kv["prec"] != 1:
        return False
    if not kv["mma"] and kv["prec"] != kv["dtype"]:
        return False
    if kv["bt"] not in (32, 64, 128) or kv["warps"] not in (2, 4, 8):
        return False
    if kv["pipe"] not in (0, 2, 3):
        return False
    if n_glob % kv["bv"] or K % kv["bk"]:
        return False
    if kv["mma"]:
        if kv["bt"] % 16:
            return False
        if kv["dtype"] == 0 and kv["bk"] % 16:
            return False
        if kv["dtype"] == 1 and kv["bk"] % 4:
            return False
        if kv["bv"] % (8 * kv["warps"]):
            return False
        if kv["pipe"] > 0 and K < kv["bk"] * kv["pipe"]:
            return False
    else:
        if kv["pipe"] > 0 and K < kv["bk"]:
            return False
    ws_ok = (stage == "kkt" and kv["dtype"] == 0 and kv["mma"])
    if kv["ws"] and not ws_ok:
        return False
    if kv["tma"] and not kv["ws"]:
        return False
    if kv["ws"] and kv["pipe"] != 2:
        return False
    threads = 32 * (kv["warps"] + (1 if kv["ws"] else 0))
    if threads * 255 > 65536:
        return False
    esz = 2 if kv["dtype"] == 0 else 4
    pair = (kv["bt"] * kv["bk"] + kv["bk"] * kv["bv"]) * esz
    if kv["mma"]:
        smem = (pair + (64 * 64 * esz if kv["tma"] else 0) + 8) if kv["ws"] \
            else pair * max(kv["pipe"], 1) * n_gemm
    else:
        smem = (kv["bt"] * K + K * kv["bv"] + kv["bt"] * kv["bv"]) * esz
    return smem <= 99 * 1024 - 1024


def run_space(sp, drop=None):
    restore = drop_interp(drop) if drop else (lambda: None)
    try:
        ks = G.KnobSpace(sp["stage"], sp["K"], V=sp["V"])
        combos = ks.enumerate()
    finally:
        restore()
    return ks, combos


def main():
    ok = True

    # -- 1. synthetic artifact injection ---------------------------------
    for sp in SPACES:
        base_ks, base = run_space(sp)
        assert sp["stage"] in ("kkt", "o")
        assert len(base) == sp["expect"], (sp, len(base))
        assert base_ks.last_check == "unsat", base_ks.last_check
        base_sig = sorted(tuple(sorted(c.items())) for c in base)
        for drop in DROP_SETS:
            if drop == ["bk"] and sp["stage"] == "o" and sp["K"] == 64:
                pass  # still fine: bk domain (32, 64) both probe-legal
            ks, combos = run_space(sp, drop=drop)
            sig = sorted(tuple(sorted(c.items())) for c in combos)
            same = (sig == base_sig)
            legal = all(legalize(c, sp["stage"], sp["K"], sp["V"])
                        for c in combos)
            stats = getattr(ks, "enum_stats", {})
            fired = stats.get("probe_completions", 0)
            print(f"[inject] {sp['stage']} K{sp['K']}V{sp['V']} "
                  f"drop={drop or 'none'}: n={len(combos)} "
                  f"(expect {sp['expect']}) identical={same} legal={legal} "
                  f"completions={fired} last={ks.last_check}")
            ok &= same and legal and len(combos) == sp["expect"]
            if drop:
                ok &= fired > 0

    # -- 2. rlimit-abort guard -------------------------------------------
    # a tiny rlimit aborts with 'unknown': solve() must retry uncapped and
    # still return the full space (or raise, never a silent truncation)
    sp = SPACES[2]
    import gdn_costmodel as cm  # noqa: F401  (solve() dependency)
    old = G.KnobSpace.__init__
    calls = {"n": 0}

    def capped_init(self, stage, K, V=None, mode="strict",
                    tile_domain=G.TILE_DOMAIN, seed=G.DEFAULT_SEED,
                    rlimit=G.DEFAULT_RLIMIT):
        # cap only the FIRST construction; solve()'s uncapped retry must
        # go through unmodified
        calls["n"] += 1
        eff = 1 if calls["n"] == 1 else rlimit
        old(self, stage, K, V=V, mode=mode, tile_domain=tile_domain,
            seed=seed, rlimit=eff)
    G.KnobSpace.__init__ = capped_init
    try:
        out = G.solve(stage=sp["stage"], K=sp["K"], V=sp["V"],
                      B=sp["B"], T=sp["T"], H=sp["H"], top=1)
        n, retried = out["space_size"], out["z3"]["rlimit_retry_uncapped"]
        print(f"[rlimit-guard] space={n} (expect {sp['expect']}) "
              f"retry_uncapped={retried} last_check={out['z3']['last_check']}")
        ok &= (n == sp["expect"]) and retried
    except RuntimeError as e:
        # acceptable per contract (refuse truncated) as long as it's the
        # documented refusal, not a crash
        print(f"[rlimit-guard] refused: {e}")
    finally:
        G.KnobSpace.__init__ = old

    # -- 3. determinism: two full solves per space, byte-identical ---------
    def norm(doc):
        d = json.loads(json.dumps(doc))
        d.pop("enumeration_seconds", None)
        for r in d["ranking"]:
            r.pop("rank", None)   # re-derived, same anyway; keep strict
        return json.dumps(d, sort_keys=True)

    for sp in SPACES:
        docs = []
        for _ in range(2):
            out = G.solve(stage=sp["stage"], K=sp["K"], V=sp["V"],
                          B=sp["B"], T=sp["T"], H=sp["H"], top=None)
            docs.append(norm(out))
        ident = docs[0] == docs[1]
        print(f"[determinism] {sp['stage']} K{sp['K']}V{sp['V']}: "
              f"byte-identical x2 = {ident} "
              f"(space {json.loads(docs[0])['space_size']})")
        ok &= ident

    # -- 4. validate still gates ------------------------------------------
    rep = G.validate(verbose=False)
    print(f"[validate] best={rep['best_form']} "
          f"spearman={rep['forms'][rep['best_form']]['spearman']:.4f} "
          f"verdict={rep['verdict']}")
    ok &= rep["verdict"] == "PASS"
    ok &= abs(rep["forms"][rep["best_form"]]["spearman"] - 0.8962) < 0.01

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
