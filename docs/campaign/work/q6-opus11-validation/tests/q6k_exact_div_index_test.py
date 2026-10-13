#!/usr/bin/env python3
"""Index/coverage mirror of mul_mat_vec_q6_k.comp (baseline + GGML_VK_Q6K_EXACT_DIV).
Integer stand-ins for d/scales/y make comparisons exact: checks index mapping, full
coverage, sccache slot provenance, bounds, path selection and the (assumed) sum
reduction of the MIRROR. GLSL==mirror is by review; this proves nothing about GPU numerics."""
import random

QK = 256
M32 = 0xFFFFFFFF


def req(c, m):
    if not c:
        raise AssertionError(m)


def u16(buf, k):  # data_a_packed16[..].ql/qh[k], little-endian, bounds-checked
    req(0 <= k and 2 * k + 1 < len(buf), "OOB packed16 %d/%d" % (k, len(buf)))
    return buf[2 * k] | (buf[2 * k + 1] << 8)


def make_block(rng):
    return {"ql": bytes(rng.randrange(256) for _ in range(128)),
            "qh": bytes(rng.randrange(256) for _ in range(64)),
            "sc": [rng.randrange(-128, 128) for _ in range(16)],
            "d": rng.randrange(1, 9)}


def ref_dequant(b):  # ggml dequantize_row_q6_K layout -> (q[256], scale_index[256])
    ql, qh = b["ql"], b["qh"]
    q, s = [None] * QK, [None] * QK
    for h in range(2):
        base, lo, ho, so = 128 * h, 64 * h, 32 * h, 8 * h
        for l in range(32):
            q[base + l] = ((ql[lo + l] & 0xF) | ((qh[ho + l] & 3) << 4)) - 32
            q[base + l + 32] = ((ql[lo + l + 32] & 0xF) | (((qh[ho + l] >> 2) & 3) << 4)) - 32
            q[base + l + 64] = ((ql[lo + l] >> 4) | (((qh[ho + l] >> 4) & 3) << 4)) - 32
            q[base + l + 96] = ((ql[lo + l + 32] >> 4) | (((qh[ho + l] >> 6) & 3) << 4)) - 32
            for g in range(4):
                s[base + l + 32 * g] = so + l // 16 + 2 * g
    return q, s


def select_exact(ncols, bs):  # nbr_par_th == 0 && ncols % QUANT_K == 0
    return ncols % QK == 0 and (ncols // QK) % (bs // 16) == 0


def thread_quants(b, ql_off, qh_off):
    ql0 = u16(b["ql"], ql_off // 2) | (u16(b["ql"], ql_off // 2 + 1) << 16)
    ql32 = u16(b["ql"], ql_off // 2 + 16) | (u16(b["ql"], ql_off // 2 + 17) << 16)
    qhw = u16(b["qh"], qh_off // 2) | (u16(b["qh"], qh_off // 2 + 1) << 16)
    words = [(ql0 & 0x0F0F0F0F) | (((qhw & 0x03030303) << 4) & M32),
             (ql32 & 0x0F0F0F0F) | (((qhw & 0x0C0C0C0C) << 2) & M32),
             ((ql0 >> 4) & 0x0F0F0F0F) | (qhw & 0x30303030),
             ((ql32 >> 4) & 0x0F0F0F0F) | ((qhw & 0xC0C0C0C0) >> 2)]
    return [[((w >> (8 * l)) & 0xFF) - 32 for l in range(4)] for w in words]


def run(A, Y, ncols, nrows, bs, nr, experimental, bug=False):
    req(bs % 16 == 0 and bs >= 16, "BLOCK_SIZE must be a multiple of 16")
    nb, it = ncols // QK, bs // 16
    exact = experimental and select_exact(ncols, bs)
    refs = [[ref_dequant(blk) for blk in r] for r in A]
    out = [[None] * nrows for _ in Y]
    touch, barriers = {}, 0
    for wg in range(-(-nrows // nr) + 1):  # +1 workgroup exercises early return
        first = nr * wg
        if first >= nrows:
            continue
        num_rows = min(nr, nrows - first)
        temp = [[[0] * num_rows for _ in Y] for _ in range(bs)]
        if exact:
            passes = [(i0, "direct") for i0 in range(0, nb, it)]
        else:
            full = nb - nb % it
            passes = [(i0, "all") for i0 in range(0, full, it)] + [(full, "tail")]
        for i0, kind in passes:
            for n in range(num_rows):
                row, lds = first + n, {}
                if kind != "direct":  # sccache write phase, then barrier()
                    for tid in range(bs):
                        ix, itid = divmod(tid, 16)
                        i = i0 + ix
                        if kind == "all" or i < nb:
                            req(i < nb, "OOB block in all_threads pass")
                            lds[(ix, itid)] = (row, i, A[row][i]["sc"][itid])
                    barriers += 1
                for tid in range(bs):
                    ix, itid = divmod(tid, 16)
                    i = i0 + ix
                    if kind == "tail" and i >= nb:
                        continue
                    req(i < nb, "OOB block %d in %s pass" % (i, kind))
                    v_im, v_in = divmod(itid, 8)
                    l0 = 4 * v_in
                    ql_off, qh_off = 64 * v_im + l0, 32 * v_im + l0
                    s_off, y_off = 8 * v_im + v_in // 4, 128 * v_im + l0
                    b = A[row][i]
                    qs = thread_quants(b, ql_off, qh_off)
                    sidx = [s_off + (g if bug else 2 * g) for g in range(4)]
                    req(max(sidx) < 16, "OOB scale index")
                    if kind == "direct":
                        scs = [b["sc"][k] for k in sidx]
                    else:
                        scs = []
                        for k in sidx:
                            src = lds.get((ix, k))
                            req(src is not None and src[:2] == (row, i), "sccache slot not written")
                            scs.append(src[2])
                    rq, rs = refs[row][i]
                    for j, y in enumerate(Y):
                        acc = 0
                        for g in range(4):
                            sg = 0
                            for l in range(4):
                                e = y_off + 32 * g + l
                                col = i * QK + e
                                req(0 <= col < len(y), "OOB B index")
                                req(qs[g][l] == rq[e] and sidx[g] == rs[e],
                                    "row %d blk %d elem %d q/scale mismatch" % (row, i, e))
                                if j == 0:
                                    touch[(row, i, e)] = touch.get((row, i, e), 0) + 1
                                sg += y[col] * qs[g][l]
                            acc += sg * scs[g]
                        temp[tid][j][n] += acc * b["d"]
        for j in range(len(Y)):  # reduce_result ASSUMED = sum over all BLOCK_SIZE threads
            for n in range(num_rows):
                req(out[j][first + n] is None, "row written twice")
                out[j][first + n] = sum(temp[t][j][n] for t in range(bs))
    return out, touch, {"exact": exact, "barriers": barriers}


def ref_out(A, Y, nb):
    res = []
    for y in Y:
        col = []
        for r in A:
            tot = 0
            for i in range(nb):
                q, s = ref_dequant(r[i])
                tot += r[i]["d"] * sum(r[i]["sc"][s[e]] * q[e] * y[i * QK + e] for e in range(QK))
            col.append(tot)
        res.append(col)
    return res


def check(ncols, nrows, bs, nr, experimental, expect_exact, bug=False):
    rng = random.Random(ncols * 31 + bs * 7 + nr * 3 + int(experimental))
    nb = ncols // QK
    A = [[make_block(rng) for _ in range(nb)] for _ in range(nrows)]
    Y = [[rng.randrange(-64, 65) for _ in range(ncols)] for _ in range(2)]  # NUM_COLS=2
    out, touch, st = run(A, Y, ncols, nrows, bs, nr, experimental, bug)
    req(st["exact"] == expect_exact, "path selection %r" % ((ncols, bs, st),))
    req(not st["exact"] or st["barriers"] == 0, "direct path must have 0 barriers")
    req(len(touch) == nrows * nb * QK and set(touch.values()) == {1}, "coverage")
    req(out == ref_out(A, Y, nb), "output mismatch")


def main():
    n = 0
    for ncols in (2048, 6144):  # 8 and 24 superblocks
        for bs in (16, 32, 64, 128):  # 64 = assumed RADV wave64 workgroup
            for nr in ((1, 2, 4) if bs == 64 else (2,)):  # 5 rows -> partial last workgroup
                for exp in (False, True):
                    check(ncols, 5, bs, nr, exp, expect_exact=exp)
                    n += 1
        check(ncols, 3, 256, 2, True, expect_exact=False)  # it_size 16 divides neither 8 nor 24
        n += 1
    for ncols, ex in ((256, False), (1024, True), (1280, False), (1536, False), (2304, False)):
        check(ncols, 3, 64, 2, True, expect_exact=ex)
        n += 1
    req(not select_exact(2000, 64) and not select_exact(2176, 64), "unaligned must fall back")
    try:
        check(2048, 2, 64, 2, True, True, bug=True)
    except AssertionError:
        pass
    else:
        raise SystemExit("FAIL: negative control (wrong scale stride) not detected")
    print("OK q6k exact-div mirror: %d configurations + negative control" % n)


if __name__ == "__main__":
    main()
