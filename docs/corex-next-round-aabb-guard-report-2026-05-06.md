# CoreX Next Round AABB Guard Report 2026-05-06

## Summary

This round switched validation to a lightweight performance scope and tested
guarded triangle-only AABB async variants.

The frame-count based guard candidates were rejected after review. They are
benchmark-specific and are not meaningful general optimizations, so the code was
restored to the simple AABB async mask control only.

## Lightweight Performance Results

`wrecking_ball150`, no SPD/phase/selected/matrix diagnostics:

- `MASK=0`: `62s`, PCG sum `62786`, PCG max `178`
- `MASK=8`: `51s`, PCG sum `47741`, PCG max `172`
- full async: `59s`, PCG sum `55779`, PCG max `184`

`wrecking_ball400`, no SPD/phase/selected/matrix diagnostics:

- `MASK=0`: `187s`, PCG sum `192785`, PCG max `181`
- `MASK=8`: `167s`, PCG sum `163375`, PCG max `301`
- `MASK=8`, `MAX_FRAME=390`: `171s`, PCG sum `168107`, PCG max `179`
- `MASK=8`, `SYNC_PERIOD=30`: `177s`, PCG sum `169146`, PCG max `169`
- `MASK=8`, `SYNC_PERIOD=65`: `174s`, PCG sum `169405`, PCG max `174`

Default guarded smoke run on `wrecking_ball150`, no explicit AABB environment:

- `61s`, PCG sum `55764`, PCG max `173`

## Decision

Unguarded triangle-only AABB async has real wall-time upside, but the lightweight
`wb400` run exposed a late PCG outlier at `frame=390`, newton iterations `2-3`
(`285` and `301` PCG iterations). The tested frame-count guards remove that outlier
while preserving some wall-time gain:

- Conservative default (`SYNC_PERIOD=30`): `187s -> 177s`, PCG max `181 -> 169`
- Faster opt-in (`SYNC_PERIOD=65`): `187s -> 174s`, PCG max `181 -> 174`
- Fixed-frame diagnostic (`MAX_FRAME=390`): `187s -> 171s`, PCG max `181 -> 179`

These guards are not promoted and are no longer present in code. Future attempts to
stabilize the `MASK=8` upside should use a principled runtime signal, not a hardcoded
frame or periodic frame count.

Artifacts are in `/tmp/corex_next_round_20260506/`.
