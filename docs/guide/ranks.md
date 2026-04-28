# Rank Estimation

Ranks control how many factor channels each decomposition uses.

## Automatic Ranks

```python
CompressConfig(ranks="auto")
```

Automatic mode uses VBMF on unfolded convolution weights.

## Compression Target

```python
CompressConfig(ranks=0.5)
```

Float ranks are treated as a compression target by the current backend logic.
Use this mode when you want one global target instead of per-layer manual ranks.

## Manual Ranks

```python
CompressConfig(
    ranks={
        "features.0": [8, 8],
        "features.3": [16, 16],
    }
)
```

Tucker accepts one or two rank values per layer. CPD uses the first value as the
CP rank.
