# Fine-Tuning

Compression changes model weights and structure, so a short fine-tuning pass can
recover accuracy.

```python
from tensorpress import CompressConfig, Compressor
from tensorpress.config import FinetuneConfig

cfg = CompressConfig(
    method="tucker",
    finetune=True,
    finetune_config=FinetuneConfig(
        epochs=3,
        lr=1e-4,
        scheduler="cosine",
        use_amp=True,
    ),
)

result = Compressor(cfg).compress(
    model,
    dataloader={"train": train_loader, "val": val_loader},
)
```

`FineTuner` is pure PyTorch. It selects `cuda`, then `mps`, then `cpu`, uses
AdamW, supports `"cosine"`, `"step"`, and `"none"` schedulers, and enables AMP
only when requested on CUDA.

If `finetune=True`, `Compressor.compress()` requires a dataloader.
