# AGENT_PYTORCH_FIXES — Fix All Deprecated & Broken APIs

## Goal
Before any architectural migration, fix every concrete bug and deprecated call
in the existing source files. This agent works ON the original files in-place.
Every fix listed here has been verified by reading the actual source code.

---

## 1. tensorly API Breaking Changes

### 1a. `parafac()` return value changed

**Affects:** `mod_decomposer._cp_decomposition()` (lines 530-536) and
`decompositions.cp_decomposition_conv_layer()` (lines 228-231).

**Old API (tensorly < 0.5):** returned factors directly as a tuple:
```python
last, first, vertical, horizontal = parafac(X, rank=rank, init='svd')
```

**New API (tensorly ≥ 0.5):** returns `(weights, factors)` where `factors` is a list:
```python
_, factors = parafac(X, rank=rank, init='svd')
last, first, vertical, horizontal = factors
```

**Fix every occurrence.** The `_BN` variants already use the new API — use them as
reference. Specifically fix these blocks:

In `mod_decomposer._cp_decomposition`:
```python
# BROKEN — old API
if size >= 256:
    last, first, vertical, horizontal = parafac(X, rank=rank, init='random')
else:
    last, first, vertical, horizontal = parafac(X, rank=rank, init='svd')

# FIXED — new API
if size >= 256:
    _, factors = parafac(X, rank=rank, init='random')
else:
    _, factors = parafac(X, rank=rank, init='svd')
last, first, vertical, horizontal = factors
```

In `decompositions.cp_decomposition_conv_layer` (non-BN version, same pattern).

### 1b. `partial_tucker()` parameter name changed

**Affects:** `decompositions.tucker_decomposition_conv_layer` (line 414-415),
`decompositions.tucker_decomposition_conv_layer_BN` (line 471),
`decompositions.tucker_xavier` (line 525-526).

**Old API:** `rank=ranks` (singular)
**New API:** `ranks=ranks` (plural)

```python
# BROKEN
core, [last, first] = partial_tucker(layer.weight.data.numpy(),
                                     modes=[0, 1], rank=ranks, init='svd')
# FIXED
core, [last, first] = partial_tucker(layer.weight.data.cpu().numpy(),
                                     modes=[0, 1], ranks=ranks, init='svd')
```

`mod_decomposer._tucker_layer_decomposition` already uses `ranks=ranks` — correct.

---

## 2. CUDA Incompatibility — `.numpy()` on GPU Tensors

Any `.numpy()` call on a tensor that may be on GPU will raise:
`RuntimeError: Can't call numpy() on Tensor that requires grad or is on CUDA`

**Fix: add `.cpu()` before every `.numpy()` call on weight tensors.**

### All occurrences to fix:

**`decompositions.py`:**
```python
# Line 30, 56, 92, 112, 418 and others:
weights = layer.weight.data.numpy()           # BROKEN on GPU
weights = layer.weight.data.cpu().numpy()     # FIXED
```

**`mod_decomposer._choose_compression` (line 418):**
```python
weights = layer.weight.data.numpy()           # BROKEN
weights = layer.weight.data.cpu().numpy()     # FIXED
```

**`mod_decomposer.estimate_tucker_ranks` (line 56) and `estimate_cp_ranks` (line 83):**
```python
weights = layer.weight.data.numpy()           # BROKEN
weights = layer.weight.data.cpu().numpy()     # FIXED
```

**`pytorch_utils.set_layer_weights` (line 245):**
```python
if not(layer.weight.data.numpy().shape == tensor.shape):   # BROKEN
if not(layer.weight.data.cpu().numpy().shape == tensor.shape):  # FIXED
```

**`pytorch_utils.set_layer_bias` (line 217):**
```python
if not(layer.bias.numpy().shape == tensor.shape):     # BROKEN — missing .data too
if not(layer.bias.data.cpu().numpy().shape == tensor.shape):  # FIXED
```

---

## 3. Deprecated PyTorch APIs

### 3a. `torch.autograd.Variable` — deprecated since PyTorch 0.4

**Affects:** `pytorch_utils.to_var()`, `custom_models.py` imports.

```python
# BROKEN — Variable is a no-op since PyTorch 0.4
from torch.autograd import Variable
return Variable(x)

# FIXED — just return the tensor, move to device if needed
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x
```

Remove all `Variable(...)` wrapping throughout all files.

### 3b. `torch.nn.init.xavier_uniform` — renamed with underscore (in-place)

**Affects:** `custom_models.CPD_Zhang.xavier_weights` (line 149).

```python
torch.nn.init.xavier_uniform(m.weight)    # BROKEN — removed in PyTorch 2.x
torch.nn.init.xavier_uniform_(m.weight)   # FIXED — in-place version
```

### 3c. `model.train(True)` / `model.train(False)` — semantically misleading

**Affects:** `training_algo.py` throughout.

```python
model.train(True)   # works but use model.train() for clarity
model.train(False)  # MISLEADING — use model.eval()
```

Replace:
- `model.train(True)` → `model.train()`
- `model.train(False)` → `model.eval()`

### 3d. `scheduler.step()` called BEFORE `optimizer.step()` — deprecated warning

**Affects:** `training_algo.train_model_val` (line 59).

```python
# BROKEN order — raises UserWarning in PyTorch ≥ 1.4
for phase in [...]:
    if phase == 'train':
        scheduler.step()        # called before optimizer.step()
        ...
        optimizer.step()

# FIXED — scheduler.step() must come AFTER optimizer.step()
optimizer.step()
scheduler.step()   # at end of epoch, outside the batch loop
```

### 3e. `data_iter.next()` — Python 2 style

**Affects:** `pytorch_utils.get_train_valid_loader` (line 599).

```python
data_iter = iter(sample_loader)
images, labels = data_iter.next()    # BROKEN — Python 2 style
images, labels = next(data_iter)     # FIXED — Python 3
```

---

## 4. Logic Bugs

### 4a. Syntax error in `mod_decomposer.pytorch_cp_layer_decomposition_BN`

Lines 138-150 contain a multi-line unpacking without parentheses and references
undefined names `first`, `vertical`, `horizontal` from the returned list:

```python
# BROKEN — syntax error + wrong variable names
first_pointwise, separable_vertical,
separable_horizontal, last_pointwise = _cp_decomposition(layer, rank, offline, filename)

bn_first = nn.BatchNorm2d(first.shape[1])    # 'first' is undefined here
bn_vertical = nn.BatchNorm2d(vertical.shape[1])  # same
```

```python
# FIXED — use actual return variable names, add parentheses
(first_pointwise, separable_vertical,
 separable_horizontal, last_pointwise) = _cp_decomposition(layer, rank, offline, filename)

# BN sizes come from the layer objects, not from undefined vars:
bn_first = nn.BatchNorm2d(first_pointwise.out_channels)
bn_vertical = nn.BatchNorm2d(separable_vertical.out_channels)
bn_horizontal = nn.BatchNorm2d(separable_horizontal.out_channels)
bn_last = nn.BatchNorm2d(last_pointwise.out_channels)
```

### 4b. Infinite loop risk in `_choose_compression`

If VBMF returns very small ranks (e.g. `ranks = [1, 1]`), repeatedly halving
with integer division reaches `[0, 0]`, then the compression formula divides by
zero, causing an infinite loop or `ZeroDivisionError`.

Add a guard:
```python
while compression <= compression_factor:
    ranks[0] = ranks[0] // 2
    ranks[1] = ranks[1] // 2
    if ranks[0] < 1 or ranks[1] < 1:
        ranks[0] = max(1, ranks[0])
        ranks[1] = max(1, ranks[1])
        break  # cannot compress further
    compression = ...
```

### 4c. `add_bias` logic error in `cp_decomposition_conv_layer_BN` (line 359)

```python
# BROKEN — always evaluates to True
add_bias = True and layer.bias is not None or False and not layer.bias

# FIXED
add_bias = layer.bias is not None
```

---

## 5. Dead Code to Remove (do NOT migrate)

These exist in the source but should NOT be carried into the new library:

- `decompositions.SVD_weights`, `FC_SVD_compression`, `conv1x1_SVD_compression` — SVD support is on the roadmap (v0.2), do not migrate yet. Leave a `# TODO: v0.2 SVD support` comment.
- `decompositions.tucker_xavier`, `cp_xavier_conv_layer` — Xavier-init variants are superseded; the library will use a `use_xavier_init: bool` config flag instead of separate functions.
- `mod_decomposer.pytorch_tucker_decomposition_BN` — inside a comment block, broken code. Delete.
- `pytorch_utils.tensorboard_log` — TensorBoard logging is not part of the library scope.
- All `logger.log_compression(...)` calls — the new `CompressionResult` handles reporting.
- All `print(...)` debug statements → replace with `logging.getLogger(__name__).debug(...)`.

---

## Output Contract

After this agent completes, running:
```bash
python -c "from mod_decomposer import pytorch_tucker_layer_decomposition, pytorch_cp_layer_decomposition; print('OK')"
```
must not raise any import error or API deprecation warning.

A quick smoke test with a small Conv2d layer on CPU must produce a valid `nn.Sequential`.
