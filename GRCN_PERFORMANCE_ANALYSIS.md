# GRCN Performance Analysis

## Problem
GRCN takes **1575.34s per epoch** vs DRAGON's **228s per epoch** (~7x slower), despite having fewer parameters.

## Root Cause

### The Routing Bottleneck
GRCN uses a **capsule-like routing mechanism** in the CGCN class that iterates multiple times per forward pass:

**Location:** `src/models/grcn.py` Lines 176-183

```python
for i in range(self.num_routing):  # self.num_routing = 3 (from n_layers config)
    if self.use_checkpoint and self.training:
        update = checkpoint(self._routing_forward, preference, features, edge_index, use_reentrant=False)
    else:
        update = self._routing_forward(preference, features, edge_index)
    preference = preference + update
    if self.has_norm:
        preference = F.normalize(preference)
```

**What happens:**
1. Each routing iteration calls `_routing_forward()` which performs:
   - Concatenation of user preferences and item features
   - GATConv (attention-based graph convolution) on the **entire user-item graph**
   - Extraction and update of user preferences
   
2. With `n_layers: 3` in the config, this **runs 3 times per forward pass**

3. After routing, a final attention convolution is applied

4. This happens for **each modality** (visual and text)

### Why DRAGON is Faster

1. **No iterative routing** - DRAGON uses direct graph convolutions without loops
2. **Simpler aggregation** - Multi-modal aggregation happens once on item-item KNN graph
3. **No attention mechanisms** - Uses basic message passing (SAGEConv) instead of attention (GATConv)
4. **Better batching** - Processes user-item interactions more efficiently

## Computational Complexity Comparison

### GRCN per epoch:
```
For each batch:
  - ID graph convolution (2 layers)
  - Visual modality:
    * 3 routing iterations × attention convolution
    * Final attention convolution
  - Text modality:
    * 3 routing iterations × attention convolution
    * Final attention convolution
  - Confidence weight computation
  - ID + modal fusion
```

### DRAGON per epoch:
```
For each batch:
  - Visual modality: 1 graph convolution
  - Text modality: 1 graph convolution  
  - Item-item KNN propagation (n_mm_layers times)
  - User-user graph propagation
  - Weighted fusion
```

## Solutions

### Option 1: Reduce Routing Iterations (Recommended)
**Impact:** Fastest improvement with minimal code changes

Edit `src/configs/model/GRCN.yaml`:
```yaml
n_layers: 1  # Reduce from 3 to 1
```

**Expected speedup:** ~3x faster (bringing epoch time to ~525s)

**Trade-off:** May slightly reduce accuracy as routing refinement is limited

### Option 2: Reduce to 2 Iterations (Balanced)
```yaml
n_layers: 2  # Reduce from 3 to 2
```

**Expected speedup:** ~1.5x faster (~1050s per epoch)

**Trade-off:** Better accuracy retention than Option 1

### Option 3: Optimize Routing Implementation
**More complex, requires code changes:**

1. **Vectorize routing iterations** - Batch operations better
2. **Cache attention weights** - Reuse across iterations where possible
3. **Use sparse operations** - More efficient for large graphs
4. **Reduce attention heads** - If using multi-head attention

### Option 4: Try Alternative Aggregation
Consider replacing the routing mechanism with:
- Simple mean pooling of modalities
- Single-pass attention
- Learnable weighted combination (like DRAGON)

## Memory Optimizations Already Applied

GRCN **does benefit** from the memory optimizations:
- ✅ Gradient checkpointing enabled for routing iterations
- ✅ Mixed precision training (AMP)
- ✅ CPU-based feature storage

However, these optimizations **don't help with time** because:
- Checkpointing trades memory for computation (recomputes on backward)
- The routing iterations are fundamentally expensive operations
- Attention mechanisms are memory-efficient but computationally intensive

## Recommendation

**Start with Option 1 (`n_layers: 1`)** and evaluate the accuracy trade-off:

1. Change the config
2. Run experiments
3. Compare accuracy metrics with baseline
4. If accuracy drop is acceptable, keep it
5. If not, try Option 2 (`n_layers: 2`)

The routing mechanism is GRCN's key innovation, but **3 iterations may be overkill** for your dataset. The original paper likely used different dataset sizes where this made sense.

## Why Parameter Count Doesn't Tell the Whole Story

GRCN has fewer **learnable parameters** than DRAGON, but:
- **Computational graph depth matters** - 3 routing iterations = 3× forward passes
- **Attention mechanisms are expensive** - O(E) operations for graph attention
- **Graph size matters** - GRCN processes the full bipartite graph multiple times
- DRAGON's user-user and item-item graphs are sparser

**Analogy:** Having a small car (fewer parameters) doesn't matter if you have to drive the same route 3 times (routing iterations) vs driving it once (DRAGON).
