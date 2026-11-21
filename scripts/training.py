# Aql X-1 X-ray analysis
# trains the network, functional JAX training loop with optax

# code breakdown
# model.init() with a dummy input, Adam (optax, with L2 weight decay) and class-weighted softmax cross-entropy,
# a jitted train step using jax.value_and_grad, then epochs of shuffled minibatches, keeping whichever epoch had
# the lowest validation loss. trains one model per feature set ("full" and "shape", see fourier_features.py)

# libs
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

import config as C
from neural_network import SpectrumMLP

# training settings
EPOCHS = 200          # 200 epochs takes about 4 s on CPU
BATCH = 64            # windows per minibatch
LR = 3e-4             # Adam learning rate
WEIGHT_DECAY = 0.03   # L2 on the weights, picked on the validation set. without it val loss starts climbing after ~8 epochs (overfit)


# cross-entropy where each class gets its own weight, so the ~3 quiet windows per outburst window don't swamp it
def weighted_xent(logits, y, w):
    onehot = jax.nn.one_hot(y, 2)
    return jnp.sum(optax.softmax_cross_entropy(logits, onehot) * w[y]) / jnp.sum(w[y])


# trains one model on one feature set and saves the best epoch
def train_one(feat, d):
    X, y, split = d[f"X_{feat}"], d["y"], d["split"]
    tr, va = split == 0, split == 1
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6            # standardised using the training set only
    Xs = ((X - mu) / sd).astype(np.float32)
    Xtr, ytr, Xva, yva = Xs[tr], y[tr], Xs[va], y[va]
    counts = np.bincount(ytr, minlength=2)
    cw = jnp.asarray(len(ytr) / (2 * counts), dtype=jnp.float32)   # class weights so both classes count the same

    model = SpectrumMLP()
    key = jax.random.PRNGKey(C.SEED)
    key, k_init = jax.random.split(key)
    # init with a dummy batch of zeros
    params = model.init(k_init, jnp.zeros((1, X.shape[1]), jnp.float32))["params"]
    tx = optax.chain(optax.add_decayed_weights(WEIGHT_DECAY), optax.adam(LR))
    opt_state = tx.init(params)

    # loss with dropout on (training)
    def loss_fn(p, xb, yb, rng):
        logits = model.apply({"params": p}, xb, train=True, rngs={"dropout": rng})
        return weighted_xent(logits, yb, cw)

    # one training step, jitted, value_and_grad gives the loss and the gradients in one go
    @jax.jit
    def train_step(p, s, xb, yb, rng):
        loss, grads = jax.value_and_grad(loss_fn)(p, xb, yb, rng)
        updates, s = tx.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    # validation loss (dropout off)
    @jax.jit
    def eval_loss(p, xb, yb):
        return weighted_xent(model.apply({"params": p}, xb, train=False), yb, cw)

    # validation balanced accuracy
    @jax.jit
    def eval_bacc(p, xb, yb):
        pred = jnp.argmax(model.apply({"params": p}, xb, train=False), -1)
        pos, neg = (yb == 1), (yb == 0)                      # masks not boolean indexing, jit needs fixed array sizes
        return 0.5 * (jnp.sum(pos & (pred == 1)) / jnp.sum(pos) + jnp.sum(neg & (pred == 0)) / jnp.sum(neg))

    # shuffled minibatches every epoch, keeps a copy of the params whenever val loss hits a new low
    hist = {"train_loss": [], "val_loss": [], "val_bacc": []}
    best = (np.inf, None, -1)
    rng = np.random.default_rng(C.SEED)
    t_start = time.time()
    for ep in range(EPOCHS):
        perm = rng.permutation(len(ytr))
        losses = []
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            key, k = jax.random.split(key)
            params, opt_state, l = train_step(params, opt_state, Xtr[b], ytr[b], k)
            losses.append(float(l))
        vl = float(eval_loss(params, Xva, yva))
        hist["train_loss"].append(float(np.mean(losses)))
        hist["val_loss"].append(vl)
        hist["val_bacc"].append(float(eval_bacc(params, jnp.asarray(Xva), jnp.asarray(yva))))
        if vl < best[0]:
            best = (vl, jax.tree_util.tree_map(lambda a: a, params), ep)
    secs = time.time() - t_start

    # saves the best params, the normalisation and the loss history
    _, best_params, best_ep = best
    (C.MODELS / f"mlp_{feat}_params.msgpack").write_bytes(serialization.to_bytes(best_params))
    np.savez(C.MODELS / f"mlp_{feat}_norm.npz", mu=mu, sd=sd)
    hist.update({"best_epoch": best_ep, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
                 "class_weights": [float(c) for c in cw], "train_seconds": secs,
                 "n_params": int(sum(v.size for v in jax.tree_util.tree_leaves(params))),
                 "n_train": int(tr.sum()), "n_val": int(va.sum())})
    (C.MODELS / f"mlp_{feat}_history.json").write_text(json.dumps(hist))
    print(f"[{feat}] {EPOCHS} epochs in {secs:.1f} s | loss {hist['train_loss'][0]:.3f} -> "
          f"{hist['train_loss'][-1]:.3f} | best val loss {best[0]:.3f} at epoch {best_ep} "
          f"(val bal. acc {hist['val_bacc'][best_ep]:.3f}) | params {hist['n_params']}")


#                        main code                         #
if __name__ == "__main__":
    d = np.load(C.PROC / "dataset.npz")
    for feat in ("full", "shape"):
        train_one(feat, d)
