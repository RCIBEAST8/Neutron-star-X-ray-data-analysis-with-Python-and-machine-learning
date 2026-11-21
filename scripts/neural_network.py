# Aql X-1 X-ray analysis
# the neural network (Flax Linen) that classifies a power spectrum

# code breakdown
# a small MLP is plenty for a 64/65 bin spectrum. the forward pass returns logits and softmax gets applied in
# predict_proba(). training uses optax.softmax_cross_entropy on the logits directly, because that's numerically
# safer than taking log(softmax) by hand

# libs
import flax.linen as nn
import jax
import jax.numpy as jnp


# input spectrum -> Dense(128) -> ReLU -> Dropout -> Dense(64) -> ReLU -> Dense(2)
class SpectrumMLP(nn.Module):
    hidden: tuple = (128, 64)     # neurons in each hidden layer
    n_classes: int = 2            # quiet or outburst
    dropout: float = 0.5          # 50% dropout, picked on the validation set

    @nn.compact
    def __call__(self, x, train: bool = False):
        for i, h in enumerate(self.hidden):
            x = nn.Dense(h, name=f"dense_{i}")(x)
            x = nn.relu(x)
            if i == 0:                # dropout only after the first layer
                x = nn.Dropout(self.dropout, deterministic=not train)(x)
        return nn.Dense(self.n_classes, name="logits")(x)


# softmax on the output gives the class probabilities [P(quiet), P(outburst)]
def predict_proba(model, params, x):
    return jax.nn.softmax(model.apply({"params": params}, x, train=False), axis=-1)


#                        main code                         #
# quick check on its own, prints the layer table and the parameter count
if __name__ == "__main__":
    m = SpectrumMLP()
    p = m.init(jax.random.PRNGKey(0), jnp.zeros((1, 64)))["params"]
    print(m.tabulate(jax.random.PRNGKey(0), jnp.zeros((1, 64))))
    print("params:", sum(v.size for v in jax.tree_util.tree_leaves(p)))
