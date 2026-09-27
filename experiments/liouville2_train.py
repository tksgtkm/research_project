"""
uv run python experiments/liouville2_train.py --dataset spiral --opt sgd --lr 1.0 --epochs 300
uv run python experiments/liouville2_train.py --dataset mnist  --opt reversible --lr 1.0 --epochs 5 --verify
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

import liouville2 as lv
import liouville2.functions as F
from liouville2 import DataLoader, optimizers
from liouville2.datasets import MNIST, Spiral
from liouville2.models import MLP

p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=['spiral', 'mnist'], default='spiral')
p.add_argument('--opt', choices=['sgd', 'momentum', 'adam', 'reversible'], default='sgd')
p.add_argument('--epochs', type=int, default=300)
p.add_argument('--batch-size', type=int, default=30)
p.add_argument('--hidden', type=int, default=10)
p.add_argument('--lr', type=float, default=1.0)
p.add_argument('--verify', action='store_true', help='reversible: 学習後に逆向きに戻して初期値と一致するか確認')
args = p.parse_args()

if args.opt == 'reversible':
    lv.enable_x64()

Data = Spiral if args.dataset == 'spiral' else MNIST
train_set, test_set = Data(train=True), Data(train=False)
n_class = 3 if args.dataset == 'spiral' else 10
train_loader = DataLoader(train_set, args.batch_size)
test_data = test_set.arrays()

model = MLP((args.hidden, n_class))
params = model.init(jax.random.key(0), train_loader.data[0][:1])


def loss_fn(params, batch):
    x, t = batch
    return F.softmax_cross_entropy(model(params, x), t)


@jax.jit
def evaluate(params, data):
    x, t = data
    y = model(params, x)
    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


def report(epoch, params, sec):
    tr_loss, tr_acc = evaluate(params, train_loader.data)
    te_loss, te_acc = evaluate(params, test_data)
    print(f'epoch {epoch:4d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} '
          f'| test loss {te_loss:.4f} acc {te_acc:.4f} | {sec:.2f}s')


log_every = max(1, args.epochs // 10)

if args.opt == 'reversible':
    opt = optimizers.ReversibleSGD(loss_fn, lr=args.lr, gamma=(9, 10), batch_fn=DataLoader.gather)
    state0 = state = opt.init(params)
    t0 = time.time()
    for epoch in range(args.epochs):
        state = opt.run(state, train_loader.indices(epoch), train_loader.data)
        if (epoch + 1) % log_every == 0:
            report(epoch + 1, opt.params(state), time.time() - t0)

    T = args.epochs * (len(train_set) // args.batch_size)
    N = lv.tree_size(params)
    print(f'buffer: {opt.buffer_bits(state) / 8 / 1024:.1f} KiB '
          f'(theory T*N*log2(1/γ) = {opt.theoretical_bits(T, N) / 8 / 1024:.1f} KiB)')

    if args.verify:
        for epoch in reversed(range(args.epochs)):
            state = opt.reverse(state, train_loader.indices(epoch), train_loader.data)
        ok = all(bool(jnp.array_equal(a, b)) for a, b in zip(jax.tree.leaves(state0), jax.tree.leaves(state)))
        print('bit-exact round trip:', ok)
else:
    opt = {'sgd': lambda: optimizers.SGD(args.lr),
           'momentum': lambda: optimizers.MomentumSGD(args.lr),
           'adam': lambda: optimizers.Adam(args.lr)}[args.opt]()
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch):
        grads = jax.grad(loss_fn)(params, batch)
        return opt.update(params, grads, opt_state)

    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in train_loader:
            params, opt_state = step(params, opt_state, batch)
        if (epoch + 1) % log_every == 0:
            report(epoch + 1, params, time.time() - t0)