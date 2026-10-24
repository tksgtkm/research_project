"""
学習中のパラメータ軌跡を 3D で観測する実験。

uv run python experiments/observe_train.py --opt sgd --epochs 300
uv run python experiments/observe_train.py --opt reversible --epochs 100 --reverse
"""
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import jax

import liouville2 as lv
import liouville2.functions as F
from liouville2 import DataLoader, optimizers
from liouville2.datasets import Spiral
from liouville2.models import MLP
from liouville2 import observe as ob

p = argparse.ArgumentParser()
p.add_argument('--opt', choices=['sgd', 'momentum', 'adam', 'reversible'], default='sgd')
p.add_argument('--epochs', type=int, default=300)
p.add_argument('--batch-size', type=int, default=30)
p.add_argument('--hidden', type=int, default=10)
p.add_argument('--lr', type=float, default=1.0)
p.add_argument('--every', type=int, default=10, help='何ステップごとに記録するか')
p.add_argument('--hessian-every', type=int, default=1, help='何回の記録ごとにヘッセ行列を計算するか')
p.add_argument('--color', default='loss', help='3D 軌跡の色に使う指標 (loss, lam_min, grad_norm, step)')
p.add_argument('--reverse', action='store_true', help='reversible: 逆向きの軌跡も記録して重ねる')
p.add_argument('--out', default='observe_out')
args = p.parse_args()

if args.opt == 'reversible':
    lv.enable_x64()

train_set = Spiral(train=True)
train_loader = DataLoader(train_set, args.batch_size)
model = MLP((args.hidden, 3))
params = model.init(jax.random.key(0), train_loader.data[0][:1])


def loss_fn(params, batch):
    x, t = batch
    return F.softmax_cross_entropy(model(params, x), t)


# 指標は訓練データ全体 (Spiral は 300 点) で測る
rec = ob.Recorder(loss_fn, train_loader.data, every=args.every, hessian_every=args.hessian_every)
steps_per_epoch = len(train_set) // args.batch_size

if args.opt == 'reversible':
    opt = optimizers.ReversibleSGD(loss_fn, lr=args.lr, gamma=(9, 10), batch_fn=DataLoader.gather)
    state = opt.init(params)
    for epoch in range(args.epochs):
        state = ob.run_recorded(opt, state, train_loader.indices(epoch), train_loader.data,
                                rec, step0=epoch * steps_per_epoch)
    rec_rev = None
    if args.reverse:
        rec_rev = ob.Recorder(loss_fn, train_loader.data, every=args.every, hessian_every=0, mlp=False)
        rec_rev(args.epochs * steps_per_epoch, opt.params(state), force=True)
        for epoch in reversed(range(args.epochs)):
            state = ob.reverse_recorded(opt, state, train_loader.indices(epoch), train_loader.data,
                                        rec_rev, step0=epoch * steps_per_epoch)
else:
    opt = {'sgd': lambda: optimizers.SGD(args.lr),
           'momentum': lambda: optimizers.MomentumSGD(args.lr),
           'adam': lambda: optimizers.Adam(args.lr)}[args.opt]()
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch):
        grads = jax.grad(loss_fn)(params, batch)
        return opt.update(params, grads, opt_state)

    it = 0
    rec(0, params)
    for epoch in range(args.epochs):
        for batch in train_loader:
            params, opt_state = step(params, opt_state, batch)
            it += 1
            rec(it, params)
    rec_rev = None

os.makedirs(args.out, exist_ok=True)
res = rec.result()
print(res.summary())

rev = None
if rec_rev is not None:
    s, P, m = rec_rev.arrays()
    order = s.argsort()
    rev = ob.Result(s[order], P[order], {k: v[order] for k, v in m.items()}, None, None, {})
    # forward と reverse の同じステップのパラメータが一致しているか
    fwd = dict(zip(res.steps.tolist(), res.P))
    diff = max(float(abs(fwd[k] - p).max()) for k, p in zip(rev.steps.tolist(), rev.P) if k in fwd)
    print(f'max |forward - reverse| over recorded steps: {diff:.3e}')

ob.plot3d(res, os.path.join(args.out, f'trajectory_{args.opt}.html'), color=args.color, reverse=rev)
ob.plot_metrics(res, os.path.join(args.out, f'metrics_{args.opt}.html'))
rec.save(os.path.join(args.out, f'record_{args.opt}.npz'))
print(f'wrote {args.out}/trajectory_{args.opt}.html, metrics_{args.opt}.html, record_{args.opt}.npz')