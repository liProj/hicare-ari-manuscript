"""TabM (parameter-efficient MLP ensemble, Gorishniy et al., ICLR 2025) for binary outcomes.

k implicit members (16 here; the shared GPU makes 32 too slow at this sample size) share most weights; each member is trained on the same batch with its
own loss, and the prediction is the mean member probability.  Numeric features use
piecewise-linear embeddings, categoricals are one-hot inside the model.
"""
import numpy as np
import torch
import torch.nn.functional as Fn
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from tabm import TabM


def _predict(model, Xn, Xc, dev, bs=32768):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, len(Xn), bs):
            lg = model(torch.from_numpy(Xn[s:s + bs]).to(dev),
                       torch.from_numpy(Xc[s:s + bs]).to(dev)).float().squeeze(-1)   # (B, k)
            out.append(torch.sigmoid(lg).mean(1).cpu().numpy())
    return np.concatenate(out)


def train_predict(Xn_tr, Xc_tr, y_tr, Xn_va, Xc_va, y_va, Xn_te, Xc_te, cards, seed,
                  max_epochs=12, patience=2, bs=8192, lr=3e-3, wd=3e-4, d_block=256,
                  n_blocks=2, dropout=0.1, k=16, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda")
    sub = np.random.RandomState(seed).choice(len(Xn_tr), min(len(Xn_tr), 200000), replace=False)
    bins = compute_bins(torch.from_numpy(Xn_tr[sub]), n_bins=48)
    model = TabM.make(
        n_num_features=Xn_tr.shape[1], cat_cardinalities=cards, d_out=1,
        num_embeddings=PiecewiseLinearEmbeddings(bins, d_embedding=16, activation=False,
                                                 version="B"),
        d_block=d_block, n_blocks=n_blocks, dropout=dropout, k=k).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    xn = torch.from_numpy(Xn_tr).to(dev)
    xc = torch.from_numpy(Xc_tr).to(dev)
    yt = torch.from_numpy(y_tr).to(dev)
    n = len(xn)
    steps = max_epochs * (n // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    best, best_state, bad = np.inf, None, 0
    g = torch.Generator(device=dev).manual_seed(seed)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=dev, generator=g)
        for s in range(0, n - bs + 1, bs):
            idx = perm[s:s + bs]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(xn[idx], xc[idx]).float().squeeze(-1)             # (B, k)
            loss = Fn.binary_cross_entropy_with_logits(
                lg, yt[idx, None].expand_as(lg))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        pv = np.clip(_predict(model, Xn_va, Xc_va, dev), 1e-7, 1 - 1e-7)
        ll = float(-np.mean(y_va * np.log(pv) + (1 - y_va) * np.log(1 - pv)))
        if verbose:
            print(f"  tabm epoch {ep} val logloss {ll:.5f}", flush=True)
        if ll < best - 1e-5:
            best, bad = ll, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    p = _predict(model, Xn_te, Xc_te, dev)
    pv = _predict(model, Xn_va, Xc_va, dev)
    del model, xn, xc, yt, opt
    torch.cuda.empty_cache()
    return p, pv
