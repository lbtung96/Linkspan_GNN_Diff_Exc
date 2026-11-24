"""Several excitation records per training structure, one drawn per structure per epoch.

Every training structure has K records (K "pools": folders holding the same structures under different
excitation records, SeedSequence([base_k, sample_index])).  Each epoch, each structure is presented with
one of its K records:

    slots = numpy.random.default_rng([EPOCH_SEED, epoch]).integers(0, K, n_train)

a Generator of its own, so the draw neither consumes nor depends on torch's or numpy's global RNG (the
shuffle, dropout and initialisation streams are exactly those of a one-record run), and it is a function of
the epoch alone, so a resumed run draws what an uninterrupted one would.  Every draw is appended to the run's
variant_draws.csv.  Only the excitation-dependent inputs may differ between pools; VariantSet checks that
edges, node labels, node ids and frequency labels are identical.

Archive: k_train.install_augmentation (EC T13), which substituted these two classes for the library's
dataset and dgl's GraphDataLoader in memory.  The code below is that code, unchanged in arithmetic and in
the order of every draw; fr_train uses it directly when given record pools.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from dgl.dataloading import GraphDataLoader

EPOCH_SEED = 20261130  # seed registry: the per-epoch record draw (EC T13 onwards)


class VariantSet(torch.utils.data.Dataset):
    """K library datasets of the same structures (one per record pool); item i comes from pool slots[i]."""

    def __init__(self, sets: list, *, draws_path: Path, start_epoch: int = 0, epoch_seed: int = EPOCH_SEED):
        self.sets = sets
        base = sets[0]
        self.labels, self.graphs = base.labels, base.graphs
        self.node_feat_dim, self.edge_feat_dim = base.node_feat_dim, base.edge_feat_dim
        for s in sets[1:]:
            if not torch.equal(s.labels, base.labels):
                raise RuntimeError("frequency labels differ between record folders")
            for ga, gb in zip(base.graphs, s.graphs):
                ea, eb = ga.edges(), gb.edges()
                if not (torch.equal(ea[0], eb[0]) and torch.equal(ea[1], eb[1])
                        and torch.equal(ga.ndata["y"], gb.ndata["y"])
                        and torch.equal(ga.ndata["graph_node_id"], gb.ndata["graph_node_id"])):
                    raise RuntimeError("graphs / node labels differ between record folders")
        self.slots = np.zeros(len(base), dtype=np.int64)
        self.epoch = int(start_epoch)
        self.epoch_seed = int(epoch_seed)
        self.draws_path = Path(draws_path)

    def set_epoch(self) -> None:
        kk = len(self.sets)
        if kk == 1:
            self.slots = np.zeros(len(self.sets[0]), dtype=np.int64)
        else:
            self.slots = np.random.default_rng([self.epoch_seed, self.epoch]).integers(0, kk, size=len(self.sets[0]))
        new = not self.draws_path.exists()
        with self.draws_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["epoch"] + [f"n_k{k}" for k in range(kk)] + ["slots_head"])
            w.writerow([self.epoch] + [int(np.sum(self.slots == k)) for k in range(kk)]
                       + ["".join(str(int(s)) for s in self.slots[:24])])
        self.epoch += 1

    def __getitem__(self, idx):
        return self.sets[int(self.slots[idx])][idx]

    def __len__(self):
        return len(self.sets[0])


class EpochLoader(GraphDataLoader):
    """dgl's GraphDataLoader; starting an iteration (one per epoch) first draws that epoch's records."""

    def __iter__(self):
        if isinstance(self.dataset, VariantSet):
            self.dataset.set_epoch()
        return super().__iter__()


def build_variant_set(dataset_cls, train_paths: list[Path], pools: list[Path], *, run_dir: Path, start_epoch: int,
                      epoch_seed: int = EPOCH_SEED, **dataset_kw) -> VariantSet:
    """pools[0] holds train_paths; pools[1:] hold the same file names under their own records."""
    paths = list(train_paths)
    sets = [dataset_cls(paths, **dataset_kw)]
    for pool in list(pools)[1:]:
        sets.append(dataset_cls([Path(pool) / Path(p).name for p in paths], **dataset_kw))
    return VariantSet(sets, draws_path=Path(run_dir) / "variant_draws.csv", start_epoch=start_epoch,
                      epoch_seed=epoch_seed)
