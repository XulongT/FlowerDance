import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dance_dataset import AISTPPDataset
from dataset.preprocess import increment_path
from model.adan import Adan
# from model.rectifiedflow import RectifiedFlow
from model.flowmatching import FlowMatching
from model.model import DanceDecoder
from vis import SMPLSkeleton
from einops import rearrange
import numpy as np
from model.metric import Metric
import random, numpy

def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class EDGE:
    def __init__(
        self,
        feature_type="baseline",
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
    ):

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        set_seed(42)
        state = AcceleratorState()
        num_processes = state.num_processes
        use_baseline_feats = feature_type == "baseline"

        pos_dim = 3
        rot_dim = 24 * 6  # 24 joints, 6dof
        self.repr_dim = repr_dim = pos_dim + rot_dim + 4

        feature_dim = 35 if use_baseline_feats else 4800

        horizon_seconds = 40
        FPS = 30
        self.horizon = horizon = horizon_seconds * FPS

        self.accelerator.wait_for_everyone()

        checkpoint = None
        if checkpoint_path != "":
            checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)
            self.normalizer = checkpoint["normalizer"]

        model = DanceDecoder(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )

        smpl = SMPLSkeleton(self.accelerator.device)
        flow_matching = FlowMatching(
            model,
            horizon,
            repr_dim,
            smpl,
            sigma=0.1,        # 保留噪声扰动
            loss_type="l2"
        )

        print("Model has {} parameters".format(sum(y.numel() for y in model.parameters())))

        self.model = self.accelerator.prepare(model)
        self.flow_matching = flow_matching.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            self.model.load_state_dict(maybe_wrap(checkpoint["model_state_dict"], num_processes,))

    def eval(self):
        self.flow_matching.eval()

    def train(self):
        self.flow_matching.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def test_loop(self, opt):
        # load datasets
        train_tensor_dataset_path = os.path.join(opt.processed_data_dir, f"train_tensor_dataset.pkl")
        train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))

        test_tensor_dataset_path = os.path.join(opt.processed_data_dir, f"test_tensor_dataset.pkl")
        if (not opt.no_cache and os.path.isfile(train_tensor_dataset_path) and os.path.isfile(test_tensor_dataset_path)):
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            test_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                cond_normalizer=train_dataset.cond_normalizer,
                force_reload=opt.force_reload,
            )
            print(f"Saving test dataset to: {test_tensor_dataset_path}")
            pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))
        self.normalizer = test_dataset.normalizer

        # data loaders
        test_data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False, worker_init_fn=self.worker_init_fn)
        save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
        opt.exp_name = save_dir.split("/")[-1]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gt_dir = './data/test'
        print('Build metric')
        metric = Metric(gt_dir, device)

        self.eval()
        print("Generating Sample")
        (x, cond, filename, wavnames, genre) = next(iter(test_data_loader))
        cond, genre, epoch = cond.to(self.accelerator.device), genre.to(self.accelerator.device), 0
        print(os.path.join(opt.eval_dir, "train_" + opt.exp_name))
        self.flow_matching.render_sample(x.shape, cond, genre, self.normalizer, epoch, os.path.join(opt.render_dir, "train_" + opt.exp_name), os.path.join(opt.eval_dir, "train_" + opt.exp_name), name=wavnames, sound=True,)
        metric.calculate_metric(os.path.join(opt.eval_dir, "train_" + opt.exp_name, str(epoch)), device)

    def inpaint_loop(self, opt):
        # load datasets
        train_tensor_dataset_path = os.path.join(opt.processed_data_dir, f"train_tensor_dataset.pkl")
        train_dataset = pickle.load(open(train_tensor_dataset_path, "rb"))

        test_tensor_dataset_path = os.path.join(opt.processed_data_dir, f"test_tensor_dataset.pkl")
        if (not opt.no_cache and os.path.isfile(train_tensor_dataset_path) and os.path.isfile(test_tensor_dataset_path)):
            test_dataset = pickle.load(open(test_tensor_dataset_path, "rb"))
        else:
            test_dataset = AISTPPDataset(
                data_path=opt.data_path,
                backup_path=opt.processed_data_dir,
                train=False,
                normalizer=train_dataset.normalizer,
                cond_normalizer=train_dataset.cond_normalizer,
                force_reload=opt.force_reload,
            )
            print(f"Saving test dataset to: {test_tensor_dataset_path}")
            pickle.dump(test_dataset, open(test_tensor_dataset_path, "wb"))
        self.normalizer = test_dataset.normalizer

        # data loaders
        test_data_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False, worker_init_fn=self.worker_init_fn)
        save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
        opt.exp_name = save_dir.split("/")[-1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eval()
        print("Generating Sample")
        (x, cond, filename, wavnames, genre) = next(iter(test_data_loader))
        x, cond, genre, epoch = x.to(self.accelerator.device), cond.to(self.accelerator.device), genre.to(self.accelerator.device), 0
        print(os.path.join(opt.eval_dir, "train_" + opt.exp_name))
        self.flow_matching.inpaint_sample(x, cond, genre, self.normalizer, epoch, os.path.join(opt.render_dir, "train_" + opt.exp_name), os.path.join(opt.eval_dir, "train_" + opt.exp_name), name=wavnames, sound=True,)

    def worker_init_fn(self, worker_id):
        seed = 42 + worker_id
        numpy.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
