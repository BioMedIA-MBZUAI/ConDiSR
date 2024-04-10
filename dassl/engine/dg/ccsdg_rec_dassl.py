import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator

from pytorch_revgrad import RevGrad
from ccsdg.models.unet import UnetBlock, SaveFeatures
from ccsdg.models.resnet import resnet34, resnet18, resnet50, resnet101, resnet152

class SimpleNet(nn.Module):
    """A simple neural network composed of a CNN backbone
    and optionally a head such as mlp for classification.
    """

    def __init__(self, size=48, c=64):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.Conv2d(8, 3, 3, stride=1, padding=1),  # [batch, 24, 8, 8]
            # nn.Sigmoid(),
        )


    @property
    def fdim(self):
        return 256

    def forward(self, x):
        f = self.decoder(x)

        return f
    
class UNetCCSDG(nn.Module):
    def __init__(self, cfg, model_cfg, num_classes, **kwargs):
        super().__init__()
        self.backbone = build_backbone(
            model_cfg.BACKBONE.NAME,
            verbose=cfg.VERBOSE,
            pretrained=model_cfg.BACKBONE.PRETRAINED,
            **kwargs,
        )

        self.num_classes = num_classes

        self._fdim = self.backbone.out_features

        self.channel_prompt = nn.Parameter(torch.randn(2, 64, 1, 1))

    @property
    def fdim(self):
        return self._fdim

    def forward_first_layers(self, x, tau=0.1):
        x = self.backbone.forward_first(x)  # 8 64 256 256

        channel_prompt_onehot = torch.softmax(self.channel_prompt/tau, dim=0)
        f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
        f_style = x * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)
        return f_content, f_style
    
    def forward(self, x, tau=0.1):
        x = self.backbone.forward_first(x)  # 8 64 256 256

        channel_prompt_onehot = torch.softmax(self.channel_prompt/tau, dim=0)
        f_content = x * channel_prompt_onehot[0].view(1, *channel_prompt_onehot[0].shape)
        f_style = x * channel_prompt_onehot[1].view(1, *channel_prompt_onehot[1].shape)
        
        return self.backbone.forward_rn(f_content)

class Projector(nn.Module):
    def __init__(self, output_size=512, c=64, size=48):
        super(Projector, self).__init__()
        self.conv = nn.Conv2d(c, 8, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(8)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(8 * size * size, output_size)

    def forward(self, x_in):
        x = self.conv(x_in)
        x = self.bn(x)
        x = F.relu(x)
        # x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, dim=1)
        return x

class TrainerBase:
    """Base class for iterative trainer."""

    def __init__(self):
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        self._writer = None

    def register_model(self, name="model", model=None, optim=None, sched=None):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )

        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )

        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )

        assert name not in self._models, "Found duplicate model names"

        self._models[name] = model
        self._optims[name] = optim
        self._scheds[name] = sched

    def get_model_names(self, names=None):
        names_real = list(self._models.keys())
        if names is not None:
            names = tolist_if_not(names)
            for name in names:
                assert name in names_real
            return names
        else:
            return names_real

    def save_model(
        self, epoch, directory, is_best=False, val_result=None, model_name=""
    ):
        names = self.get_model_names()

        for name in names:
            model_dict = self._models[name].state_dict()

            optim_dict = None
            if self._optims[name] is not None:
                optim_dict = self._optims[name].state_dict()

            sched_dict = None
            if self._scheds[name] is not None:
                sched_dict = self._scheds[name].state_dict()

            save_checkpoint(
                {
                    "state_dict": model_dict,
                    "epoch": epoch + 1,
                    "optimizer": optim_dict,
                    "scheduler": sched_dict,
                    "val_result": val_result
                },
                osp.join(directory, name),
                is_best=is_best,
                model_name=model_name,
            )

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        file_missing = False

        for name in names:
            path = osp.join(directory, name)
            if not osp.exists(path):
                file_missing = True
                break

        if file_missing:
            print("No checkpoint found, train from scratch")
            return 0

        print(f"Found checkpoint at {directory} (will resume training)")

        for name in names:
            path = osp.join(directory, name)
            start_epoch = resume_from_checkpoint(
                path, self._models[name], self._optims[name],
                self._scheds[name]
            )

        return start_epoch

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            val_result = checkpoint["val_result"]
            print(
                f"Load {model_path} to {name} (epoch={epoch}, val_result={val_result:.1f})"
            )
            self._models[name].load_state_dict(state_dict)

    def set_model_mode(self, mode="train", names=None):
        names = self.get_model_names(names)

        for name in names:
            if mode == "train":
                self._models[name].train()
            elif mode in ["test", "eval"]:
                self._models[name].eval()
            else:
                raise KeyError

    def update_lr(self, names=None):
        names = self.get_model_names(names)

        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()

    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")

    def init_writer(self, log_dir):
        if self.__dict__.get("_writer") is None or self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)

    def close_writer(self):
        if self._writer is not None:
            self._writer.close()

    def write_scalar(self, tag, scalar_value, global_step=None):
        if self._writer is None:
            # Do nothing if writer is not initialized
            # Note that writer is only used when training is needed
            pass
        else:
            self._writer.add_scalar(tag, scalar_value, global_step)

    def train(self, start_epoch, max_epoch):
        """Generic training loops."""
        self.start_epoch = start_epoch
        self.max_epoch = max_epoch

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def run_epoch(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def parse_batch_train(self, batch):
        raise NotImplementedError

    def parse_batch_test(self, batch):
        raise NotImplementedError

    def forward_backward(self, batch):
        raise NotImplementedError

    def model_inference(self, input):
        raise NotImplementedError

    def model_zero_grad(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].zero_grad()

    def model_backward(self, loss):
        self.detect_anomaly(loss)
        loss.backward()

    def model_update(self, names=None):
        names = self.get_model_names(names)
        for name in names:
            if self._optims[name] is not None:
                self._optims[name].step()

    def model_backward_and_update(self, loss, names=None):
        self.model_zero_grad(names)
        self.model_backward(loss)
        self.model_update(names)


class SimpleTrainer(TrainerBase):
    """A simple trainer class implementing generic functions."""

    def __init__(self, cfg, dev):
        super().__init__()
        self.check_cfg(cfg)

        # if torch.cuda.is_available() and cfg.USE_CUDA:
        #     self.device = torch.device("cuda")
        # else:
        #     self.device = torch.device("cpu")
        self.device = torch.device(dev)

        # Save as attributes some frequently used variables
        self.start_epoch = self.epoch = 0
        self.max_epoch = cfg.OPTIM.MAX_EPOCH
        self.output_dir = cfg.OUTPUT_DIR

        self.cfg = cfg
        self.build_data_loader()
        self.build_model()
        self.evaluator = build_evaluator(cfg, lab2cname=self.lab2cname)
        self.best_result = -np.inf

    def check_cfg(self, cfg):
        """Check whether some variables are set correctly for
        the trainer (optional).

        For example, a trainer might require a particular sampler
        for training such as 'RandomDomainSampler', so it is good
        to do the checking:

        assert cfg.DATALOADER.SAMPLER_TRAIN == 'RandomDomainSampler'
        """
        pass

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def build_model(self):
        """Build and register model.

        The default builds a classification model along with its
        optimizer and scheduler.

        Custom trainers can re-implement this method if necessary.
        """
        cfg = self.cfg

        print("Building model")
        self.model = UNetCCSDG(cfg, cfg.MODEL, self.num_classes)
        self.model.to(self.device)

        self.projector = Projector(c=cfg.CCSDG.C, size=cfg.CCSDG.SIZE)
        self.projector.to(self.device)

        self.autoenc = SimpleNet(size=cfg.CCSDG.SIZE, c=cfg.CCSDG.C)
        self.autoenc.to(self.device)

        print(f"# params: {count_num_param(self.model):,}")
        
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.optim_proj = build_optimizer(self.projector, cfg.OPTIM)
        self.optim_autoenc = build_optimizer(self.autoenc, cfg.OPTIM)
        
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.sched_proj = build_lr_scheduler(self.optim_proj, cfg.OPTIM)
        self.sched_autoenc = build_lr_scheduler(self.optim_autoenc, cfg.OPTIM)
        
        self.register_model("model", self.model, self.optim, self.sched)
        self.register_model("projector", self.projector, self.optim_proj, self.sched_proj)
        self.register_model("autoenc", self.autoenc, self.optim_autoenc, self.sched_autoenc)

        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Detected {device_count} GPUs (use nn.DataParallel)")
        #     self.model = nn.DataParallel(self.model)

    def train(self):
        super().train(self.start_epoch, self.max_epoch)

    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_inference(self, input):
        return self.model(input)

    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)

        return input, label

    def get_current_lr(self, names=None):
        names = self.get_model_names(names)
        name = names[0]
        return self._optims[name].param_groups[0]["lr"]


class TrainerXU(SimpleTrainer):
    """A base trainer using both labeled and unlabeled data.

    In the context of domain adaptation, labeled and unlabeled data
    come from source and target domains respectively.

    When it comes to semi-supervised learning, all data comes from the
    same domain.
    """

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        train_loader_x_iter = iter(self.train_loader_x)
        train_loader_u_iter = iter(self.train_loader_u)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_x_iter)
            except StopIteration:
                train_loader_x_iter = iter(self.train_loader_x)
                batch_x = next(train_loader_x_iter)

            try:
                batch_u = next(train_loader_u_iter)
            except StopIteration:
                train_loader_u_iter = iter(self.train_loader_u)
                batch_u = next(train_loader_u_iter)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x, batch_u)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]

        input_x = input_x.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)

        return input_x, label_x, input_u


class TrainerX(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain

@TRAINER_REGISTRY.register()
class CCSDGRec(TrainerX):
    """CCSDG rec model.
    
    A.k.a. CCSDG rec
    """

    def forward_backward(self, batch):
        input, fda, gla, target = self.parse_batch_train(batch)

        f_content, f_style = self.model.forward_first_layers(input, tau=0.1)
        f_content_fda, f_style_fda = self.model.forward_first_layers(fda, tau=0.1)
        f_content_GLA, f_style_GLA = self.model.forward_first_layers(gla, tau=0.1)
        f_content_p = self.projector(f_content)
        f_style_p = self.projector(f_style)
        f_content_fda_p = self.projector(f_content_fda)
        f_style_fda_p = self.projector(f_style_fda)
        f_content_GLA_p = self.projector(f_content_GLA)
        f_style_GLA_p = self.projector(f_style_GLA)
        
        content_loss = F.l1_loss(f_content_p, f_content_fda_p, reduction='mean') + \
                        F.l1_loss(f_content_fda_p, f_content_p, reduction='mean') + \
                        F.l1_loss(f_content_p, f_content_GLA_p, reduction='mean') + \
                        F.l1_loss(f_content_GLA_p, f_content_p, reduction='mean') + \
                        F.l1_loss(f_content_GLA_p, f_content_fda_p, reduction='mean') + \
                        F.l1_loss(f_content_fda_p, f_content_GLA_p, reduction='mean')
        style_loss = F.l1_loss(f_style_p, f_style_fda_p, reduction='mean') + \
                        F.l1_loss(f_style_fda_p, f_style_p, reduction='mean') + \
                        F.l1_loss(f_style_p, f_style_GLA_p, reduction='mean') + \
                        F.l1_loss(f_style_GLA_p, f_style_p, reduction='mean') + \
                        F.l1_loss(f_style_GLA_p, f_style_fda_p, reduction='mean') + \
                        F.l1_loss(f_style_fda_p, f_style_GLA_p, reduction='mean')
        style_loss = - style_loss

        out_fda = self.model(fda, tau=0.1)
        out_gla = self.model(gla, tau=0.1)
        out_base = self.model(input, tau=0.1)

        low = self.autoenc(f_style)
        low_fda = self.autoenc(f_style_fda)
        low_GLA = self.autoenc(f_style_GLA)

        res = T.Resize(48, antialias=True)
        blur = T.GaussianBlur(kernel_size=(7,7), sigma=1.0)

        loss_fda = F.cross_entropy(out_fda, target)
        loss_gla = F.cross_entropy(out_gla, target)
        loss_base = F.cross_entropy(out_base, target)

        # loss_rec1 = F.mse_loss(low, blur(res(input)))
        # loss_rec2 = F.mse_loss(low_fda, blur(res(fda)))
        # loss_rec3 = F.mse_loss(low_GLA, blur(res(gla)))
        loss_rec1 = F.mse_loss(low, res(input))
        loss_rec2 = F.mse_loss(low_fda, res(fda))
        loss_rec3 = F.mse_loss(low_GLA, res(gla))

        loss_rec = loss_rec1 + loss_rec3 + loss_rec2

        loss = content_loss + style_loss + loss_fda + loss_gla + loss_base + loss_rec
        # loss = content_loss + loss_fda + loss_gla + loss_base + loss_rec
        
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_str" : content_loss.item(),
            "loss_sty" : style_loss.item(),
            "loss_class" : (loss_fda.item() + loss_gla.item() + loss_base.item()),
            "loss_rec" : loss_rec.item(),
            "acc": compute_accuracy(out_base, target)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["data"]
        fda = batch["fda_img"]
        gla = batch["GLA_img"]
        target = batch["label"]
        # high = batch["seg"]
        # dom = batch["domain"]

        
        input = input.to(self.device)
        fda = fda.to(self.device)
        gla = gla.to(self.device)
        target = target.to(self.device)
        # high = high.to(self.device)
        # dom = dom.to(self.device)

        return input, fda, gla, target

