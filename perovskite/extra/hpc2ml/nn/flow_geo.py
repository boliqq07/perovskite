import os
import shutil
import sys
import time
import warnings
from typing import Union, Tuple, List, Optional

import numpy as np
import torch
from sklearn import metrics
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from hpc2ml.utils.log_model import LogModule, HookGradientModule, make_dot_, AverageMeterTotal


def class_eval(prediction, target):
    """Classification."""
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------
    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


class ProcessOutLabel():
    def __init__(self, multi_loss=False, target_name=("energy", "forces", 'stress'),
                 process_label=None, process_out=None,
                 ):

        self.multi_loss = multi_loss

        self.target_name = target_name

        if target_name is None:
            target_name = ("energy", "forces", "stress")

        self.target_name = tuple(target_name) if isinstance(target_name, (tuple, list)) else (
            target_name,)

        if not self.multi_loss:
            self.target_name = [self.target_name[0], ]
            self.target_number = 1
        else:
            self.target_number = len(target_name)

        def func11(data):
            return tuple([data[i] for i in self.target_name])

        if process_label is None:
            self._process_label = func11
        else:
            self._process_label = process_label

        def func2(y_or_ys, data=None):
            if isinstance(y_or_ys, dict):
                return tuple([y_or_ys[i] for i in self.target_name])
            else:
                return y_or_ys

        if process_out is None:
            self._process_out = func2
        else:
            self._process_out = process_out

    def process_out(self, y_or_ys, data=None):
        y_or_ys = self._process_out(y_or_ys, data)

        if isinstance(y_or_ys, torch.Tensor):
            return y_or_ys

        if len(y_or_ys) < self.target_number:
            raise KeyError(f"The target number {self.target_number} is not"
                           f" consist with output {len(y_or_ys)}. "
                           f"check `multi_loss` and `target_name`")

        if len(y_or_ys) > self.target_number:
            warnings.warn(f"The target number {self.target_number} less than"
                          f" the output {len(y_or_ys)}. "
                          f"Please defined your loss_method, and neglect the redundant output-i."
                          f"Or add your `target_name`", UserWarning)
        return y_or_ys if len(y_or_ys) > 1 else y_or_ys[0]

    def process_label(self, data):
        label = self._process_label(data=data)

        if isinstance(label, torch.Tensor):
            return label

        return label if len(label) > 1 else label[0]


class LearningFlow:
    """
    LearningFlow for training.
    
    Examples:

        >>> test_dataset = dataset[:1000]
        >>> val_dataset = dataset[1000:2000]
        >>> train_dataset = dataset[2000:3000]
        >>> import torch_geometric.transforms as T
        >>> train_dataset = SimpleDataset(data_train, pre_transform=T.ToSparseTensor())
        >>> test_dataset = SimpleDataset(data_test,pre_transform=T.ToSparseTensor())
        >>> val_dataset = SimpleDataset(val_data,pre_transform=T.ToSparseTensor())

        >>> train_loader = DataLoader(
        ... dataset=train_dataset,
        ... batch_size=200,
        ... shuffle=False,
        ... num_workers=0)

        >>> test_loader = ...
        >>> val_loader = ...


        >>>  model = CrystalGraphConvNet(nfeat_node=91,
        ...  nfeat_edge=3,
        ...  nfeat_state=29,
        ...  nc_node_hidden=128,
        ...  nc_node_interaction=64,
        ...  num_interactions=2,)
        >>> # model = CrystalGraphGCN(...)
        >>> # model = CrystalGraphGCN2(...)
        >>> # model = CrystalGraphGAT(...)
        >>> # model = SchNet(...)
        >>> # model = MEGNet(...)
        >>> # model = SchNet(...)


        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        >>> scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=2,...min_lr=0.001)
        >>> device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

        >>> lf= LearningFlow(model, train_loader, test_loader=val_loader, device= "cuda:1",
        ... optimizer=None, clf= False, loss_method=None, learning_rate = 1e-3,
        ... weight_decay= 0.01, checkpoint=True, scheduler=scheduler,
        ... loss_threshold= 0.1, print_freq= None, print_what="all")

        >>> lf.run(50)

    where the dataset could from StructureToData or :class:`featurebox.test_featurizers.base_graph_geo.StructureGraphGEO`
    """

    def __init__(self, model: Module, train_loader: DataLoader, test_loader: DataLoader,
                 device: Union[str, object] = "cpu",
                 optimizer=None, clf: bool = False, loss_method=None, learning_rate: float = 1e-3,
                 weight_decay: float = 0.0, checkpoint=True, scheduler=None, debug: str = None,
                 target_layers: Union[str, List, Tuple] = "all", multi_loss=False,
                 target_name: Optional[Tuple] = ("energy", "forces",),
                 loss_threshold: float = 0.1, print_freq: Union[int, str, None] = 10, print_what="all",
                 process_label=None, process_out=None, note=None, store_filename='checkpoint.pth.tar'):
        """

        Parameters
        ----------
        model: module
        train_loader: DataLoader
        test_loader: DataLoader
        device:str
            such as "cuda:0","cpu"
        optimizer:torch.Optimizer
        clf:bool
            under exploit......
        loss_method:torch._Loss
            1. In default, for classification, loss_method = torch.nn.CrossEntropyLoss().
            And for claffication, torch.nn.MSELoss(), see more in torch
            2. If multi_loss, use function to deal with multi-output and return one single score. such has:
            >>> def mlm(ys_pred, ys_true):
            ...     mse = torch.potlayer.MSELoss()
            ...     ls = []
            ...     for i,j in zip(ys_pred, ys_true):
            ...           ls.append(mse(i,j))
            ...     return torch.mean(ls)
        multi_loss:bool
            for multi-optput metwork.
        target_name:tuple of str
            other names as target in data, except 'y'.
        learning_rate:float
            see more in torch
        weight_decay:float
            see more in torch
        checkpoint:bool
            save checkpoint or not.
        loss_threshold:
            see more in torch
        print_freq:int
            print frequency
        print_what:str
            "all","train","test" log.
        scheduler:
            scheduler, see more in torch
        process_label:Callable,
            function to get true y/label, mainly change shape. input: Data, output: List[Tensor].
            default:
            >>> def func11(data):
            ...    return tuple([data[i] for i in self.target_name])

        process_out:Callable
            function to get predict y, mainly change shape.
            input:(Union[Tensor,List[Tensor], Data), output: List[Tensor].
            default:
            >>> def func2(y_or_ys, data=None):
            >>> if isinstance(y_or_ys, dict):
            ...    return tuple([y_or_ys[i] for i in self.target_name])
            >>> else:
            ...    return y_or_ys
        store_filename:str
            filename to store.
        """

        if note is None:
            note = {}
        self.note = note

        self.train_loader = train_loader
        self.test_loader = test_loader

        device = torch.device(device)

        self.device = device
        self.model = model
        self.model.to(device)
        self.clf = clf
        self.loss_method = loss_method
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        self.store_filename = store_filename

        self.weight_log = LogModule(model=model, target_layer="weight")

        self.debug = "" if debug is None else debug
        if "hook" in self.debug or "loop" in self.debug:
            self.gradient_log = HookGradientModule(model=model, target_layer=target_layers)

        self.train_batch_number = len(self.train_loader)

        if print_freq is None:
            self.print_freq = self.train_batch_number
        else:
            self.print_freq = self.train_batch_number if not isinstance(print_freq, int) else print_freq

        if self.optimizer is None or self.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif self.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            # L2 regularization
        else:
            self.optimizer = optimizer

        self.multi_loss = multi_loss

        if self.loss_method is None:
            if self.clf is True:
                lm = torch.nn.CrossEntropyLoss()
            elif self.clf == "multi_label":
                lm = torch.nn.L1Loss()
                # 主要是用来判定实际的输出与期望的输出的接近程度 MAE: 1/N |y_pred-y| y 为多列
            else:
                lm = torch.nn.MSELoss()

            if not self.multi_loss:
                self.loss_method = lm
            else:
                def mlm(y_preds: tuple, batch_ys: tuple):
                    ls = []
                    for i, j in zip(y_preds, batch_ys):
                        ls.append(lm(i, j))
                    return sum(ls)

                warnings.warn("The default 'loss_method' for 'multi_loss' is just add all loss. which is simple, "
                              "we suggest use your own 'loss_method' !!!", UserWarning)
                self.loss_method = mlm

        else:
            self.loss_method = loss_method

        if scheduler is None:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=7, min_lr=0.00001)
        else:
            self.optimizer = scheduler.optimizer
            self.scheduler = scheduler
        self.best_error = 1000000.0
        self.threshold = loss_threshold
        # *.pth.tar or str
        self.run_train = self.run
        self.fit = self.run_train
        self.print_what = print_what

        if self.clf and process_label is None:
            warnings.warn("We suggest defined the process_label for classification problem.")

        self.pol = ProcessOutLabel(multi_loss=self.multi_loss, target_name=target_name,
                                   process_out=process_out, process_label=process_label)

        self.start_epoch = 0

        if self.clf:
            self.meters = AverageMeterTotal("Time", "losses", "accuracies", "precisions", "recalls", "fscores",
                                            "auc_scores")
        else:
            self.meters = AverageMeterTotal("Time", "losses", "mae_errors")

        self.single_print = True

    def my_debug(self):
        # debug

        if "loop" in self.debug:
            if "weight" in self.debug:
                self.weight_log.stats_loop()
            if "hook" in self.debug:
                self.gradient_log.stats_loop()

        if "weight" in self.debug:
            # just for 'single',
            self.model.train()
            for m, data in enumerate(self.train_loader):  # just record for once,
                data = data.to(self.device)
                self.model(data)
                self.weight_log.record(append=False)
                break
            self.weight_log.stats_single()

        if "graphviz" in self.debug:
            # just for 'single',
            self.model.train()
            for m, data in enumerate(self.train_loader):
                data = data.to(self.device)
                y_pred = self.model(data)
                vis_graph = make_dot_(y_pred, params=dict(list(self.model.named_parameters())))
                vis_graph.render(format="pdf", view=False)
                break

        if "hook" in self.debug:
            # hook just for 'single'
            # which layer to hook.
            self.model.train()
            for m, data in enumerate(self.train_loader):  # just record for once,
                data = data.to(self.device)
                batch_y = self.process_label(data=data)
                self.model.zero_grad()
                y_pred = self.model(data)
                y_pred = self.process_out(y_pred)
                lossi = self.loss_method(y_pred, batch_y)
                lossi.backward(retain_graph=True)
                # lossi.backward()
                self.gradient_log.record(append=False)
                break
            self.gradient_log.stats_single()

    def process_label(self, data=None):
        return self.pol.process_label(data=data)

    def process_out(self, y, data=None):
        return self.pol.process_out(y, data)

    def save_checkpoint(self, state, is_best, store_filename):
        if store_filename is not None:
            self.store_filename = store_filename
        torch.save(state, self.store_filename)
        if is_best:
            if self.store_filename != "model_best.pth.tar":
                shutil.copyfile(self.store_filename, 'model_best.pth.tar')

    def run(self, epoch=50, warm_start: bool = False, store_filename=None):
        """
        run loop.

        Parameters
        ----------
        epoch:int
            epoch.
        warm_start: bool
            warm start or not.
        store_filename: str
            The name of resume file, 'checkpoint.pth.tar' or 'model_best.pth.tar'
            If warm_start, try to resume from local disk.

        """
        if store_filename is not None:
            self.store_filename = store_filename
        else:
            self.store_filename = 'checkpoint.pth.tar'

        if warm_start:
            resume = self.store_filename
            if os.path.isfile(resume):
                print("=> loading checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume)
                self.start_epoch = checkpoint['epoch']
                self.best_error = checkpoint['best_error']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.note = checkpoint['note']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(resume))

        if self.start_epoch > 0:
            print("Try to run start from 'resumed epoch' {} to 'epoch' {}".format(self.start_epoch,
                                                                                  epoch + self.start_epoch))
            if epoch == 0:
                print("No new epoch added.")

        for epochi in range(self.start_epoch, epoch + self.start_epoch):

            self._train(epochi)

            if "loop" in self.debug:  # record each epochi
                self.weight_log.record()
                self.gradient_log.record()

            score = self._validate(epochi)

            if score != score:
                print('Exit due to NaN output value.')
                sys.exit(1)
            try:
                self.scheduler.step(score)
            except TypeError:
                self.scheduler.step()

            is_best = score < self.best_error
            self.best_error = min(score, self.best_error)

            if self.checkpoint:
                self.save_checkpoint({
                    'epoch': epochi + 1,
                    'state_dict': self.model.state_dict(),
                    'best_error': score,
                    'optimizer': self.optimizer.state_dict(),
                    'note': self.note
                }, is_best, store_filename=self.store_filename)

            if score <= self.threshold:
                print("Up to requirements and early termination in epoch ({})".format(epochi))
                break

        self.my_debug()
        self.start_epoch = self.start_epoch + epoch

    def _train(self, epochi):
        self.model.train()
        self.meters.reset()

        for m, data in enumerate(self.train_loader):

            point = time.time()
            data = data.to(self.device)
            batch_y = self.process_label(data=data)

            self.optimizer.zero_grad()

            y_pred = self.model(data)

            y_pred = self.process_out(y_pred, data)

            lossi = self.loss_method(y_pred, batch_y)

            lossi.backward()

            self.optimizer.step()

            del data

            # ###########

            self.print_rec(lossi, y_pred, batch_y, point)

            if self.print_freq != self.train_batch_number and m % self.print_freq == 0 \
                    and self.print_what in ["all", "train"]:
                print('Train: [{0}][{1}/{2}] {3}'.format(epochi, m, len(self.train_loader), self.meters.text()))

        if self.print_freq == self.train_batch_number:
            print('Train: [{}] {}'.format(epochi, self.meters.text()))

    def print_rec(self, lossi, y_pred, data_y, point):

        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        if isinstance(data_y, tuple):
            data_y = data_y[0]

        batch_size = data_y.size(0)

        lossi = float(lossi.__repr__()[7:].split(",")[0])  # more speed?
        self.meters['losses'].record(lossi, batch_size)

        if self.clf is False:
            mae_error = mae(y_pred, data_y)
            mae_error = float(mae_error.__repr__()[7:].split(",")[0])  # more speed?

            self.meters['mae_errors'].record(mae_error, batch_size)

        else:
            accuracy, precision, recall, fscore, auc_score = \
                class_eval(y_pred.detach().cpu(), data_y.cpu())
            self.meters['accuracies'].record(accuracy, batch_size)
            self.meters['precisions'].record(precision, batch_size)
            self.meters['recalls'].record(recall, batch_size)
            self.meters['fscores'].record(fscore, batch_size)
            self.meters['auc_scores'].record(auc_score, batch_size)

        time_range = time.time() - point
        self.meters['Time'].record(time_range)

    def _validate(self, epochi):
        self.model.eval()
        self.meters.reset()

        for m, data in enumerate(self.test_loader):
            point = time.time()

            data = data.to(self.device)
            batch_y = self.process_label(data=data)
            y_pred = self.model(data)
            y_pred = self.process_out(y_pred, data)

            lossi = self.loss_method(y_pred, batch_y)

            del data

            # ###########

            self.print_rec(lossi, y_pred, batch_y, point)

            if self.print_freq != self.train_batch_number and m % self.print_freq == 0 \
                    and self.print_what in ["all", "train"]:
                print('Test: [{0}][{1}/{2}] {3}'.format(epochi, m, len(self.train_loader), self.meters.text()))

        if self.print_freq == self.train_batch_number:
            print('Test: [{}] {}'.format(epochi, self.meters.text()))

        if self.clf is False:
            return self.meters['mae_errors'].avg
        else:
            return self.meters['auc_scores'].avg

    def score(self, predict_loader, scoring=None):
        """Return ."""
        y_pre, y_true = self.predict(predict_loader)
        if not scoring:
            scoring = self.loss_method
        return float(scoring(y_pre, y_true))

    def predict(self, predict_loader: DataLoader, return_y_true=False,
                add_hook=False, hook_layer=None, device='cpu'):
        """
        Just predict by model,and add one forward hook to get processing output.

        Parameters
        ----------
        predict_loader:DataLoader
            MGEDataLoader, the target_y could be ``None``.
        return_y_true:bool
            if return_y_true, return (y_preds, y_true)
        add_hook:bool
        hook_layer:
            one layer of model. could get layers by ``get_target_layer`` (the values).

        Returns
        -------
        y_pred:tensor
        y_true:tensor
            if return_y_true

        """

        handles = []
        if add_hook:
            try:
                self.forward_hook_list = []

                def for_hook(module, input, output):
                    self.forward_hook_list.append(output.detach().cpu())

                handles.append(hook_layer.register_forward_hook(for_hook))
            except BaseException as e:
                print(e)
                raise AttributeError("use ``hook_layer`` to defined the hook layer.")

        res = simple_predict(self.model, predict_loader, return_y_true=return_y_true, device=device,
                             process_out=self.pol.process_out, process_label=self.pol._process_label,
                             multi_loss=self.multi_loss)



        if add_hook:
            [i.remove() for i in handles]  # del

        return res


def cat(y_preds):
    if isinstance(y_preds[0], (tuple, list)):
        y_preds = list(zip(*y_preds))
        if len(y_preds) == 1:
            y_preds = [i[0] for i in y_preds]
        else:
            y_preds = [torch.cat(i, dim=0) for i in y_preds]
    else:
        if len(y_preds) == 1:
            y_preds = y_preds[0]
        else:
            y_preds = torch.cat(y_preds, dim=0)
    return y_preds


def simple_predict(model, predict_loader: DataLoader, return_y_true=False, device='cpu', process_out=None,
                   process_label=None, target_name=("energy", "forces", "stress"), multi_loss=False):
    """
    Just predict by model,and add one forward hook to get processing output.

    Parameters
    ----------
    model:torch.nn.Module
        network.
    predict_loader:DataLoader
        MGEDataLoader, the target_y could be ``None``.
    return_y_true:bool
        if return_y_true, return (y_preds, y_true)

    Returns
    -------
    y_pred:tensor
    y_true:tensor
        if return_y_true

    """

    pol = ProcessOutLabel(multi_loss=multi_loss, target_name=target_name,
                          process_label=process_label, process_out=process_out)

    model.eval()
    model.to(device)

    ############

    y_preds = []
    y_true = []
    for data in predict_loader:
        data = data.to(device)

        y_prei = model(data)
        if isinstance(y_prei, tuple):
            y_prei = [i.detach() for i in y_prei]
        else:
            y_prei = y_prei.detach()

        y_preds.append(pol.process_out(y_prei, data))
        if return_y_true:
            y_true.append(pol.process_label(data))

    y_preds = cat(y_preds)

    if return_y_true:
        y_true = cat(y_true)
        return y_preds, y_true
    else:
        return y_preds


def load_check_point(model, resume_file="checkpoint.pth.tar", optimizer=None, device="cpu"):
    """
    load check_point without flow, just for predict

    Parameters
    ----------
    model:
        network
    resume_file:
        'model_best.pth.tar' or 'checkpoint.pth.tar'
    optimizer:
        optimizer

    Returns
    -------
    model, optimizer, start_epoch, best_error

    """
    print("=> loading checkpoint '{}'".format(resume_file))
    checkpoint = torch.load(resume_file, map_location=torch.device(device))
    start_epoch = checkpoint['epoch']
    best_error = checkpoint['best_error']
    note = checkpoint['note']

    if model is not None:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = None

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        optimizer = None

    print("=> loaded checkpoint '{}' (epoch {}) with score {}"
          .format(resume_file, start_epoch, best_error))
    return model, optimizer, start_epoch, best_error, note
