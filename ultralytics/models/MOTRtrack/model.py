# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
RT-DETR model interface
"""

from pathlib import Path

import torch.nn as nn

from ultralytics.cfg import get_cfg
from ultralytics.engine.exporter import Exporter
from ultralytics.nn.tasks import TrackingModel, attempt_load_one_weight, yaml_model_load
from ultralytics.utils import DEFAULT_CFG, DEFAULT_CFG_DICT, LOGGER, RANK, ROOT, is_git_dir
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.torch_utils import model_info, smart_inference_mode

from .predict import TrackPredictor
from .train import TrackTrainer
from .val import TrackValidator


class MOTRTrack:

    def __init__(self, model='rtdetr-l.pt', pretrain_rtdetr_model=None) -> None:
        if pretrain_rtdetr_model is None:
            if model and not model.endswith('.pt') and not model.endswith('.yaml'):
                raise NotImplementedError('This model only supports creating from pt file or yaml file.')
            # Load or create new YOLO model
            self.predictor = None
            self.ckpt = None
            self.task = 'track'
            suffix = Path(model).suffix
            if suffix == '.yaml':
                self._new(model)
            else:
                self._load(model)
        else:
            if model and not model.endswith('.pt') and not model.endswith('.yaml'):
                raise NotImplementedError('This model only supports creating from pt file or yaml file.')
            if pretrain_rtdetr_model and not pretrain_rtdetr_model.endswith('.pt'):
                raise NotImplementedError('pretrain_rtdetr_model only supports creating from pt file.')
            self.predictor = None
            self.ckpt = None
            self.task = 'track'
            suffix = Path(model).suffix
            if suffix == '.yaml':
                self._new(model)
            else:
                self._load(model)

            self._load_pretrain_detr(pretrain_rtdetr_model)

    def _load_pretrain_detr(self, weights: str):
        import torch
        # 加载 RTDETRDetectionModel 的权重
        checkpoint = torch.load(weights)
        # 获取 RTDETRDetectionModel 的权重字典
        model_weights = {}
        for name, param in checkpoint['model'].model.named_parameters():
            if name[0:2] == '22':
                name_new = name[0:3] + 'decoder.' + name[3:]
                name = name_new
            model_weights[name] = param.clone()
        # print(model_state_dict)
        self.model.model.load_state_dict(model_weights, False)
        weights_path = 'weights_temp.pt'
        from ultralytics.utils.torch_utils import de_parallel
        from copy import deepcopy
        import os
        ckpt1 = {
            'model': deepcopy(de_parallel(self.model)).half(),
        }
        # Use dill (if exists) to serialize the lambda functions where pickle does not do this
        try:
            import dill as pickle
        except ImportError:
            import pickle
        torch.save(ckpt1, weights_path, pickle_module=pickle)

        self._load(weights_path)
        os.remove(weights_path)

        # 创建两个文件来保存模型参数
        model_1_params_file = r'C:\Users\a\Documents\test\detr_load_motr.txt'
        model_2_params_file = r'C:\Users\a\Documents\test\motr.txt'
        # 将模型参数保存到两个不同的文本文件
        # with open(model_1_params_file, 'w') as file_1:
        #     for name, param in model_weights.items():
        #         file_1.write(f'{name}: {param}\n')
        #
        # with open(model_2_params_file, 'w') as file_2:
        #     for name, param in self.model.model.named_parameters():
        #         file_2.write(f'{name}: {param}\n')
        # exit()

    def _new(self, cfg: str, verbose=True):
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = 'track'
        self.model = TrackingModel(cfg_dict, verbose=verbose)  # build model
        # print(self.model)
        # exit()
        # Below added to allow export from YAMLs
        self.model.args = DEFAULT_CFG_DICT  # attach args to model
        self.model.task = self.task

    @smart_inference_mode()
    def _load(self, weights: str):

        self.model, self.ckpt = attempt_load_one_weight(weights)
        self.model.args = DEFAULT_CFG_DICT  # attach args to model
        self.task = self.model.args['task']

    @smart_inference_mode()
    def load(self, weights='yolov8n.pt'):
        """
        Transfers parameters with matching names and shapes from 'weights' to model.
        """
        if isinstance(weights, (str, Path)):
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    @smart_inference_mode()
    def predict(self, source=None, stream=True, **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.engine.results.Results]): The prediction results.
        """
        if source is None:
            source = ROOT / 'assets' if is_git_dir() else 'https://ultralytics.com/images/bus.jpg'
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")
        overrides = dict(conf=0.25, task='track', mode='predict')
        overrides.update(kwargs)  # prefer kwargs
        import torch
        with torch.no_grad():
            if not self.predictor:
                self.predictor = TrackPredictor(overrides=overrides)
                self.predictor.setup_model(model=self.model)
                self.predictor.model.eval()

                # model_2_params_file = r'C:\Users\a\Documents\test\motr_predictor____.txt'
                # # 将模型参数保存到两个不同的文本文件
                # with open(model_2_params_file, 'w') as file_1:
                #     for name, param in self.predictor.model.named_parameters():
                #         file_1.write(f'{name}: {param}\n')

            else:  # only update args if predictor is already setup
                self.predictor.args = get_cfg(self.predictor.args, overrides)
        return self.predictor(source, stream=stream)

    def train(self, **kwargs):
        """
        Trains the model on a given dataset.

        Args:
            **kwargs (Any): Any number of arguments representing the training configuration.
        """
        overrides = dict(task='track', mode='train')
        overrides.update(kwargs)
        overrides['deterministic'] = False
        if not overrides.get('data'):
            raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
        if overrides.get('resume'):
            overrides['resume'] = self.ckpt_path
        self.task = overrides.get('task') or self.task

        self.trainer = TrackTrainer(overrides=overrides)
        if not overrides.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.train()
        # Update model and cfg after training
        if RANK in (-1, 0):
            self.model, _ = attempt_load_one_weight(str(self.trainer.best))
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)  # TODO: no metrics returned by DDP

    def val(self, **kwargs):
        """Run validation given dataset."""
        overrides = dict(task='track', mode='val')
        overrides.update(kwargs)  # prefer kwargs
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)
        validator = TrackValidator(args=args)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def info(self, verbose=True):
        """Get model info"""
        return model_info(self.model, verbose=verbose)

    def _check_is_pytorch_model(self):
        """
        Raises TypeError is model is not a PyTorch model
        """
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == '.pt'
        pt_module = isinstance(self.model, nn.Module)
        if not (pt_module or pt_str):
            raise TypeError(f"model='{self.model}' must be a *.pt PyTorch model, but is a different type. "
                            f'PyTorch models can be used to train, val, predict and export, i.e. '
                            f"'yolo export model=yolov8n.pt', but exported formats like ONNX, TensorRT etc. only "
                            f"support 'predict' and 'val' modes, i.e. 'yolo predict model=yolov8n.onnx'.")

    def fuse(self):
        """Fuse PyTorch Conv2d and BatchNorm2d layers."""
        self._check_is_pytorch_model()
        self.model.fuse()

    @smart_inference_mode()
    def export(self, **kwargs):
        """
        Export model.

        Args:
            **kwargs : Any other args accepted by the predictors. To see all args check 'configuration' section in docs
        """
        overrides = dict(task='track')
        overrides.update(kwargs)
        overrides['mode'] = 'export'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz:
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        if args.batch == DEFAULT_CFG.batch:
            args.batch = 1  # default to 1 if not modified
        return Exporter(overrides=args)(model=self.model)

    def __call__(self, source=None, stream=False, **kwargs):
        """Calls the 'predict' function with given arguments to perform object detection."""
        return self.predict(source, stream, **kwargs)

    def __getattr__(self, attr):
        """Raises error if object has no requested attribute."""
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")
