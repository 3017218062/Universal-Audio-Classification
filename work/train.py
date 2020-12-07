from __init__ import *
from arg import args
from preprocess import TrainValSplit
from transform import UacTransformer
from dataset import UacDataset
from callback import CSVLogger, ModelCheckpoint, FlexibleTqdm, LearningCurve
from model import UacClassifier
from util import *

print(args)
#####################################################################################################################
SEED = 2020
pl.seed_everything(SEED)

input_path = "../input/train/"
output_path = "../output/"
device = "cuda" if torch.cuda.is_available() else "cpu"
#####################################################################################################################
splitter = TrainValSplit(input_path, folds=5, fold=args.fold, one_hot=True, seed=SEED)
transforms = UacTransformer()
train_dataset = UacDataset(*splitter["train"], args.target_size, transforms["train"])
val_dataset = UacDataset(*splitter["val"], args.target_size, transforms["val"])
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                              drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)
#####################################################################################################################
postfix = "_%s_fold%d" % (args.model, args.fold)
trainer_params = {
    # hardware configuration
    "gpus": str(args.gpu),  # None
    "auto_select_gpus": False,
    "num_nodes": 1,
    "tpu_cores": None,
    # auto mixed precision
    "precision": args.precision,  # 32
    "amp_backend": "native",
    "amp_level": "O2",
    # training configuration
    "max_epochs": args.epochs,  # 1000
    "min_epochs": 1,
    "max_steps": None,
    "min_steps": None,
    # logger and checkpoint
    "checkpoint_callback": False,  # True
    "logger": False,  # TensorBoardLogger
    "default_root_dir": os.getcwd(),
    "flush_logs_every_n_steps": 100,
    "log_every_n_steps": 50,
    "log_gpu_memory": None,
    "check_val_every_n_epoch": 1,
    "val_check_interval": 1.0,
    "resume_from_checkpoint": None,
    "progress_bar_refresh_rate": 0,  # 1
    "weights_summary": None,  # "top"
    "weights_save_path": os.getcwd(),
    # catch any bugs
    "num_sanity_val_steps": 0,  # 2
    "fast_dev_run": False,
    "reload_dataloaders_every_epoch": False,
    # distributed backend
    "accelerator": None,
    "accumulate_grad_batches": args.accumulate_grad_batches,  # 1
    # auto optimization
    "automatic_optimization": True,
    "auto_scale_batch_size": None,
    "auto_lr_find": False,
    # deterministic
    "benchmark": False,
    "deterministic": True,  # False

    "gradient_clip_val": 0.0,
    "sync_batchnorm": False,
    # limit and sample
    "limit_train_batches": 1.0,
    "limit_val_batches": 1.0,
    "limit_test_batches": 1.0,
    "overfit_batches": 0.0,
    "prepare_data_per_node": True,
    "replace_sampler_ddp": True,
    # other
    "callbacks": [
        FlexibleTqdm(len(train_dataset) // args.batch_size, column_width=12),
        CSVLogger(dirpath=output_path, filename="history" + postfix),
        ModelCheckpoint(dirpath=output_path, filename="checkpoint" + postfix, monitor="val_acc", mode="max"),
        LearningCurve(dirpath=output_path, filename="history" + postfix, figsize=(12, 4), names=("loss", "acc", "f1")),
    ],  # None
    "process_position": 0,
    "profiler": None,
    "track_grad_norm": -1,
    "truncated_bptt_steps": None,
}
#####################################################################################################################
trainer = pl.Trainer(**trainer_params)
model = UacClassifier(classes=len(splitter["class"]), model=args.model, pretrained=args.pretrained,
                      learning_rate=args.learning_rate, weight_decay=args.weight_decay, epochs=args.epochs,
                      min_lr=args.min_lr, smooth=args.smooth, ohem=args.ohem)
#####################################################################################################################
trainer.fit(model, train_dataloader, val_dataloader)
