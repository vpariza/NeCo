import click
import os
import torch
import pandas as pd
import sacred

from datetime import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import NeptuneLogger
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
from torchvision.transforms.functional import InterpolationMode

from data.coco.coco_data_module import CocoDataModule
from data.imagenet.imagenet_data_module import ImageNetDataModule
from data.utils import TrainDataModule
from data.VOCdevkit.vocdata import VOCDataModule, TrainXVOCValDataModule
from experiments.utils import get_backbone_weights

from src.neco import NeCo
from src.neco_transforms import NeCoTransforms
from src.evaluate_attn_maps import EvaluateAttnMaps

ex = sacred.experiment.Experiment()
api_key = "<ADD YOUR API KEY HERE>"


@click.command()
@click.option("--config_path", type=str)
@click.option("--seed", type=int, default=400)
def entry_script(config_path, seed):
    if config_path is not None:
        ex.add_config(os.path.join(os.path.abspath(os.path.dirname(__file__)), config_path))
    else:
        ex.add_config(os.path.join(os.path.abspath(os.path.dirname(__file__)), "neco_config_dev.yml"))
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    ex_name = f"neco-{time}"
    checkpoint_dir = os.path.join(ex.configurations[0]._conf["train"]["checkpoint_dir"], ex_name)
    ex.observers.append(sacred.observers.FileStorageObserver(checkpoint_dir))
    ex.run(config_updates={'seed': seed}, options={'--name': ex_name})


@ex.main
@ex.capture
def finetune_with_spatial_loss(_config, _run):
    # Setup logging
    print("Online mode")
    neptune_logger = NeptuneLogger(
        api_key=api_key,
        offline_mode=_config["log_status"] == 'offline' if _config.get("log_status") else True,
        project_name="<Your Project Name>",
        experiment_name=_run.experiment_info["name"],
        params=pd.json_normalize(_config).to_dict(orient='records')[0],
        tags=_config["tags"].split(','),
    )

    # Process config
    print("Config:")
    print(_config)
    data_config = _config["data"]
    train_config = _config["train"]
    seed_everything(_config["seed"])

    # Init data modules and tranforms
    data_dir = data_config["data_dir"]
    dataset_name = data_config["dataset_name"]
    train_transforms = NeCoTransforms(size_crops=data_config["size_crops"],
                                         nmb_crops=data_config["nmb_samples"],
                                         min_intersection=data_config["min_intersection_crops"],
                                         min_scale_crops=data_config["min_scale_crops"],
                                         max_scale_crops=data_config["max_scale_crops"],
                                         jitter_strength=data_config["jitter_strength"],
                                         blur_strength=data_config["blur_strength"])

    # Setup voc dataset used for evaluation
    val_size = data_config["size_crops_val"]
    val_image_transforms = Compose([Resize((val_size, val_size)),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    val_target_transforms = Compose([Resize((val_size, val_size), interpolation=InterpolationMode.NEAREST),
                                     ToTensor()])
    val_batch_size = train_config.get("val_batch_size", train_config["batch_size"])

    val_data_module = VOCDataModule(batch_size=val_batch_size,
                                    num_workers=_config["num_workers"],
                                    train_split="trainaug",
                                    val_split="val",
                                    data_dir=data_config["voc_data_path"],
                                    train_image_transform=train_transforms,
                                    val_image_transform=val_image_transforms,
                                    val_target_transform=val_target_transforms)

    # Setup train data
    if dataset_name == "coco":
        file_list = os.listdir(os.path.join(data_dir, "train2017"))
        train_data_module = CocoDataModule(batch_size=train_config["batch_size"],
                                           num_workers=_config["num_workers"],
                                           file_list=file_list,
                                           data_dir=data_dir,
                                           train_transforms=train_transforms,
                                           val_transforms=None)
    elif dataset_name == 'imagenet100k':
        num_images = 126689
        with open(os.path.join(data_dir, "imagenet100.txt")) as f:
            class_names = [line.rstrip('\n') for line in f]
        train_data_module = ImageNetDataModule(train_transforms=train_transforms,
                                               batch_size=train_config["batch_size"],
                                               class_names=class_names,
                                               num_workers=_config["num_workers"],
                                               data_dir=os.path.join(data_dir, "train"),
                                               num_images=num_images)
    elif dataset_name == 'imagenet':
        num_images = 1281167
        data_dir = os.path.join(data_dir, "train")
        class_names = os.listdir(data_dir)
        assert len(class_names) == 1000
        train_data_module = ImageNetDataModule(train_transforms=train_transforms,
                                               batch_size=train_config["batch_size"],
                                               class_names=class_names,
                                               num_workers=_config["num_workers"],
                                               data_dir=data_dir,
                                               num_images=num_images)
    elif dataset_name == 'voc':
        train_data_module = VOCDataModule(batch_size=train_config["batch_size"],
                                    num_workers=_config["num_workers"],
                                    train_split="trainaug",
                                    val_split="val",
                                    data_dir=data_config["voc_data_path"],
                                    train_image_transform=train_transforms,
                                    val_image_transform=val_image_transforms,
                                    val_target_transform=val_target_transforms)
    elif dataset_name == 'coco,imagenet100k':
        cc_dir, in100k_dir = data_dir[0], data_dir[1]
        file_list = os.listdir(os.path.join(cc_dir, "train2017"))
        train_data_module_1 = CocoDataModule(batch_size=train_config["batch_size"],
                                           num_workers=_config["num_workers"],
                                           file_list=file_list,
                                           data_dir=cc_dir,
                                           train_transforms=train_transforms,
                                           val_transforms=None)
        train_data_module_1.setup()
        dataset_1 = train_data_module_1.coco_train
        num_images = 126689
        with open(os.path.join(in100k_dir, "imagenet100.txt")) as f:
            class_names = [line.rstrip('\n') for line in f]
        train_data_module_2 = ImageNetDataModule(train_transforms=train_transforms,
                                               batch_size=train_config["batch_size"],
                                               class_names=class_names,
                                               num_workers=_config["num_workers"],
                                               data_dir=os.path.join(in100k_dir, "train"),
                                               num_images=num_images)
        train_data_module_2.setup()
        dataset_2 = train_data_module_2.im_train
        # Combine the two datasets
        train_data_module = TrainDataModule(
                            torch.utils.data.ConcatDataset([dataset_1, dataset_2]),
                              _config["num_workers"], train_config["batch_size"])
    elif dataset_name == 'coco,imagenet':
        cc_dir, imgnet_dir = data_dir[0], data_dir[1]

        # Load and Setup COCO Dataset Module
        file_list = os.listdir(os.path.join(cc_dir, "train2017"))
        train_data_module_1 = CocoDataModule(batch_size=train_config["batch_size"],
                                           num_workers=_config["num_workers"],
                                           file_list=file_list,
                                           data_dir=cc_dir,
                                           train_transforms=train_transforms,
                                           val_transforms=None)
        train_data_module_1.setup()
        dataset_1 = train_data_module_1.coco_train
        
        # Load and Setup Imagenet Dataset Module
        num_images = 1281167
        imgnet_dir = os.path.join(imgnet_dir, "train")
        class_names = os.listdir(imgnet_dir)
        assert len(class_names) == 1000
        train_data_module_2 = ImageNetDataModule(train_transforms=train_transforms,
                                               batch_size=train_config["batch_size"],
                                               class_names=class_names,
                                               num_workers=_config["num_workers"],
                                               data_dir=imgnet_dir,
                                               num_images=num_images)
        train_data_module_2.setup()
        dataset_2 = train_data_module_2.im_train
        # Combine the two datasets
        train_data_module = TrainDataModule(
                            torch.utils.data.ConcatDataset([dataset_1, dataset_2]),
                              _config["num_workers"], train_config["batch_size"])
    else:

        raise ValueError(f"Data set {dataset_name} not supported")

    # Use data module wrapper to have train_data_module provide train loader and voc data module the val loader
    data_module = TrainXVOCValDataModule(train_data_module, val_data_module)

    # Init method
    model = NeCo(
        use_teacher=train_config["use_teacher"],
        loss_mask=train_config["loss_mask"],
        queue_length=train_config["queue_length"],
        momentum_teacher=train_config["momentum_teacher"],
        momentum_teacher_end=train_config["momentum_teacher_end"],
        num_clusters_kmeans=train_config["num_clusters_kmeans_miou"],
        weight_decay_end=train_config["weight_decay_end"],
        roi_align_kernel_size=train_config["roi_align_kernel_size"],
        val_downsample_masks=train_config["val_downsample_masks"],
        arch=train_config["arch"],
        patch_size=train_config["patch_size"],
        lr_heads=train_config["lr_heads"],
        gpus=_config["gpus"],
        num_classes=data_config["num_classes_val"],
        batch_size=train_config["batch_size"],
        num_samples=len(data_module),
        projection_feat_dim=train_config["projection_feat_dim"],
        projection_hidden_dim=train_config["projection_hidden_dim"],
        n_layers_projection_head=train_config["n_layers_projection_head"],
        max_epochs=train_config["max_epochs"],
        val_iters=train_config["val_iters"],
        temperature=train_config["temperature"],
        crops_for_assign=train_config["crops_for_assign"],
        nmb_crops=data_config["nmb_samples"],
        optimizer=train_config["optimizer"],
        exclude_norm_bias=train_config["exclude_norm_bias"],
        lr_backbone=train_config["lr_backbone"],
        final_lr=train_config["final_lr"],
        weight_decay=train_config["weight_decay"],
        arch_version=train_config["arch_version"],
        student_steepness=train_config["student_steepness"],
        teacher_steepness=train_config["teacher_steepness"] if "teacher_steepness" in train_config else None,
        grad_norm_clipping=train_config["grad_norm_clipping"] if "grad_norm_clipping" in train_config else None,
        sort_net=train_config["sort_net"],
        trainable_blocks=train_config["trainable_blocks"] if "trainable_blocks" in train_config else None,
        sim_dist=train_config["sim_dist"] if "sim_dist" in train_config else None,
        is_queue_usable=train_config["is_queue_usable"] if "is_queue_usable" in train_config else False, # using a queue to stabilize finding neighbors from, not necessary
    )

    # Optionally load weights
    if train_config["checkpoint"] is not None and train_config["only_load_weights"]:
        state_dict = torch.load(train_config["checkpoint"])
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "cls_token" in state_dict:
            state_dict_model = {f"model.{k}": v for k, v in state_dict.items()}
            teacher_dict_model = {f"teacher.{k}": v for k, v in state_dict.items()}
            state_dict = {**state_dict_model, **teacher_dict_model}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    elif train_config["checkpoint"] is None:
        if train_config["pretrained_weights"] is not None:
            w_student = get_backbone_weights(train_config["arch"],
                                             train_config["pretrained_weights"],
                                             patch_size=train_config.get("patch_size"),
                                             weight_prefix="model")
            w_teacher = get_backbone_weights(train_config["arch"],
                                             train_config["pretrained_weights"],
                                             patch_size=train_config.get("patch_size"),
                                             weight_prefix="teacher")
            msg = model.load_state_dict({**w_student, **w_teacher}, strict=False)
            print(msg)

    # Setup attention map evaluation callback
    checkpoint_dir = os.path.join(train_config["checkpoint_dir"], _run.experiment_info["name"])
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_dir,'all_ckps'),
        save_top_k=-1,
        verbose=True,
        save_on_train_epoch_end=False
    )
    callbacks = [checkpoint_callback]
    eval_attn = EvaluateAttnMaps(voc_root=data_config["voc_data_path"], train_input_height=data_config["size_crops"][0],
                                 attn_batch_size=train_config["batch_size"] * 4, num_workers=_config["num_workers"])
    callbacks.append(eval_attn)

    # Used if train data is small as for pvoc
    val_every_n_epochs = train_config.get("val_every_n_epochs")
    if val_every_n_epochs is None:
        val_every_n_epochs = 1

    # Setup trainer and start training
    trainer = Trainer(
        check_val_every_n_epoch=val_every_n_epochs,
        logger=neptune_logger,
        max_epochs=train_config["max_epochs"],
        gpus=_config["gpus"],
        accelerator='ddp' if _config["gpus"] > 1 else None,
        fast_dev_run=train_config["fast_dev_run"],
        log_every_n_steps=50,
        benchmark=True,
        deterministic=False,
        amp_backend='native',
        num_sanity_val_steps=0, # train_config['val_iters'],
        resume_from_checkpoint=train_config['checkpoint'] if not train_config["only_load_weights"] else None,
        terminate_on_nan=True,
        callbacks=callbacks
    )
    # trainer.validate(model, val_data_module)
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    entry_script()
                             