# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""X3D training script."""

import argparse

from mindspore import nn, context, load_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.metrics import Accuracy

from src.utils.check_param import Validator, Rel
from src.data.kinetics400 import Kinetic400
from src.data.transforms import VideoRandomCrop, VideoRandomHorizontalFlip, VideoRescale
from src.data.transforms import VideoNormalize, VideoShortEdgeResize, VideoReOrder
from src.schedule.lr_schedule import warmup_cosine_annealing_lr_v1
from src.models.x3d import x3d_m, x3d_l, x3d_s, x3d_xs


def x3d_kinetics400_train(args_opt):
    """X3D train"""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data Pipeline.
    if args_opt.run_distribute:
        if args_opt.device_target == "Ascend":
            init()
        else:
            init("nccl")

        rank_id = get_rank()
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num,
                                          parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        dataset = Kinetic400(args_opt.data_url,
                             split="train",
                             seq=args_opt.seq,
                             seq_mode=args_opt.seq_mode,
                             num_parallel_workers=args_opt.num_parallel_workers,
                             shuffle=True,
                             num_shards=device_num,
                             shard_id=rank_id,
                             batch_size=args_opt.batch_size,
                             repeat_num=args_opt.repeat_num,
                             frame_interval=args_opt.frame_interval)
        ckpt_save_dir = args_opt.ckpt_save_dir + "ckpt_" + str(get_rank()) + "/"
    else:
        dataset = Kinetic400(args_opt.data_url,
                             split="train",
                             seq=args_opt.seq,
                             seq_mode=args_opt.seq_mode,
                             num_parallel_workers=args_opt.num_parallel_workers,
                             shuffle=True,
                             batch_size=args_opt.batch_size,
                             repeat_num=args_opt.repeat_num,
                             frame_interval=args_opt.frame_interval)
        ckpt_save_dir = args_opt.ckpt_save_dir

    # perpare dataset.
    if args_opt.model_name == "x3d_m":
        transforms = [VideoShortEdgeResize((256)),
                      VideoRandomCrop([224, 224]),
                      VideoRandomHorizontalFlip(0.5),
                      VideoRescale(shift=0),
                      VideoReOrder((3, 0, 1, 2)),
                      VideoNormalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])]
    elif args_opt.model_name == "x3d_l":
        transforms = [VideoShortEdgeResize((356)),
                      VideoRandomCrop([312, 312]),
                      VideoRandomHorizontalFlip(0.5),
                      VideoRescale(shift=0),
                      VideoReOrder((3, 0, 1, 2)),
                      VideoNormalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])]
    elif args_opt.model_name == "x3d_s" or args_opt.model_name == "x3d_xs":
        transforms = [VideoShortEdgeResize((182)),
                      VideoRandomCrop([160, 160]),
                      VideoRandomHorizontalFlip(0.5),
                      VideoRescale(shift=0),
                      VideoReOrder((3, 0, 1, 2)),
                      VideoNormalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])]

    dataset.transform = transforms
    dataset_train = dataset.run()
    Validator.check_int(dataset_train.get_dataset_size(), 0, Rel.GT)
    step_size = dataset_train.get_dataset_size()

    # Create model.
    if args_opt.model_name == "x3d_m":
        network = x3d_m(num_classes=args_opt.num_classes,
                        dropout_rate=args_opt.dropout_rate,
                        depth_factor=args_opt.depth_factor,
                        num_frames=args_opt.num_frames,
                        train_crop_size=args_opt.train_crop_size)
    elif args_opt.model_name == "x3d_s":
        network = x3d_s(num_classes=args_opt.num_classes,
                        dropout_rate=args_opt.dropout_rate,
                        depth_factor=args_opt.depth_factor,
                        num_frames=args_opt.num_frames,
                        train_crop_size=args_opt.train_crop_size)
    elif args_opt.model_name == "x3d_xs":
        network = x3d_xs(num_classes=args_opt.num_classes,
                         dropout_rate=args_opt.dropout_rate,
                         depth_factor=args_opt.depth_factor,
                         num_frames=args_opt.num_frames,
                         train_crop_size=args_opt.train_crop_size)
    elif args_opt.model_name == "x3d_l":
        network = x3d_l(num_classes=args_opt.num_classes,
                        dropout_rate=args_opt.dropout_rate,
                        depth_factor=args_opt.depth_factor,
                        num_frames=args_opt.num_frames,
                        train_crop_size=args_opt.train_crop_size)

    # Load pretrained model.
    if args_opt.pretrained:
        param_dict = load_checkpoint(args_opt.pretrained_model)
        load_param_into_net(network, param_dict)

    # Set lr scheduler.
    learning_rate = warmup_cosine_annealing_lr_v1(lr=args_opt.learning_rate,
                                                  steps_per_epoch=step_size,
                                                  warmup_epochs=args_opt.warmup_epochs,
                                                  max_epoch=args_opt.epoch_size,
                                                  t_max=args_opt.epoch_size,
                                                  eta_min=0)

    # Define optimizer.
    network_opt = nn.SGD(network.trainable_params(),
                         learning_rate,
                         momentum=args_opt.momentum,
                         weight_decay=args_opt.weight_decay)

    # Define loss function.
    network_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Set the checkpoint config for the network.
    ckpt_config = CheckpointConfig(
        save_checkpoint_steps=step_size,
        keep_checkpoint_max=args_opt.keep_checkpoint_max)
    ckpt_callback = ModelCheckpoint(prefix='x3d_kinetics400',
                                    directory=ckpt_save_dir,
                                    config=ckpt_config)

    # Init the model.
    model = Model(network,
                  loss_fn=network_loss,
                  optimizer=network_opt,
                  metrics={"Accuracy": Accuracy()})

    # Begin to train.
    print('[Start training `{}`]'.format('x3d_kinetics400'))
    print("=" * 80)
    model.train(args_opt.epoch_size,
                dataset_train,
                callbacks=[ckpt_callback, LossMonitor()],
                dataset_sink_mode=args_opt.dataset_sink_mode)
    print('[End of training `{}`]'.format('x3d_kinetics400'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='X3D train.')
    parser.add_argument('--device_target', type=str, default="GPU",
                        choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--run_distribute', type=bool, default=False,
                        help='Distributed parallel training.')
    parser.add_argument('--data_url', type=str, default="/home/publicfile/kinetics-400",
                        help='Location of data.')
    parser.add_argument('--seq', type=int, default=16,
                        help='Number of frames of captured video. X3D_XS=4, X3D_S=13, X3D_M/L=16')
    parser.add_argument('--seq_mode', type=str, default='interval')
    parser.add_argument('--num_parallel_workers', type=int, default=8,
                        help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Number of batch size. X3D_XS/S/M=16, X3D_L=8')
    parser.add_argument('--repeat_num', type=int, default=1,
                        help='Number of repeat.')
    parser.add_argument('--frame_interval', type=int, default=5,
                        help='Interval between sampling frames. X3D_XS=12, X3D_S=6, X3D_M/L=5')
    parser.add_argument('--ckpt_save_dir', type=str, default="./x3d",
                        help='Location of training outputs.')
    parser.add_argument("--model_name", type=str, default="x3d_m",
                        help="Name of model.", choices=["x3d_m", "x3d_l", "x3d_s", "x3d_xs"])
    parser.add_argument('--num_classes', type=int, default=400,
                        help='Number of classification.')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout rate.')
    parser.add_argument('--depth_factor', type=float, default=2.2,
                        help='Depth expansion factor.')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='The number of frames of the input clip.'
                             'X3D_XS=4, X3D_S=13, X3D_M/L=16')
    parser.add_argument('--train_crop_size', type=int, default=224,
                        help='The spatial crop size for training.'
                             'X3D_S/XS=160, X3D_M=224, X3D_L=312')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='Load pretrained model.')
    parser.add_argument('--pretrained_model', type=str, default="",
                        help='Location of Pretrained Model.')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Initial learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for the moving average.')
    parser.add_argument('--warmup_epochs', type=int, default=35,
                        help='Warmup epochs.')
    parser.add_argument('--epoch_size', type=int, default=300,
                        help='Train epoch size.')
    parser.add_argument('--weight_decay', type=float, default=0.00005,
                        help='Weight decay for the optimizer.')
    parser.add_argument('--keep_checkpoint_max', type=int, default=10,
                        help='Max number of checkpoint files.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False,
                        help='The dataset sink mode.')

    args = parser.parse_known_args()[0]
    x3d_kinetics400_train(args)
