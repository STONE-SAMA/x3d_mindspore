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
"""X3D eval script."""

import argparse

from mindspore import nn
from mindspore import context, load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import Callback

from src.data.kinetics400 import Kinetic400
from src.data.transforms import VideoReOrder, VideoRescale, VideoNormalize
from src.data.transforms import VideoCenterCrop, VideoShortEdgeResize
from src.models.x3d import x3d_m, x3d_l, x3d_s, x3d_xs


class PrintEvalStep(Callback):
    """ print eval step """
    def step_end(self, run_context):
        """ eval step """
        cb_params = run_context.original_args()
        print("eval: {}/{}".format(cb_params.cur_step_num, cb_params.batch_num))


def x3d_kinetics400_eval(args_opt):
    """X3D eval"""
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # Data Pipeline.
    dataset_eval = Kinetic400(args_opt.data_url,
                              split="val",
                              seq=args_opt.seq,
                              seq_mode=args_opt.seq_mode,
                              num_parallel_workers=args_opt.num_parallel_workers,
                              shuffle=False,
                              batch_size=args_opt.batch_size,
                              repeat_num=args_opt.repeat_num,
                              frame_interval=args_opt.frame_interval,
                              num_clips=args_opt.num_clips)

    # perpare dataset.
    if args_opt.model_name == "x3d_m":
        transforms = [VideoShortEdgeResize(size=256),
                      VideoCenterCrop([256, 256]),
                      VideoRescale(shift=0),
                      VideoReOrder((3, 0, 1, 2)),
                      VideoNormalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])]
    elif args_opt.model_name == "x3d_l":
        transforms = [VideoShortEdgeResize(size=356),
                      VideoCenterCrop([356, 356]),
                      VideoRescale(shift=0),
                      VideoReOrder((3, 0, 1, 2)),
                      VideoNormalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])]
    elif args_opt.model_name == "x3d_s" or args_opt.model_name == "x3d_xs":
        transforms = [VideoShortEdgeResize(size=182),
                      VideoCenterCrop([182, 182]),
                      VideoRescale(shift=0),
                      VideoReOrder((3, 0, 1, 2)),
                      VideoNormalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225])]
    dataset_eval.transform = transforms
    dataset_eval = dataset_eval.run()

    # Create model.
    if args_opt.model_name == "x3d_m":
        network = x3d_m(num_classes=args_opt.num_classes, 
                        eval_with_clips=args_opt.eval_with_clips)
    elif args_opt.model_name == "x3d_s":
        network = x3d_s(num_classes=args_opt.num_classes,
                        eval_with_clips=args_opt.eval_with_clips)
    elif args_opt.model_name == "x3d_xs":
        network = x3d_xs(num_classes=args_opt.num_classes,
                        eval_with_clips=args_opt.eval_with_clips)
    elif args_opt.model_name == "x3d_l":
        network = x3d_l(num_classes=args_opt.num_classes,
                        eval_with_clips=args_opt.eval_with_clips)

    # Define loss function.
    network_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # Load pretrained model.
    if args_opt.pretrained:
        param_dict = load_checkpoint(args_opt.pretrained_model)
        load_param_into_net(network, param_dict)

    # Define eval_metrics.
    eval_metrics = {'Loss': nn.Loss(),
                    'Top_1_Accuracy': nn.Top1CategoricalAccuracy(),
                    'Top_5_Accuracy': nn.Top5CategoricalAccuracy()}
    print_cb = PrintEvalStep()

    # Init the model.
    model = Model(network, loss_fn=network_loss, metrics=eval_metrics)

    # Begin to eval.
    print('[Start eval `{}`]'.format('x3d_kinetics400'))
    result = model.eval(dataset_eval,
                        callbacks=[print_cb],
                        dataset_sink_mode=args_opt.dataset_sink_mode)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='X3D eval.')
    parser.add_argument('--device_target', type=str, default="GPU",
                        choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--data_url', type=str, default="/home/publicfile/kinetics-400",
                        help='Location of data.')
    parser.add_argument('--seq', type=int, default=16,
                        help='Number of frames of captured video.'
                             'X3D_XS=4, X3D_S=13, X3D_M/L=16')
    parser.add_argument('--seq_mode', type=str, default='interval')
    parser.add_argument('--num_parallel_workers', type=int, default=8,
                        help='Number of parallel workers.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of batch size.')
    parser.add_argument('--repeat_num', type=int, default=1,
                        help='Number of repeat.')
    parser.add_argument('--frame_interval', type=int, default=5,
                        help='Interval between sampling frames.'
                             'X3D_XS=12, X3D_S=6, X3D_M/L=5')
    parser.add_argument("--model_name", type=str, default="x3d_m",
                        help="Name of model.", choices=["x3d_m", "x3d_l", "x3d_s", "x3d_xs"])
    parser.add_argument('--num_classes', type=int, default=400,
                        help='Number of classification.')
    parser.add_argument('--eval_with_clips', type=bool, default=False,
                        help='Whether use 10-clip eval.')
    parser.add_argument('--num_clips', type=int, default=1,
                        help='Number of clips.', choices=[1, 10])
    parser.add_argument('--pretrained', type=bool, default=True,
                        help='Load pretrained model.')
    parser.add_argument('--pretrained_model', type=str,
                        default="",
                        help='Location of Pretrained Model.')
    parser.add_argument('--dataset_sink_mode', type=bool, default=False,
                        help='The dataset sink mode.')

    args = parser.parse_known_args()[0]
    x3d_kinetics400_eval(args)
