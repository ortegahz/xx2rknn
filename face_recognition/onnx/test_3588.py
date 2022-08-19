import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

import torch
import onnx
import onnxruntime

DIR_OUT = '/home/manu/tmp'

PATH_IMG = '/media/manu/samsung/pics/1001_org.bmp'
# PATH_DS = '/home/manu/tmp/dataset.txt'
# PATH_DS = '/media/manu/kingstop/itx-3588j/Linux_SDK/rk3588/external/rknn-toolkit2/examples/onnx/resnet50v2/dataset.txt'
PATH_DS = '/home/manu/tmp/dataset_single.txt'

ONNX_MODEL = '/home/manu/tmp/wf42m_pfc02_8gpus_r50_bs1k/model.onnx'
RKNN_MODEL = '/home/manu/tmp/model.rknn'


def readable_speed(speed):
    speed_bytes = float(speed)
    speed_kbytes = speed_bytes / 1024
    if speed_kbytes > 1024:
        speed_mbytes = speed_kbytes / 1024
        if speed_mbytes > 1024:
            speed_gbytes = speed_mbytes / 1024
            return "{:.2f} GB/s".format(speed_gbytes)
        else:
            return "{:.2f} MB/s".format(speed_mbytes)
    else:
        return "{:.2f} KB/s".format(speed_kbytes)


def show_progress(blocknum, blocksize, totalsize):
    speed = (blocknum * blocksize) / (time.time() - start_time)
    speed_str = " Speed: {}".format(readable_speed(speed))
    recv_size = blocknum * blocksize

    f = sys.stdout
    progress = (recv_size / totalsize)
    progress_str = "{:.2f}%".format(progress * 100)
    n = round(progress * 50)
    s = ('#' * n).ljust(50, '-')
    f.write(progress_str.ljust(8, ' ') + '[' + s + ']' + speed_str)
    f.flush()
    f.write('\r\n')


if __name__ == '__main__':

    # inference using onnx model (same code for acc verification)
    net_onnx = onnx.load(ONNX_MODEL)
    onnx.checker.check_model(net_onnx)
    ort_session = onnxruntime.InferenceSession(ONNX_MODEL)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy().astype(np.float32)

    img = cv2.imread(PATH_IMG)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}  # single input
    ort_outs = ort_session.run(None, ort_inputs)
    feat_onnx = ort_outs[0].flatten()  # single output
    np.savetxt(os.path.join(DIR_OUT, 'feat_onnx.txt'), feat_onnx, fmt="%f", delimiter="\n")

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(mean_values=[127.5, 127.5, 127.5], std_values=[127.5, 127.5, 127.5], target_platform='rk3588')
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset=PATH_DS)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(PATH_IMG)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Accuracy analysis
    print('--> Accuracy analysis')
    ret = rknn.accuracy_analysis(inputs=[PATH_IMG], output_dir=os.path.join(DIR_OUT, 'snapshot'))
    if ret != 0:
        print('Accuracy analysis failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])
    np.savetxt(os.path.join(DIR_OUT, 'feat_rknn.txt'), outputs[0], fmt="%f", delimiter="\n")
    print('done')

    rknn.release()
