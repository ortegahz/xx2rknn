import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN

PATH_DS = '/home/manu/tmp/dataset_single.txt'
PATH_IMG = '/media/manu/samsung/pics/1001_org.bmp'

ONNX_MODEL = '/home/manu/tmp/wf42m_pfc02_8gpus_r50_bs1k/model.onnx'
RKNN_MODEL = '/home/manu/tmp/face_recog_f16_1126.rknn'


def show_outputs(outputs):
    output = outputs[0][0]
    output_sorted = sorted(output, reverse=True)
    top5_str = 'model\n-----TOP 5-----\n'
    for i in range(5):
        value = output_sorted[i]
        index = np.where(output == value)
        for j in range(len(index)):
            if (i + j) >= 5:
                break
            if value > 0:
                topi = '{}: {}\n'.format(index[j], value)
            else:
                topi = '-1: 0.0\n'
            top5_str += topi
    print(top5_str)


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

    # Create RKNN object
    rknn = RKNN()

    # If model does not exist, download it.
    # Download address:
    # https://s3.amazonaws.com/onnx-model-zoo/resnet/model/model.onnx
    if not os.path.exists(ONNX_MODEL):
        print('--> Download {}'.format(ONNX_MODEL))
        url = 'https://s3.amazonaws.com/onnx-model-zoo/resnet/model/model.onnx'
        download_file = ONNX_MODEL
        try:
            start_time = time.time()
            urllib.request.urlretrieve(url, download_file, show_progress)
        except:
            print('Download {} failed.'.format(download_file))
            print(traceback.format_exc())
            exit(-1)
        print('done')
    
    # pre-process config
    print('--> config model')
    # rknn.config(channel_mean_value='123.675 116.28 103.53 58.82', reorder_channel='0 1 2')
    # ([0 255] - 127.5) / 127.5 --> [-1 1]
    # rknn.config(channel_mean_value='127.5 127.5 127.5 127.5', reorder_channel='0 1 2', target_platform=['rk3399pro'])
    rknn.config(channel_mean_value='127.5 127.5 127.5 127.5', reorder_channel='0 1 2', target_platform=['rv1126'])
    print('done')

    # Load tensorflow model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    # ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    ret = rknn.build(do_quantization=False, dataset=PATH_DS, pre_compile=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export model.rknn failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread(PATH_IMG)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[img])

    # save outputs
    for save_i in range(len(outputs)):
        save_output = outputs[save_i].flatten()
        np.savetxt('/home/manu/tmp/rknn_output_%s.txt' % save_i, save_output,
                   fmt="%f", delimiter="\n")

    # x = outputs[0]
    # output = np.exp(x)/np.sum(np.exp(x))
    # outputs = [output]
    # show_outputs(outputs)

    print('done')

    rknn.release()

