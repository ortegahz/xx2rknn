import numpy as np
import cv2
from rknn.api import RKNN


def show_outputs(outputs):
    output = outputs[0].reshape(-1)
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


if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

    # pre-process config
    print('--> config model')
    rknn.config(channel_mean_value='123.6750 116.2800 103.5300 58.3950', reorder_channel='0 1 2')
    print('done')



    # Load tensorflow model
    print('--> Loading model')
    ret = rknn.load_caffe(model='/media/manu/intel/workspace/MXNet2Caffe/model_caffe/model-0001.prototxt',
                          proto='caffe',
                          blobs='/media/manu/intel/workspace/MXNet2Caffe/model_caffe/model-0001.caffemodel')
    if ret != 0:
        print('Load model failed! Ret = {}'.format(ret))
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=True, dataset='./dataset.txt')
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn('./model.rknn')
    if ret != 0:
        print('Export model.rknn failed!')
        exit(ret)
    print('done')

    # Set inputs
    img = cv2.imread('/media/manu/samsung/pics/126_1152_576.bmp')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print('--> Init runtime environment')
    # ret = rknn.init_runtime()
    ret = rknn.init_runtime(target='rk3399', device_id='TDs33101190500149')
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

    # show_outputs(outputs)
    print('done')

    # perf
    print('--> Begin evaluate model performance')
    perf_results = rknn.eval_perf(inputs=[img])
    print('done')

    rknn.release()

