import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from utils.common import *
from model import SRCNN


def begin_sr(architecture="915", image_path="dataset/test1.png", ckpt_path="checkpoint/SRCNN915/SRCNN-915.h5", scale=2):
    if scale < 1 or scale > 5:
        ValueError("scale must be 2, 3, or 4")
    if architecture not in ["915", "935", "955"]:
        ValueError("architecture must be 915, 935 or 955")
#считываем изображение и сохраняем бикубическое
    lr_image = read_image(image_path)
    bicubic_image = upscale(lr_image, scale)
    write_image("bicubic.png", bicubic_image)
# предварительная обработка изображения низкого качества(исходного)
    lr_image = gaussian_blur(lr_image, sigma=0.3)
    lr_image = upscale(lr_image, scale)
    lr_image = rgb2ycbcr(lr_image)
    lr_image = norm01(lr_image)
    lr_image = tf.expand_dims(lr_image, axis=0)
# преобразовываем с помощью нейросети изображение и сохраняем
    model = SRCNN(architecture)
    model.load_weights(ckpt_path)
    sr_image = model.predict(lr_image)[0]

    sr_image = denorm01(sr_image)
    sr_image = tf.cast(sr_image, tf.uint8)
    sr_image = ycbcr2rgb(sr_image)

    write_image("sr.png", sr_image)