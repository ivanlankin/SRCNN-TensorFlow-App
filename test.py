import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from utils.common import *
from model import SRCNN 
from demo import *

def test_psnr(scale=2, architecture="915", mode="hr"):
    k = 6 + int(architecture[1]) // 2

    if scale < 1 or scale > 5:
        ValueError("scale must be 2, 3, or 4")

    if architecture not in ["915", "935", "955"]:
        ValueError("architecture must be 915, 935 or 955")

# -----------------------------------------------------------
#  load model
# -----------------------------------------------------------

    ckpt_path = f"checkpoint/SRCNN{architecture}/SRCNN-{architecture}.h5"

    model = SRCNN(architecture)
    model.load_weights(ckpt_path)


# -----------------------------------------------------------
# test 
# -----------------------------------------------------------

    ls_data = sorted_list(f"dataset/test/x{scale}/data")
    ls_labels = sorted_list(f"dataset/test/x{scale}/labels")

    sum_psnr = 0
    for i in range(0, len(ls_data)):
        lr_image = read_image(ls_data[i])
        lr_image = gaussian_blur(lr_image, sigma=0.3)
        lr_image = upscale(lr_image, scale)
        lr_image = rgb2ycbcr(lr_image)
        lr_image = norm01(lr_image)
        lr_image = tf.expand_dims(lr_image, axis=0)
        sr_image = model.predict(lr_image)[0]
        
        hr_image = read_image(ls_labels[i])
        hr_image = hr_image[k:-k, k:-k]
        hr_image = rgb2ycbcr(hr_image)
        hr_image = norm01(hr_image)
        

        sum_psnr += PSNR(hr_image, sr_image, max_val=1).numpy()

    return (sum_psnr / len(ls_data))