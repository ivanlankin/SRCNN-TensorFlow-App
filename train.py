import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from utils.dataset import dataset
from utils.common import PSNR
from model import SRCNN

def train_mod(steps=100000, batch_size=128, architecture="915", ckpt_dir="checkpoint/SRCNN915", save_every=100, save_best_only=0):
    model_path = ckpt_dir + f"SRCNN-{architecture}.h5"

# Инициализация датасета
    # путь до папки с изображениями для датасета
    dataset_dir = "dataset"
    # размер изображения низкого разрешения
    lr_crop_size = 33
    # высокого разрешения
    hr_crop_size = 21 
    # для каждой модели разные размеры
    if architecture == "935":
        hr_crop_size = 19
    elif architecture == "955":
        hr_crop_size = 17
    # датасет с обучаемых изображений
    train_set = dataset(dataset_dir, "train")
    train_set.generate(lr_crop_size, hr_crop_size)
    train_set.load_data()
    # датасет изображений с высоким разрешением
    valid_set = dataset(dataset_dir, "validation")#
    valid_set.generate(lr_crop_size, hr_crop_size)
    valid_set.load_data()

# обучение

    # создаем модель с нужной архитектурой
    srcnn = SRCNN(architecture)
    srcnn.setup(optimizer=Adam(learning_rate=2e-5),
                loss=MeanSquaredError(),
                model_path=model_path,
                metric=PSNR)
    # запускаем обучение с путём для сохранения модели
    srcnn.load_checkpoint(ckpt_dir)
    srcnn.train(train_set, valid_set, 
                steps=steps, batch_size=batch_size,
                save_best_only=save_best_only, 
                save_every=save_every)
print("Input steps(int), batch size(int), architecture(string), save step(int), save best only(bool, 0/1)")
steps = int(input())
batch_size = int(input())
architecture = input()
ckpt_dir=f"checkpoint/SRCNN{architecture}"
save_every = int(input())
save_best_only = int(input())
train_mod(steps, batch_size, architecture, ckpt_dir, save_every, save_best_only)