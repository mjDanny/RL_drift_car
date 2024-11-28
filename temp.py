import logging
import tensorflow as tf

# Проверка доступных устройств
gpus = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
# Настройка уровня логирования TensorFlow
tf.get_logger().setLevel(logging.ERROR)
# Если GPU доступны, настроим TensorFlow для их использования
if gpus:
    try:
        # Ограничим TensorFlow до использования только первой GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Ошибка возникнет, если не удастся настроить видимые устройства
        print(e)