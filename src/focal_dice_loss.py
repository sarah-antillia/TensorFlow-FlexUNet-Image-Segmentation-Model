# 2026/09/23 Toshiyuki Arai
#
# focal_dice_loss.py

import tensorflow as tf
from tensorflow.keras import backend as K

# Please refer to Custom losses
# in https://www.tensorflow.org/guide/keras/training_with_built_in_methods

# Custom loss function 
class CategoricalFocalDiceLoss(tf.keras.losses.Loss):
    def __init__(self, epsilon=1e-6, name="categorical_focal_dice_loss"):
        super().__init__(name=name)
        self.epsilon = epsilon
 
    def dice_loss_multiclass(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        intersection = K.sum(y_true * y_pred, axis=-1)
        sum_ = K.sum(y_true + y_pred, axis=-1)
        dice = (2. * intersection + self.epsilon) / (sum_ + self.epsilon)
        return 1.0 - K.mean(dice)
   
    def call(self, y_true, y_pred):
        # Mixed loss = cce + dice
        cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        dice = self.dice_loss_multiclass(y_true, y_pred)
        return cce + dice

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


