#
# dice_coef for multi-classe 
#
import tensorflow as tf
import tensorflow.keras.backend as K

from ConfigParser import ConfigParser

# Please see:
# Multiclass segmentation for different loss functions(Dice loss, Focal loss, Total loss = (Summation of Dice and focal loss)) in Tensorflow
# https://medium.com/@mb16biswas/multiclass-segmentation-for-different-loss-functions-dice-loss-focal-loss-total-loss-summation-455178517cea

# https://gist.github.com/sohiniroych/68ce46adfae0400acc5fe833d96f6464#file-loss_functions-py

# https://stackoverflow.com/questions/61488732/how-calculate-the-dice-coefficient-for-multi-class-segmentation-task-using-pytho

# https://www.kaggle.com/code/mb16biswas/multiclass-segmentation-for-diff-loss-functions

# You may cutomize the following values for dice_coef_hybrid
# 

# 2026/09/25
class DiceCoefHybrid:
  ALPHA = 0
  BETA  = 0

  def set_weight_parameters(config_file):
    parser = ConfigParser(config_file)
    DiceCoefHybrid.ALPHA = parser.get(ConfigParser.MODEL, "hybrid_alpha", dvalue=1.5)
    DiceCoefHybrid.BETA  = parser.get(ConfigParser.MODEL, "hybrid_beta",  dvalue=0.5)
    print("=== dice_coef_hybrid: alpha ={} bet={}".format(DiceCoefHybrid.ALPHA, DiceCoefHybrid.BETA))
    if (DiceCoefHybrid.ALPHA + DiceCoefHybrid.BETA) != 2.0:
       raise Exception("DiceCoefHybrid: Error ALPHA + BETA !=1.0")

  def dice_coef_hybrid(y_true, y_pred): 
    alpha = DiceCoefHybrid.ALPHA
    beta  = DiceCoefHybrid.BETA
    print("=== dice_coef_hybrid: alpah = {}, beta = {}".format(alpha, beta))
    bg_fg_dice   = dice_coef_multiclass(y_true, y_pred, smooth=1)
    fg_only_dice = dice_coef_foreground(y_true, y_pred, epsilon=1e-6)
    mean_dice    = (bg_fg_dice * alpha + fg_only_dice * beta) / (alpha + beta)
    return mean_dice
  

# Dice_coef for multi-class segmentation
def dice_coef_multiclass(y_true, y_pred, smooth=1):
    """
    Dice coefficient for multi-class segmentation.
    Args:
        y_true: Ground truth tensor (one-hot encoded). Shape: (batch, height, width, num_classes)
        y_pred: Prediction tensor (probabilities). Shape: (batch, height, width, num_classes)
        smooth: Smoothing factor to avoid division by zero.
    Returns:
        Dice coefficient.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def dice_loss_multiclass(y_true, y_pred, smooth=1):
    """
    Dice loss, which can be minimized during training.
    """
    return 1 - dice_coef_multiclass(y_true, y_pred, smooth)

# 2026/09/23 Added: Toshiyuki Arai
"""
This metric calculates the Dice coefficient exclusively for the foreground, excluding the background.
This is a straightforward way to mitigate the class imbalance problem that occurs when background 
pixels occupy an overwhelmingly large portion of a mask image.
"""
def dice_coef_foreground(y_true, y_pred, epsilon=1e-6):
    """
    Args:
        y_true: Ground truth tensor (one-hot encoded). Shape: (batch, height, width, num_classes)
        y_pred: Prediction tensor (probabilities). Shape: (batch, height, width, num_classes)
    """
    y_true = tf.cast(y_true, tf.float32)
    # Get y_true and y_pred foregrounds by excludeing backgrounds (channel 0)
    y_true_foreground = y_true[..., 1:]
    y_pred_foreground = y_pred[..., 1:]
    
    intersection = K.sum(y_true_foreground * y_pred_foreground, axis=[1, 2])
    sum_ = K.sum(y_true_foreground + y_pred_foreground, axis=[1, 2])
    
    dice = (2. * intersection + epsilon) / (sum_ + epsilon)
    return dice


def dice_loss_foreground(y_true, y_pred, epsilon=1e-6):
    dice = dice_coef_foreground(y_true, y_pred, epsilon)
    return 1.0 - K.mean(dice)

# 2026/09/25 Added: Toshiyuki Arai
# 2026/09/26 Modified to calculate a weighted average 
"""
This "dice_coef_hybrid" metric function calculates a weighted average of "dice_coef_multiclass" and 
"dice_coef_foreground" using two weight parameters, alpha and beta. 
These parameters can be determined based on the pixel distribution of the foreground and background 
in a mask image.
"""
def dice_coef_hybrid(y_true, y_pred, alpha = 0, beta = 0): 
    if alpha == 0 and beta == 0:
      alpha = DiceCoefHybrid.ALPHA
      beta  = DiceCoefHybrid.BETA
    print("=== dice_coef_hybrid: alpah = {}, beta = {}".format(alpha, beta))
    bg_fg_dice   = dice_coef_multiclass(y_true, y_pred, smooth=1)
    fg_only_dice = dice_coef_foreground(y_true, y_pred, epsilon=1e-6)
    mean_dice    = (bg_fg_dice * alpha + fg_only_dice * beta) / (alpha + beta)
    return mean_dice
