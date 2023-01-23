import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import load_model
from keras import backend as keras


def dice_coef(y_true, y_pred):
    y_true_f = keras.flatten(y_true)
    y_pred_f = keras.flatten(y_pred)
    intersection = keras.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (keras.sum(y_true_f) + keras.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


model = load_model('/Users/serhatsezn/Desktop/covidPro/model.h5', custom_objects={'dice_coef_loss':                   
dice_coef_loss, 'dice_coef': dice_coef})

filePaths = [#"/Users/serhatsezn/Desktop/covidPro/covid_1.png",
            "/Users/serhatsezn/Desktop/covidPro/covid_6.png",
            #"/Users/serhatsezn/Desktop/covidPro/covid_2.png", 
            #"/Users/serhatsezn/Desktop/covidPro/covid_3.png"
            ]

X_shape = 512
for file in filePaths: 
    x_im = cv2.resize(cv2.imread(file),(X_shape,X_shape))[:,:,0]
    op = model.predict((x_im.reshape(1, 512, 512, 1)-127.0)/127.0)
    plt.imshow(x_im, cmap="bone", label="Input Image")
    plt.title("Input")
    plt.show()
  
    img = cv2.resize(cv2.imread(file), (512, 512))
    colored_img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
    alpha = 0.3
    result = cv2.addWeighted(img, alpha, colored_img, alpha, 0)
    percentage = round((np.count_nonzero(result) / (1024 * 1024)) * 100, 4)
    print("Covid percentage : %",percentage)

    plt.imshow(x_im, cmap="bone", label="Output Image")
    plt.imshow(op.reshape(512, 512), alpha=0.5, cmap="jet")
  
    plt.title("Output Percentage: " + str(percentage))
    plt.show()


