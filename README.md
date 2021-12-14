# COS429_final_project
Few Shot Image Classification


The Mini-ImageNet pkl objects can be found here:
https://www.kaggle.com/whitemoon/miniimagenet

The pretrained version of MobileNetV2 we used for some experiments can be downloaded in the following way:
```python
import tensorflow as tf
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
```
