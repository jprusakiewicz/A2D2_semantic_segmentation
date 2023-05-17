import numpy as np
from keras.backend import int_shape
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Add, BatchNormalization, Input, Activation, Concatenate, \
    ZeroPadding2D, Cropping2D


def res_unet(num_classes, filter_root=3, layers=5, input_shape=(151, 240, 3), activation='relu', batch_norm=True, final_activation='sigmoid'):
    inputs = Input(input_shape)
    inputs_padded = ZeroPadding2D(((0, 105), (0, 16)))(inputs)

    x = inputs_padded
    # Dictionary for long connections
    long_connection_store = {}

    # Down sampling
    for i in range(layers):
        out_channel = 2**i * filter_root
        print("out_channel downsampling: {}".format(out_channel))

        # Residual/Skip connection
        res = Conv2D(out_channel, kernel_size=1,
                     padding='same', use_bias=False, name="Identity{}_1".format(i))(x)

        # First Conv Block with Conv, BN and activation
        conv1 = Conv2D(out_channel, kernel_size=3, padding='same', name="Conv{}_1".format(i))(x)
        if batch_norm:
            conv1 = BatchNormalization(name="BN{}_1".format(i))(conv1)
        act1 = Activation(activation, name="Act{}_1".format(i))(conv1)

        # Second Conv block with Conv and BN only
        conv2 = Conv2D(out_channel, kernel_size=3, padding='same', name="Conv{}_2".format(i))(act1)
        if batch_norm:
            conv2 = BatchNormalization(name="BN{}_2".format(i))(conv2)

        resconnection = Add(name="Add{}_1".format(i))([res, conv2])

        act2 = Activation(activation, name="Act{}_2".format(i))(resconnection)

        # Max pooling
        if i < layers - 1:
            long_connection_store[str(i)] = act2
            x = MaxPooling2D(padding='same', name="MaxPooling{}_1".format(i))(act2)
        else:
            x = act2
    print("\n")
    # Upsampling
    for i in range(layers - 2, -1, -1):
        print("i upsampling: {}".format(i))

        out_channel = 2**(i) * filter_root
        print("out_channel upsampling: {}".format(out_channel))

        # long connection from down sampling path.
        long_connection = long_connection_store[str(i)]
        print("long_connection: {}".format(long_connection))

        up1 = UpSampling2D(name="UpSampling{}_1".format(i))(x)
        up_conv1 = Conv2D(out_channel, 2, activation='relu', padding='same', name="upsamplingConv{}_1".format(i))(up1)
        print("up_conv1: {}".format(up_conv1))

        crop_shape = get_crop_shape(int_shape(up_conv1), int_shape(long_connection))

        crop_connection = Cropping2D(cropping=crop_shape, name="upCrop{}_1".format(i))(long_connection)

        #  Concatenate.
        up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, crop_connection])

        #  Convolutions
        up_conv2 = Conv2D(out_channel, 3, padding='same', name="upConv{}_1".format(i))(up_conc)
        if batch_norm:
            up_conv2 = BatchNormalization(name="upBN{}_1".format(i))(up_conv2)
        up_act1 = Activation(activation, name="upAct{}_1".format(i))(up_conv2)

        up_conv2 = Conv2D(out_channel, 3, padding='same', name="upConv{}_2".format(i))(up_act1)
        if batch_norm:
            up_conv2 = BatchNormalization(name="upBN{}_2".format(i))(up_conv2)

        # Residual/Skip connection
        res = Conv2D(out_channel, kernel_size=1,
                     padding='same', use_bias=False, name="upIdentity{}_1".format(i))(up_conc)

        resconnection = Add(name="upAdd{}_1".format(i))([res, up_conv2])

        x = Activation(activation, name="upAct{}_2".format(i))(resconnection)

    # Final convolution
    output = Conv2D(num_classes, 1, padding='same',
                    activation=final_activation, name='output')(x)

    outputs_unpadded = Cropping2D(cropping=((0, 105), (0, 16)))(output)

    return Model(inputs, outputs=outputs_unpadded, name='Res-UNet')
#
# def res_unet(num_classes, filter_root=3, depth=5, input_shape=(151, 240, 3), activation='relu', batch_norm=False, final_activation='sigmoid'):
#     """
#     source: https://github.com/Nishanksingla/UNet-with-ResBlock/blob/master/resnet34_unet_model.py
#     Build UNet model with ResBlock.
#
#     Args:
#         filter_root (int): Number of filters to start with in first convolution.
#         depth (int): How deep to go in UNet i.e. how many down and up sampling you want to do in the model.
#                     Filter root and image size should be multiple of 2^depth.
#         num_classes (int, optional): How many classes in the output layer. Defaults to 2.
#         input_size (tuple, optional): Input image size. Defaults to (256, 256, 1).
#         activation (str, optional): activation to use in each convolution. Defaults to 'relu'.
#         batch_norm (bool, optional): To use Batch normaliztion or not. Defaults to True.
#         final_activation (str, optional): activation for output layer. Defaults to 'softmax'.
#
#     Returns:
#         obj: keras model object
#     """
#     inputs = Input(input_shape)
#
#     inputs_padded = ZeroPadding2D(((0, 105), (0, 16)))(inputs)
#
#     x = inputs_padded
#     # Dictionary for long connections
#     long_connection_store = {}
#
#     # Down sampling
#     for i in range(depth):
#         out_channel = 2**i * filter_root
#
#         # Residual/Skip connection
#         res = Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False, name="Identity{}_1".format(i))(x)
#
#         # First Conv Block with Conv, BN and activation
#         conv1 = Conv2D(out_channel, kernel_size=3, padding='same', name="Conv{}_1".format(i))(x)
#         if batch_norm:
#             conv1 = BatchNormalization(name="BN{}_1".format(i))(conv1)
#         act1 = Activation(activation, name="Act{}_1".format(i))(conv1)
#
#         # Second Conv block with Conv and BN only
#         conv2 = Conv2D(out_channel, kernel_size=3, padding='same', name="Conv{}_2".format(i))(act1)
#         if batch_norm:
#             conv2 = BatchNormalization(name="BN{}_2".format(i))(conv2)
#
#         resconnection = Add(name="Add{}_1".format(i))([res, conv2])
#
#         act2 = Activation(activation, name="Act{}_2".format(i))(resconnection)
#
#         # Max pooling
#         if i < depth - 1:
#             long_connection_store[str(i)] = act2
#             x = MaxPooling2D(padding='same', name="MaxPooling{}_1".format(i))(act2)
#         else:
#             x = act2
#
#     # Upsampling
#     for i in range(depth - 2, -1, -1):
#         out_channel = 2**(i) * filter_root
#
#         # long connection from down sampling path.
#         long_connection = long_connection_store[str(i)]
#
#         up1 = UpSampling2D(name="UpSampling{}_1".format(i))(x)
#         up_conv1 = Conv2D(out_channel, 2, activation='relu', padding='same', name="upConv{}_1".format(i))(up1)
#
#         #  Concatenate.
#         up_conc = Concatenate(axis=-1, name="upConcatenate{}_1".format(i))([up_conv1, long_connection])
#
#         #  Convolutions
#         up_conv2 = Conv2D(out_channel, 3, padding='same', name="upConv{}_2".format(i))(up_conc)
#         if batch_norm:
#             up_conv2 = BatchNormalization(name="upBN{}_1".format(i))(up_conv2)
#         up_act1 = Activation(activation, name="upAct{}_1".format(i))(up_conv2)
#
#         up_conv2 = Conv2D(out_channel, 3, padding='same', name="upConv{}_2_2".format(i))(up_act1)
#         if batch_norm:
#             up_conv2 = BatchNormalization(name="upBN{}_2".format(i))(up_conv2)
#
#         # Residual/Skip connection
#         res = Conv2D(out_channel, kernel_size=1, padding='same', use_bias=False, name="upIdentity{}_1".format(i))(up_conc)
#
#         resconnection = Add(name="upAdd{}_1".format(i))([res, up_conv2])
#
#         x = Activation(activation, name="upAct{}_2".format(i))(resconnection)
#
#     # Final convolution
#     output = Conv2D(num_classes, 1, padding='same', activation=final_activation, name='output')(x)
#
#     outputs_unpadded = Cropping2D(cropping=((0, 105), (0, 16)))(output)
#
#     return Model(inputs=inputs, outputs=outputs_unpadded, name='Res-UNet')


# This is not useful, when the input size is not the multiple of 2^layers.
# Dimensions of activation map from down path gets smaller than acitvation map after up sampling operation.
def get_crop_shape(target, source):
    # source is coming from down sampling path.
    # target is coming from up sampling operation.

    source_height_width = np.array(source[1:-1])
    target_height_widht = np.array(target[1:-1])

    diff = (source_height_width - target_height_widht).tolist()

    diff_tup = map(lambda x: (x//2, x//2) if x%2 == 0 else (x//2, x//2 + 1), diff)

    return tuple(diff_tup)