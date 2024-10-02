import Unet.unet as unet

input_shape = (1,512,512)

unet_model = unet.unet_model(input_shape)
unet_model.summary()