from tensorflow.keras import layers, Model

filter_num = [64, 128, 256, 512, 1024]

def unet_model(input_data):

    inputs = layers.Input(shape=input_data)

    #contracting path
    x1 = layers.Conv2D(filter_num[0], 3, activation = 'relu', padding = 'same')(inputs)
    x1 = layers.Conv2D(filter_num[0], 3, activation = 'relu',  padding = 'same')(x1)
    out1 = layers.MaxPooling2D((2,2))(x1)

    x2 = layers.Conv2D(filter_num[1], 3, activation = 'relu',  padding = 'same')(out1)
    x2 = layers.Conv2D(filter_num[1], 3, activation = 'relu',  padding = 'same')(x2)
    out2 = layers.MaxPooling2D((2,2))(x2)

    x3 = layers.Conv2D(filter_num[2], 3, activation = 'relu',  padding = 'same')(out2)
    x3 = layers.Conv2D(filter_num[2], 3, activation = 'relu',  padding = 'same')(x3)
    out3 = layers.MaxPooling2D((2,2))(x3)

    x4 = layers.Conv2D(filter_num[3], 3, activation = 'relu',  padding = 'same')(out3)
    x4 = layers.Conv2D(filter_num[3], 3, activation = 'relu',  padding = 'same')(x4)
    out4 = layers.MaxPooling2D((2,2))(x4)

    x5 = layers.Conv2D(filter_num[4], 3, activation = 'relu',  padding = 'same')(out4)
    x5 = layers.Conv2D(filter_num[4], 3, activation = 'relu',  padding = 'same')(x5)
    
    #expansive path
    y1 = layers.Concatenate([layers.Conv2DTranspose(filter_num[3], 3)(x5), out4])
    y1 = layers.Conv2D(filter_num[3],3, activation = 'relu', padding = 'same')(y1)
    up1 = layers.Conv2D(filter_num[3],3, activation = 'relu', padding = 'same')(y1)

    y2 = layers.Concatenate([layers.Conv2DTranspose(filter_num[2], 3)(up1), out3])
    y2 = layers.Conv2D(filter_num[3],3, activation = 'relu', padding = 'same')(y2)
    up2 = layers.Conv2D(filter_num[3],3, activation = 'relu', padding = 'same')(y2)

    y3 = layers.Concatenate([layers.Conv2DTranspose(filter_num[1], 3)(up2), out2])
    y3 = layers.Conv2D(filter_num[3],3, activation = 'relu', padding = 'same')(y3)
    up3 = layers.Conv2D(filter_num[3],3, activation = 'relu', padding = 'same')(y3)

    y4 = layers.Concatenate([layers.Conv2DTranspose(filter_num[0], 3)(up3), out1])
    y4 = layers.Conv2D(filter_num[3],3, activation = 'relu', padding = 'same')(y4)
    y4 = layers.Conv2D(filter_num[3],3, activation = 'relu', padding = 'same')(y4)

    output = layers.Conv2D(2, 1, activation = 'sigmoid', padding='same')(y4)

    model = Model(inputs = inputs, outputs = output)

    return model


