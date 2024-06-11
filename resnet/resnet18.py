# tensorflow로 renset-18 구현하기

from tensorflow.keras import layers, Model

# resnet18
filter_size = [64, 128, 256, 512]
kernel_size = 3

def conv_block2(input, filter, stride, kernel):
        x = layers.Conv2D(filters=filter, kernel_size=kernel,strides=stride, padding = 'same')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters=filter, kernel_size=kernel,strides=stride, padding = 'same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

def residual_block(inputs, filter_size, check=True):
        
        if check:
                x = conv_block2(inputs, filter = filter_size, stride =1, kernel= 3)
                x = conv_block2(x,filter = filter_size, stride =1, kernel = 3)
                x = layers.Add()([inputs, x])
                x = layers.Activation('relu')(x)
        
        else:
             inputs = conv_block2(inputs, filter = filter_size, stride =1, kernel=1) 
             inputs = conv_block2(inputs, filter = filter_size, stride =2, kernel=3)

             x = conv_block2(inputs, filter = filter_size, stride =1,kernel= 3)
             x = conv_block2(x,filter = filter_size, stride =1,kernel = 3)
             x = layers.Add()([inputs, x])
             x = layers.Activation('relu')(x)   

        return x

def Resnet18(input_size, num_classes):

        input = layers.Input(shape=input_size)

        x = layers.Conv2D(filters=64, kernel_size = 7, strides=2)(input)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        x = residual_block(x, filter_size[0])
        x = residual_block(x, filter_size[1],False)
        x = residual_block(x, filter_size[2],False)
        x = residual_block(x, filter_size[3],False)

        output = layers.GlobalAveragePooling2D()(x)
        output = layers.Dense(num_classes, activation = 'softmax')(output)
        
        model = Model(inputs = input, outputs=output)

        return model


if __name__ == '__main__':
        resnet18 = Resnet18((224,224,1), 10)
        resnet18.summary()




        

        

