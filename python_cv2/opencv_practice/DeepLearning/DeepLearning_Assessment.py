from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt 
import cv2

# loading data from mnist
(x_train, y_train), (x_test,y_test) = fashion_mnist.load_data()


#### visualization the image data ####
print(x_train.shape)
img = x_train[0]

#plt.imshow(img,'gray')
#plt.show()

#### preprocessing the image data ####

# 1. scale the value 0,255---> 0,1
#print(x_train.max())
x_train = x_train/255
x_test = x_test/255
#print(x_train.max())

# 2. reshape the data to 4 dim
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

# 3. use one-hot decode to change label format
from keras.utils.np_utils import to_categorical

cat_y_train = to_categorical(y_train,10)
cat_y_test = to_categorical(y_test,10)


#### Build the training model ####
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,Activation,MaxPool2D

model = Sequential()
input_shapes = (28,28,1)

# 1. Convolution Layer
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=input_shapes))
model.add(Activation('relu'))

# 2. Pooling Layer
model.add(MaxPool2D(pool_size=(2,2)))

# 3. Flatten 
model.add(Flatten())

# 4. Hidden Layer
model.add(Dense(128))
model.add(Activation('relu'))

# 5. 0-9 output Layer
model.add(Dense(10))
model.add(Activation('softmax'))

# 6. Compile the model
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

print(model.summary())


#### Training the model ####
model.fit(x_train,cat_y_train,batch_size=2,epochs=2)

#### Evaluating the model ####
model.metrics_names

model.evaluate(x_test,cat_y_test)

from sklearn.metrics import classification_report

prediction = model.predict_classes(x_test)

print(classification_report(y_test,prediction))

