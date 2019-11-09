from keras.datasets import cifar10

(x_train, y_train), (x_test,y_test) = cifar10.load_data()


print(x_train.shape)

x_test = x_test/255
x_train = x_train/255



from keras.utils import to_categorical

y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)

from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D,Flatten

model = Sequential()

# image:32*32*3
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(32,32,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

### hidden layer
# 128,256,512

model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

print(model.summary())

model.fit(x_train,y_cat_train,verbose=1,epochs=10)

model.metrics_names

model.evaluate(x_test,y_cat_test)

from sklearn.metrics import classification_report

prediction = model.predict_classes(x_test)

print(classification_report(y_test,prediction))
