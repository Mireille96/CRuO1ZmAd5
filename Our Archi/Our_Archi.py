import tensorflow as tf
import glob
import os
import cv2
import numpy as np
from PIL import Image
#import ipdb
#import matplotlib.pyplot as plt

RGBimage_list = []
Grayimage_list = []
Temp_RGBimage_list = []
Temp_Grayimage_list = []
size = 200, 200
binSIZE=110
x = tf.placeholder(tf.float32, [None, 40000])
y_ = tf.placeholder(tf.int64, [None,200,200,3])
keep_prob=tf.placeholder(tf.float32)

def read_images_in_folder():
    imagePath = glob.glob('M:\\4th Year\\GP\\Seminar 4\\dr Islam\\Train weights from doctor Islam\\gray colorization\\rgb\\*.jpg')
    RGBimage_stack = np.array(np.array([np.array(cv2.imread(imagePath[i])) for i in range(len(imagePath))]))
    grayimage_stack = np.array(np.array([np.array(cv2.imread(imagePath[i],cv2.IMREAD_GRAYSCALE)) for i in range(len(imagePath))]))
    return RGBimage_stack,grayimage_stack
def resize_image(image_stack, hResize, wResize):
    im_resized_stack = np.array( [np.array(cv2.resize(img, (hResize, wResize), interpolation=cv2.INTER_CUBIC)) for img in image_stack]) 
    return im_resized_stack
def Read_GreyTest_Images_in_folder():
    imagePath = glob.glob('M:\\4th Year\\GP\\Seminar 4\\dr Islam\\Train weights from doctor Islam\\gray colorization\\test\\*.jpg')
    grayTestimage_stack = np.array( [np.array(cv2.imread(imagePath[i],cv2.IMREAD_GRAYSCALE)) for i in range(len(imagePath))] )
    print("read images")
    return grayTestimage_stack
def Read_RGBLabelsTest_Images_in_folder():
    imagePath = glob.glob('M:\\4th Year\\GP\\Seminar 4\\dr Islam\\Train weights from doctor Islam\\gray colorization\\test\\*.jpg')
    RGBTestimage_stack = np.array( [np.array(cv2.imread(imagePath[i])) for i in range(len(imagePath))] )
    return RGBTestimage_stack
def createColorVec():
    #vec =tf.placeholder(tf.float32,[None,2,1]
    vec = []
    for a in range(-110,110):#make range jump 55 instead?? and this would be my bin
        for b in range(-110,110):
            vec.append([a,b])
    vec=np.array(vec)
    return vec
def BIN(colors):
    colorVec =[]
    for i in range(0,len(colors)-1,binSIZE):
        colorVec.append(colors[i])
    colorVec=np.array(colorVec)
    return colorVec
def binarySearch(colorVec, pixelValue):
    #lower = 0
    #upper = len(colorVec)
    #while lower < upper:   # use < instead of <=
    #    x = lower + (upper - lower) 
    #    val = colorVec[x]
    #    if pixelValue[0] == val[0]:
    #        return x#get b
    #    elif pixelValue[0] > val[0]:
    #        if lower == x:   # these two are the actual lines
    #            break        # you're looking for
    #        lower = x
    #    elif pixelValue < val:
    #        upper = x
    return 1
def getPixelLabel(pixelValue,colorVec):#LESSAAAA
    for i in range(1,len(colorVec)):#BS
        label =binarySearch(colorVec,pixelValue)
    return label
def getLabels(Img,colorVec):
    labels = tf.Variable(tf.zeros([200,200],tf.int64))
    for i in range(0,len(Img.get_shape())-1):
        for j in range(0,len(Img.get_shape())-1):
            labels[i,j].assign(getPixelLabel([Img[i,j,1],Img[i,j,2]],colorVec))
    tf.reshape(labels,shape=(200,200))
    return labels#labels
def getColors(labels,colorVec):
    aChannel = labels
    bChannel = labels
    for i in range(0,len(Img.get_shape())):
        for j in range(0,len(Img.get_shape())):
            aChannel=colorVec[labels[i,j],0]
            bChannel=colorVec[labels[i,j],1]
    return aChannel,bChannel#colored a,b
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#def conv2d(x,W,padding):
#  return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding=padding)
def conv2d(x,W):
  return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def MaxPool2d(x):
  return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
def convolutional_neural_network(x):
    weights = {'W_conv1':weight_variable([5,5,1,64]),#pad
               'W_conv2':weight_variable([5,5,64,128]),#output size = (N-F)/stride +1 --- *N is dimension of input & * F is filter size
               'W_conv3':weight_variable([57,57,128,256]),#LESSA - max pool
               'W_conv4':weight_variable([29,29,256,512]),
               'W_conv5':weight_variable([5,5,512,512]),#pad
               'W_conv6':weight_variable([5,5,512,512]),#pad
               'W_conv7':weight_variable([5,5,512,512]),#pad
               'W_conv8':weight_variable([5,5,512,256]),#dilate
               'W_conv9':weight_variable([5,5,256,313])}#dilate
    biases = {'b_conv1':bias_variable([64]),
              'b_conv2':bias_variable([128]),
              'b_conv3':bias_variable([256]),
              'b_conv4':bias_variable([512]),
              'b_conv5':bias_variable([512]),
              'b_conv6':bias_variable([512]),
              'b_conv7':bias_variable([512]),
              'b_conv8':bias_variable([256]),
              'b_conv9':bias_variable([313])}
    x=(tf.reshape(x,shape=[-1,200,200,1])-128)/128
    #pad 1,5,6,7---dilate 8
    #conv1=tf.nn.relu(conv2d(x,weights['W_conv1'],'SAME') + biases['b_conv1'])
    #conv2=tf.nn.relu(conv2d(conv1,weights['W_conv2'],'VALID') + biases['b_conv2'])
    #conv3=tf.nn.relu(conv2d(conv2,weights['W_conv3'],'VALID') + biases['b_conv3'])
    #conv3=MaxPool2d(conv3)
    #conv4=tf.nn.relu(conv2d(conv3,weights['W_conv4'],'VALID') + biases['b_conv4'])
    #conv5=tf.nn.relu(conv2d(conv4,weights['W_conv5'],'SAME') + biases['b_conv5'])
    #conv6=tf.nn.relu(conv2d(conv5,weights['W_conv6'],'SAME') + biases['b_conv6'])
    #conv7=tf.nn.relu(conv2d(conv6,weights['W_conv7'],'SAME') + biases['b_conv7'])
    #conv8=tf.nn.relu(conv2d(conv7,weights['W_conv8'],'VALID') + biases['b_conv8'])
    #conv9=(conv2d(conv8,weights['W_conv9'],'VALID') + biases['b_conv9'])
    conv1=tf.nn.relu(conv2d(x,weights['W_conv1']) + biases['b_conv1'])
    conv2=tf.nn.relu(conv2d(conv1,weights['W_conv2']) + biases['b_conv2'])
    conv2=MaxPool2d(conv2)
    conv3=tf.nn.relu(conv2d(conv2,weights['W_conv3']) + biases['b_conv3'])
    conv3=MaxPool2d(conv3)
    conv4=tf.nn.relu(conv2d(conv3,weights['W_conv4']) + biases['b_conv4'])
    conv5=tf.nn.relu(conv2d(conv4,weights['W_conv5']) + biases['b_conv5'])
    conv6=tf.nn.relu(conv2d(conv5,weights['W_conv6']) + biases['b_conv6'])
    conv7=tf.nn.relu(conv2d(conv6,weights['W_conv7']) + biases['b_conv7'])
    conv8=tf.nn.relu(conv2d(conv7,weights['W_conv8']) + biases['b_conv8'])
    conv9=(conv2d(conv8,weights['W_conv9']) + biases['b_conv9'])
    output=(tf.nn.sigmoid(conv9))*255
    output = tf.image.resize_nearest_neighbor(output,[200,200])#make with depth 2 only walla eh ??
    return output
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
RGBimage_list, Grayimage_list = read_images_in_folder()
Grayimage_list = Grayimage_list.reshape([-1,40000])
unison_shuffled_copies(RGBimage_list,Grayimage_list)
colors = createColorVec()
colorVec = BIN(colors)#size 880
#colorBox = tf.placeholder([tf.tile(colorVec(i),3136) for i in range(1,len(colorVec)])#???repeat a vector along depth dimension
#colorBox = tf.reshape(colorBox,shape=[-1,56,56])#??
#call BIN fn
output = convolutional_neural_network(x)
#ipdb.set_trace()
p = tf.losses.softmax=(output)#ezzay a7ot ma3ah el output kaman??
prediction=tf.arg_max(p,3)
#CLASSIFY HERE (prediction = my labels), y_ should be converted to labels
labels = getLabels(y_,colorVec)
y=tf.Variable(labels,tf.int64)
y_ = tf.reshape(y_,shape=(200,200))
#tf.Print(prediction, [prediction], message="This is a: ")
#b =tf.add(prediction,prediction)
#prediction = cv2.resize(prediction, (1, 40000), interpolation=cv2.INTER_CUBIC)
#y_= cv2.resize(prediction, (1, 40000), interpolation=cv2.INTER_CUBIC)
#cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.subtract(prediction,y_) ** 2) ** 0.5)
cross_entropy = tf.reduce_mean(tf.pow(tf.to_float(tf.reduce_sum(tf.pow(tf.subtract(prediction,y_), 2))),0.5))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



def testNN():
    saver = tf.train.Saver()
    TempTestGray = Read_GreyTest_Images_in_folder().reshape([-1,40000])
    c = TempTestGray[0]
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver.restore(sess, "M:/4th Year/GP/Seminar 4/dr Islam/Train weights from doctor Islam/gray colorization/model/model.ckpt")   
        yy = sess.run(prediction,feed_dict = {x:TempTestGray,keep_prob: 0.5})
        result = np.floor(yy[0])
        a,b =getColors(result)
        res=TempTestGray+a+b
        cv2.imwrite('M:/4th Year/GP/Seminar 4/dr Islam/Train weights from doctor Islam/gray colorization/test result/res.jpg',res);
        cv2.imwrite('M:/4th Year/GP/Seminar 4/dr Islam/Train weights from doctor Islam/gray colorization/test result/input.jpg',TempTestGray.reshape([-1,200,200])[0])
        img = cv2.imread("M:/4th Year/GP/Seminar 4/dr Islam/Train weights from doctor Islam/gray colorization/test result/res.jpg")
        cv2.startWindowThread()
        cv2.namedWindow("Colored Image")
        cv2.imshow("Colored Image", img)
        cv2.waitKey(0)
        
def trainNN():
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        #saver.restore(sess, "M:/4th Year/GP/Seminar 4/dr Islam/Train weights from doctor Islam/gray colorization/model/model.ckpt")
        for epoch in range(300):
            epoch_loss = 0
            for i in range(int(324/10)):
                print("Batch Num ",i + 1)
                a, c = sess.run([train_step,cross_entropy],feed_dict={x: Grayimage_list[i*10:(i+1)*10], y_: RGBimage_list[i*10:(i+1)*10], keep_prob: 0.5})
                epoch_loss +=c
            print("epoch: ",epoch + 1, ",Loss: ",epoch_loss)
            save_path = saver.save(sess, "M:/4th Year/GP/Seminar 4/dr Islam/Train weights from doctor Islam/gray colorization/model/model.ckpt")


#testNN()
trainNN()
