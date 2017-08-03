import tensorflow as tf  
import numpy as np  
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data  
  
def weight_variable(shape):  
    return tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=0.1), name='weight')  
  
def bias_variable(shape):  
    return tf.Variable(initial_value=tf.constant(value=0.1, shape=shape), name='bias')  
      
def conv_2d(x, w):  
    return tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding="SAME")  
      
def max_pool_2x2(x):  
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #!!!!!!  
def evaluate(y, y_):  
    y = tf.arg_max(input=y, dimension=1)  
    y_ = tf.arg_max(input=y_, dimension=1)  
    return tf.reduce_mean(input_tensor=tf.cast(tf.equal(y, y_), tf.float32))      
def showLayer(layerOUT,id=0,showShape=[],title=""):
    w,h,channel=np.shape(layerOUT[id])
    plt.figure(title)
    for i in range(channel):
            plt.subplot(showShape[0],showShape[1],i+1) 
            plt.imshow(layerOUT[id,:,:,i])  
    plt.suptitle(title)
    plt.show()
       
def test_cnn(batch_size=50, lr=0.0001, num_iter=20000):   
    dataset = input_data.read_data_sets(train_dir='MNIST_data/', one_hot=True)  
    Image=np.random.rand(28,28)
    Image=tf.reshape(Image,[-1,28,28,1])

    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='images') #后面的卷积操作输入参数必须为‘float32’或者‘float64’  
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='labels')  
    
    w_conv1 = weight_variable(shape=[5, 5, 1, 32])  #作为filter,卷积核，在1通道5*5图像上施加32个卷积核  
    b_conv1 = bias_variable(shape=[32])  
    reshape_x = tf.reshape(x, shape=[-1, 28, 28, 1])  #省略的形式区别于占位符!!!!!!  
    conv1_out = tf.nn.relu(conv_2d(reshape_x, w_conv1)+b_conv1)  
    #一般的全连接网络是 ωx+b,卷积网络则是conv2d(x)+b，得到32幅28×28的图像，也可以看做32通道    
    pool1_out = max_pool_2x2(conv1_out) #经过池化层,输出一幅14*14 的32 通道图像 
    
    #输入层(28×28 一通道图像)→卷积层#1(输出一幅32通道28×28图像)→池化层#1(输出一幅32通道的14×14图像)→

     #第二个卷积层
    w_conv2 = weight_variable(shape=[5, 5, 32, 64]) #第二个卷积层大小，施加在池化层输出(一幅32通道的14×14图像)
    b_conv2 = bias_variable(shape=[64])  
    conv2_out = tf.nn.relu(conv_2d(pool1_out, w_conv2)+b_conv2) # 输出一幅14×14 64通道的图像
    pool2_out = max_pool_2x2(conv2_out)  #池化层#2：输出一幅64通道7×7图像
    # 卷积层#2(输出64通道14×14的图像)→池化层#2(输出64通道7×7图像)→全连接层  
    #全连接层
    full_connected_in = tf.reshape(pool2_out, shape=[-1, 7*7*64])  #一维变量
    w_full_connected = weight_variable(shape=[7*7*64, 1024])  #好多变量啊3211264+1024个
    b_full_connected = bias_variable(shape=[1024])  
    full_connected_out1 = tf.nn.relu(tf.matmul(full_connected_in, w_full_connected)+b_full_connected)  
    dropout_prob = tf.placeholder(dtype=tf.float32, name='dropout_probability')  
    full_connected_out = tf.nn.dropout(x=full_connected_out1, keep_prob=dropout_prob) #drop out防止过拟合  
    #全连接层输出1*1024
    #到了输出层
    w_softmax = weight_variable(shape=[1024, 10]) #隐含层与输出层的权值矩阵 
    b_softmax = bias_variable(shape=[10])      
    softmax_in = tf.matmul(full_connected_out, w_softmax)+b_softmax  
    softmax_out = tf.nn.softmax(logits=softmax_in, name='softmax_layer')  
    #定义损失函数
    Loss = tf.nn.softmax_cross_entropy_with_logits(logits=softmax_in, labels=y)  
    #定义训练方法
    Step_train = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=Loss)  
    
    accuracy = evaluate(y, softmax_out) #在测试数据集上评估算法的准确率      
    
    initialized_variables = tf.initialize_all_variables()  
      
    print('Start to train the convolutional neural network......')  
    sess = tf.Session()  
    sess.run(fetches=initialized_variables)  
    for iter in range(num_iter):  
        batch = dataset.train.next_batch(batch_size=batch_size)  
        #batch[0]-[50,28,28,32]
        sess.run(fetches=Step_train, feed_dict={x:batch[0], y:batch[1], dropout_prob:0.5})  

        #conv1OUT=sess.run(conv1_out,feed_dict={x:batch[0]})
        if (iter+1)%100==0:  #计算在当前训练块上的准确率 
            if 0:
                showLayer(sess.run(conv1_out,feed_dict={x:batch[0]}),2,[4,8],"The output of convolutional layer 1" ) 
                showLayer(sess.run(pool1_out,feed_dict={x:batch[0]}),2,[4,8],"The output of maxpool layer 1" ) 
                showLayer(sess.run(conv2_out,feed_dict={x:batch[0]}),2,[8,8],"The output of convolutional layer 2" ) 
                showLayer(sess.run(pool2_out,feed_dict={x:batch[0]}),2,[8,8],"The output of maxpool layer 2" ) 

            Accuracy = sess.run(fetches=accuracy, feed_dict={x:batch[0], y:batch[1], dropout_prob:1})  
            print('Iter num %d ,the train accuracy is %.3f' % (iter+1, Accuracy))  
              
    Accuracy = sess.run(fetches=accuracy, feed_dict={x:dataset.test.images, y:dataset.test.labels, dropout_prob:1})  
    sess.close()  
    print('Train process finished, the best accuracy is %.3f' % Accuracy)  
   
if __name__ == '__main__':  
    test_cnn()  