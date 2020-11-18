#!/usr/bin/python3
#coding=utf-8
# 从python 在一个py文件中调用另一个文件夹下py文件模块
import os
import sys

current_dir = os.getcwd()  # obtain work dir
sys.path.append(current_dir)  # add work dir to sys path

# 不显示警告信息
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

# from config import data_config

image_size = 64
labels_path = "F:\\ShuffleNet_tensorflow\\dataset\\tiny_imagenet_200\\wnids.txt"
#labels_path = "F:\\ShuffleNet_tensorflow\\labels.txt"
tf_save_path = "F:\\train2.tfrecords"
image_path = "F:\\ShuffleNet_tensorflow\\dataset\\tiny_imagenet_200\\train\\"
image_1st_path = "F:\\ShuffleNet_tensorflow\\val\\"  # 从原数据集(image_path)抽取的图像为生成test数据集做图片数据准备
tf_test_path = "F:\\val2.tfrecords"  # 生成的test集路径


def read_label_name(labels_path):

    """function:读取样本分类文件夹的名字
    param:labels_path->表示标签文本的路径"""

    print("/****************************/")
    print(" read label txt ")
    # 读取分类标签文本
    f = open(labels_path)  # 通过文本的方式读取分类文件夹名称

    class_id_cnt = 0  # 分类计数

    classes_read = []  # 定义变长数组，python的数组也是从 0 开始的
    print("\nread sample:")
    while True:
        line = f.readline()
        if line:
            class_id_cnt = class_id_cnt + 1  # 记录分类总数
            line = line.strip()
            classes_read.append(line)  # 给变长数组添加分类的文件夹名字
            print(class_id_cnt, ")", "--", classes_read[class_id_cnt - 1])
        else:
            break
    f.close()

    return classes_read, class_id_cnt


def Whitening(image, is_flip, image_size):
    """function:
    #功能：样本扩增技术，左右镜像翻转图片。
    #且进行图像白化操作（PCA）。
    # image 原图。
    # images size :为原图缩放的尺寸。
    # is_flip：是否进行图像的左右镜像翻转。"""

    image_reg = image

    if is_flip:  # 水平旋转
        image_reg = cv2.flip(image, 1)
    # resize 缩放原图
    img_dst_rgb = cv2.resize(image_reg, (image_size, image_size), interpolation=cv2.INTER_AREA)
    # 分别获得RGB三通道的 均值mean 和 标准差 std。
    mean0 = np.mean(img_dst_rgb[:, :, 0])
    std0 = np.std(img_dst_rgb[:, :, 0])
    mean1 = np.mean(img_dst_rgb[:, :, 1])
    std1 = np.std(img_dst_rgb[:, :, 1])
    mean2 = np.mean(img_dst_rgb[:, :, 2])
    std2 = np.std(img_dst_rgb[:, :, 2])

    img_dst = np.zeros([image_size, image_size, 3], np.float32)  # 白化处理后的数组

    for k in range(image_size):
        for m in range(image_size):
            img_dst[k, m, 0] = (img_dst_rgb[k, m, 0] - mean0) / std0
            img_dst[k, m, 1] = (img_dst_rgb[k, m, 1] - mean1) / std1
            img_dst[k, m, 2] = (img_dst_rgb[k, m, 2] - mean2) / std2

    return img_dst  # 返回白化处理的数组


def write_tfrecord(choose, tf_path, image_path, classes_read, choose_model):
    """tfrecord文件写入
    :param:tf_path->表示写入的tfrecord文件的路径
    :param:images_path->表示原图文件夹路径
    :param:classes_read->表示分类文件夹名字
    :param:choose_model->如果其值为2，就不进行数据增广
    :param:choose->表示选择训练或则测试"""

    # function：生成 TfRecords -- 包括样本及对应的标签号
    writer = tf.python_io.TFRecordWriter(tf_path)  # 要生成的tfrecords文件

    picture_cnt = 0  # 样本计数
    classes_cnt = 0  # 类别计数

    for index, name in enumerate(classes_read):
        if choose:
            class_path = image_path + name + "\\images\\"  # 原图文件夹路径

        else:
            class_path = image_path + name + '\\'
        print('class_path', class_path)
        classes_cnt = classes_cnt + 1
        print(" the", classes_cnt, "class")
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name  # 每个图片的地址
            # opencv 显示
            img_cv = cv2.imread(img_path)  # 图片保存路径必须全英文

            #           if (picture_cnt % 120 == 0):  # 每隔120张图片进行显示一次
            #               cv2.namedWindow("image_tfrecords", 0)
            #               cv2.imshow("image_tfrecords", img_cv)
            #               cv2.waitKey(1)

  #          img_dst = Whitening(img_cv, False,  image_size)  # 白化处理

            # 写入tfrecords文件
            img_raw = img_cv.tobytes()  # 将图片转化为二进制格式
            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串
            picture_cnt = picture_cnt + 1  # 样本计数加1

            if choose_model != 2:  # 测试数据集不做镜像操作
                img_dst = Whitening(img_cv, True, image_size)  # 增广样本，左右镜像，白化处理
                img_raw = img_dst.tobytes()  # 将图片转化为二进制格式
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))  # example对象对label和image数据进行封装
                writer.write(example.SerializeToString())  # 序列化为字符串

                picture_cnt = picture_cnt + 1  # 样本计数加1

    writer.close()

    #    cv2.destroyAllWindows()  # 销毁opencv窗口
    print('\n')
    print('*'*100)
    print("\n2 )TfRecords create total of sample: ", picture_cnt)
    print('*'*100)
    print('\n')
    return picture_cnt


def random_catch_picture(classes_read):
    """用于从原训练集图中抽取一定数量的图片作为测试集
    :param:classes_read->表示分类文件夹名字
    :param:image_1st_path->表示从原数据集抽取的图像的路径,eg:image_1st_path"""

    # function:生成预训练样本
    set_sample_num = 1937  # 设置训练的每类样本的样本原图个数

    picture_cnt = 0  # 图像数量计数
    train_class = 0  # 样本分类计数
    sample_nums = 0  # 每个类中的数量

    for index, name in enumerate(classes_read):
#        class_path = image_path + name + '\\images\\'    # 每类样本文件夹路径
        class_path = image_path + name + "\\"
        one_class = 0  # 单类个训练样本计数
        train_class = train_class + 1  # 样本分类计数
        class_path1 = image_1st_path+name
        if not os.path.isdir(class_path1):
            os.mkdir(class_path1)
        print(" the", train_class, "class  smaple  success")
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name  # 每一个图片的地址
            # opencv的读取图像的路径必须全英文
            img_cv = cv2.imread(img_path)
            # 样本降采样，每隔10张图取一张图作为预训练样本
            if (picture_cnt % 5 == 0 and one_class < set_sample_num):
                one_class = one_class + 1
                out_file = image_1st_path + name + '/' + img_name
                cv2.imwrite(out_file, img_cv)  # 存储图像
                sample_nums = sample_nums + 1  # 记录抽取的样本总数
            # opencv显示
        #            cv2.waitKey(1)
        #            cv2.namedWindow("img", 0)
        #            cv2.imshow("img", img_cv)

            picture_cnt = picture_cnt + 1  #记录原图中样本总数

            # 当采集数量满足每类样本设置采集数据，跳出，进行另一类样本的采集
            if one_class >= set_sample_num:
                break

    #    cv2.destroyAllWindows()  # 销毁opencv显示窗口
    print('\n')
    print('*'*100)  
    print("The total number of images extracted from the original:", sample_nums)
    print("Record the total number of samples of the original image:", picture_cnt)
    print(" well done !")
    print('*'*100)  
    print('\n')
def create_tfrecord(type=None, test_dataset=None):

    """数据集生成
    1):当生成训练数据集时，改参数type为train
    2):当生成测试数据集时，该参数type为test
    3):如果已经存在测试数据集的图片，且和tiny_imagenet_200的train目录一样，如下：
            train/n123456/images/sfsf.jpg
    需要改写random_catch_picture函数提取图片的路径和写入write_tfrecord图片的路径"""


    if type == 'train':

        classes_read, class_id_cnt = read_label_name(labels_path)  # 读取标签文本

        picture_num = write_tfrecord(True, tf_save_path, image_path, classes_read, 2)


    if type == 'test':

        if test_dataset == None:
            classes_read, class_id_cnt = read_label_name(labels_path)  # 读取标签文本

 #           random_catch_picture(classes_read)  # 从原数据集(image_path)抽取的图像为生成test数据集做图片数据准备
#            choose_model=2
            picture_num = write_tfrecord(0, tf_test_path, image_1st_path, classes_read, 2)
        
        else:
            classes_read, class_id_cnt = read_label_name(labels_path)  # 读取标签文本
            picture_num = write_tfrecord(0, tf_test_path, image_1st_path, classes_read, 2)
            
    return picture_num, class_id_cnt




picture_cnt, class_id_cnt = create_tfrecord(type='test')




def read_and_decode(filename_queue):
    """function:读取生成的 tfrecords文件"""


    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    img = tf.decode_raw(features['img_raw'], tf.float32)  # 解码 二进制转为 float32
    img = tf.reshape(img, [image_size, image_size, 3])
    # 样本原图为 3通道
 #   img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int64)  # 获取标签号
    return img, label


# 加载tfrecords文件并进行文件解析


# # function:反向验证.tfrecords格式数据集的样本和标签对应关系， 生成对应的图像
with tf.Session() as sess:  # 开始一个会话

    filename_queue = tf.train.string_input_producer([tf_test_path])
    image2, labels2 = read_and_decode(filename_queue)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    coord = tf.train.Coordinator()  # 创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动QueueRunner

    Classes_Cnt = np.zeros([class_id_cnt], np.int32)  # 记录每个分类的样本个数

    for i in range(picture_cnt):
        example, class_num = sess.run([image2, labels2])  # 在会话中取出image和label
        #         if(i%10==0):
        #             print(example)
        #        if (i % 120 == 0):
        #            cv2.namedWindow("image_out", 0)
        #            cv2.imshow("image_out", example)
        #            cv2.waitKey(1)

        #         out_file= tf_out_path+str(i)+'_''Label_'+str(class_num)+'.jpg' # 存储和图像路径
        #         cv2.imwrite(out_file, example)#存储图像
        Classes_Cnt[class_num] = Classes_Cnt[class_num] + 1

    coord.request_stop()  # 关闭线程
    coord.join(threads)

    for i in range(class_id_cnt):
        print("the number of classification", i, " = ", Classes_Cnt[i], "sample")

    print("\n3-->TfRecords test conversion success ")

    #    cv2.destroyAllWindows()  # 销毁opencv窗口
    sess.close()
print("well done!")
