import numpy as np
import cv2 as cv
import mnist
from operator import itemgetter


def measure(image):
    blur = cv.GaussianBlur(image, (5, 5), 0)
    cv.imwrite("./aa/1.jpg", blur)
    gray = cv.cvtColor(blur, cv.COLOR_RGB2GRAY)
    cv.imwrite("./aa/2.jpg", gray)
    _, out = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    cv.imwrite("./aa/3.jpg", out)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    # ------------------------------------------------------------------------------------
    # 最重要的调参位置 --- (71,71) --- 要使用奇数 --- 根据手写的图片调节
    # ------------------------------------------------------------------------------------

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (111,111))
    erode = cv.dilate(thresh, kernel)
    cv.imwrite("./aa/4.jpg", erode)
    clone_image, contours, hierarchy = cv.findContours(erode, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    all = []
    for i, contour in enumerate(contours):
        x1, y1, w1, h1 = cv.boundingRect(contour)
        cv.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
        img = out[y1:(y1 + h1), x1:(x1 + w1)]
        clone_image, contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # ------------------------------------------------------------------------------------
        #
        # 识别两位数字的主要思想 --->
        # 使用 --- 使用形态学转换 --- 膨胀 --- 去掉两个数字之间的间隔 --- 得到两个数字的轮廓，外包矩形 ---
        # --- 在小矩形内部继续找数字的轮廓 --- 识别出两个数字 --- 这两个数字识别的顺序不定，可能是先识别到前面的数字
        # 也可能先识别到后面的数字，所以对两个数字的位子进行判断 --- r得到正确的顺序 --- 在图像上显示正确的数字
        #
        # 简单说就是在大轮廓中找到小的轮廓，再对小的轮廓进行处理，得到正确的数字 ---
        # x, y, w, h = cv.boundingRect(contour) --- 得到大矩形内部小矩形的四个参数 --- 左上角的横坐标，纵坐标，宽，高
        # cv.rectangle --- 绘制小矩形
        # if h > 50: --- 对小矩形进行一个判断，达到一定大小的时候，才认为矩形框里面确实是存在数字，
        # result = img[y:(y + h), x:(x + w)] --- 得到小矩形内部的图像
        # constant = cv.copyMakeBorder --- 对小矩形加一个轮廓 --- 原因是训练数据集的图片中的数字都在中间的位置
        # cv.imwrite --- 将处理后的图片保存
        # predict = mnist.recognition(_dir) --- 识别得到数字
        # cv.putText --- 绘制通过神经网络得到的数字
        # res.append([predict, x]) --- 将得到的预测数字和小框左上角的横坐标保存在res中 ---
        # 之后 --- 一个大矩形循环后 --- 用于判断识别的数字是在第一个位置还是在第二个位置
        # cv.imshow('rect', image) --- 显示的作用
        #
        # ------------------------------------------------------------------------------------

        res = []
        for j, contour in enumerate(contours):
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(image, (x1+x, y1+y), (x1+x+w, y1+y+h), (0, 0, 255), 2)
            if h > 50:
                result = img[y:(y + h), x:(x + w)]

                black = [0, 0, 0]
                constant = cv.copyMakeBorder(result, 40, 40, 40, 40, cv.BORDER_CONSTANT, value=black)

                _dir = './data/' + str(j) + '.png'
                cv.imwrite('./data/' + str(j) + '.png', constant)

                predict = mnist.recognition(_dir)
                text = "{}".format(predict[0])
                cv.putText(image, text, (int(x1+x), int(y1+y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                res.append([predict, x])
                cv.imshow('rect', image)
                cv.waitKey(0)
        print(res)

        # ------------------------------------------------------------------------------------
        #
        # 下面的代码用于判断一个大矩形中识别的数字哪一个是在第一个位置，哪一个是在第二个位置
        # if len(res) == 1: --- 这个用于处理单个数字 --- 如果是单个数字直接绘制识别的数字即可 ---
        # all.append([x1, y1, str(res[0][0][0])]) --- all --- 用于存放所有的数字（大矩形里面的数字，一位或者两位）
        # --- 这个用于最后的排序，因为大矩形框的识别也不是按照顺序识别的 --- 所以需要判断 ---
        # if len(res) == 2: --- 这个用于处理两位数字 ---
        # res类型 --- [[array([4]), 35], [array([6]), 148]] --- [array([4]), 35] --- 第一个小矩形 --- 内部数字是4
        # --- [array([6]), 148] --- 第二个小矩形 --- 内部数字是6
        # res[0][1] --- 35， res[1][1] --- 148，可以判断出4所在的小矩形在前面，6所在的小矩形在后面
        # --- 于是这样得到了大矩形里面的数字是 --- 46
        # 同理 --- 第一个识别个位数的小矩形，第二次得到十位数的小矩形 --- 这样的也可以按照正常的顺序排列
        # text = '' --- 初始化一个存放字符的变量
        # res --- [[array([4]), 35], [array([6]), 148]]
        # res[0] --- [array([4]), 35]
        # res[0][0] --- [4]
        # res[0][0][0] --- 4
        #
        # res[1][0][0] --- 得到的是数字，但是要转成字符串的形式 --- str()
        #
        # ------------------------------------------------------------------------------------

        print(res)
        if len(res) == 1:
            cv.putText(image, str(res[0][0][0]), (int(x1), int(y1-30)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            all.append([x1, y1, str(res[0][0][0])])
        if len(res) == 2:
            if res[0][1] > res[1][1]:
                text = ''
                text += str(res[1][0][0])
                text += str(res[0][0][0])
                print(text)
                cv.putText(image, text, (int(x1), int(y1 - 30)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                all.append([x1, y1, text])
            else:
                text = ''
                text += str(res[0][0][0])
                text += str(res[1][0][0])
                print(text)
                cv.putText(image, text, (int(x1), int(y1 - 30)), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                all.append([x1, y1, text])

    # ------------------------------------------------------------------------------------
    #
    # 两位数字的保存 --- 可以保存一行数字，或者一列数字，两行数字现在还没有想到合适的算法
    # 混乱排序的数字 --- [[1300, 173, '46'], [982, 159, '23'], [721, 151, '11'], [393, 111, '7'], [18, 107, '10']]
    # --- 拿出一个解释 --- [1300, 173, '46'] --- 第一个是大轮廓的横坐标，第二个是大轮廓的纵坐标，第三个是大轮廓里面识别出的数字
    # 顺序排列的数字 --- [[18, 107, '10'], [393, 111, '7'], [721, 151, '11'], [982, 159, '23'], [1300, 173, '46']]
    # sorted(all, key=(lambda x: [x[0], x[1]])) --- 排序 --- 以第一个位置的数据为主关键字，第二个位置为辅助关键字
    # --- 这个是按照X的大小来排序 --- 横轴较小的排在前面，也可以按照Y轴的大小进行排序 --- 用于处理竖排的数字
    # 处理之后的数字保存在txt文件中，在txt中储存的是字符，所以要转换数据类型，然后在后面加入一个换行符，这样出来的格式比较清晰
    # with open('./data/data.txt', 'w') as f: --- 打开txt文件，
    # f.write --- 写入数据
    #
    # ------------------------------------------------------------------------------------

    print(all)
    all = sorted(all, key=(lambda x: [x[0], x[1]]))
    print(all)
    with open('./data/data.txt', 'w') as f:
        for i in range(0, len(all)):
            f.write(str(all[i][2]) + "\n")


src = cv.imread('./data/test10.jpeg')
measure(src)
cv.waitKey(0)
cv.destroyAllWindows()

