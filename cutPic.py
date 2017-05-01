#coding=utf-8
try:
    import Image
except ImportError:
    from PIL import Image
#import pytesseract
from PIL import ImageEnhance  
from PIL import ImageFilter  

import picPreHandle
#纵向切割，依据X轴的投影，将图片切割为4张图片，并返回切割点的坐标
def Cut_X(im):
    Image_Value = picPreHandle.Caculate_X(im)
    X_value=[] 
    List0=[]
    List1=[]
    ListRow0=[]
    ListRow1=[]
    for i in range(len(Image_Value)):
        if Image_Value[i] ==0 and len(ListRow1)==0: #数字左侧的空白列
            ListRow0.append(i)
        elif Image_Value[i] ==0 and len(ListRow1)>0: #数字右侧的空白列
            List1.append(ListRow1)
            ListRow1=[]
            ListRow0.append(i)
        elif Image_Value[i] >0 and len(ListRow0)>0 : #数字列
            List0.append(ListRow0)
            ListRow0=[]
            ListRow1.append(i)
        elif Image_Value[i] >0 and len(ListRow0)==0: #数字列
            ListRow1.append(i)
    if len(List1)==1 : #如果只有1个数字右侧的空白列，放弃切割
        for i in range(4):
            X_value.append(1+12*i)#
            X_value.append(12*i+12)           
    elif len(List1)==2 :    #如果只有2个数字右侧的空白列，放弃切割    
        for i in range(4):
            X_value.append(1+12*i)#
            X_value.append(12*i+12) 
    elif len(List1)==3 : #如果有3个数字右侧的空白列，将数字列中最长的那段值进行拆分，拆分点在X轴投影的大于第五位后的第一个最低点。
        Max_index = Max_Index(List1)
        for i in range(len(List1)):
            if i == Max_index:
                #
                index = Cut_Two(List1[i],Image_Value)
                X_value.append(List1[i][0])
                X_value.append(List1[i][index])
                X_value.append(List1[i][(index+1)])
                X_value.append(List1[i][(len(List1[i])-1)])
            else:
                X_value.append(List1[i][0])
                X_value.append(List1[i][(len(List1[i])-1)])
    elif len(List1)==4 :#4个空白列
        for i in range(len(List1)):
            X_value.append(List1[i][0])
            X_value.append(List1[i][(len(List1[i])-1)])
    elif len(List1)==5 :#如果有5个数字右侧的空白列，取长度最长的4段。
        Min_index = Min_Index(List1)
        for i in range(len(List1)):
            if i != Min_index:
                X_value.append(List1[i][0])
                X_value.append(List1[i][(len(List1[i])-1)])
    elif len(List1)>5 :#大于5个直接放弃切割
        for i in range(4):
            X_value.append(1+12*i)#############
            X_value.append(12*i+12)            
    return X_value
 
#返回矩阵各行最大值位置的函数，以便找到有颜色的列中X轴投影最大的地方
def Max_Index(List1):
    Max = 0
    Max_index=0
    for i in range(len(List1)):
        if len(List1[i])>Max:
            Max=len(List1[i])
            Max_index=i
    return Max_index
 
#返回矩阵各行最小值位置的函数，以便找到有颜色的列中X轴投影最小的地方
def Min_Index(List1):
    Min = 50
    Min_index=0
    for i in range(len(List1)):
        if len(List1[i]): 
            Min=len(List1[i])
            Min_index=i           
    return Min_index
 
#分割两个紧挨的数字
def Cut_Two(ListRow,Image_Value):
    index = 0
    start = 0
    if len(ListRow)>=15:
        start = 3
    for i in range((1+start),(len(ListRow)-1)):
        if Image_Value[ListRow[i]]<= Image_Value[ListRow[(i+1)]] and Image_Value[ListRow[i]]<=2:#
            index = i
            break
       
    return index
 
#横向切割 4张图片，4次投影，并返回切割点的坐标
def Cut_Y(im):
    Y_value=[]
    Image_Value=[]
    Cut_Xs=Cut_X(im)
    for k in range(4):
        Image_Value=[]
        for j in range(im.size[1]):
            X_pixel=0
            for i in range(Cut_Xs[(2*k)],(Cut_Xs[(2*k+1)]+1)):
                if im.getpixel((i,j))==0:
                    X_pixel = X_pixel+1               
            Image_Value.append(X_pixel)
        for i in range(len(Image_Value)):
            if Image_Value[i]>0:
                Y_value.append(i)
                break
        for i in range((len(Image_Value)-1),0,(-1)):
            if Image_Value[i]>0:
                Y_value.append(i)
                break
           
    return Y_value
    
    
