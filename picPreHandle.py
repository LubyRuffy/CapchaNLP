#coding=utf-8
try:
    import Image
except ImportError:
    from PIL import Image
#import pytesseract
from PIL import ImageEnhance 
from PIL import ImageFilter 
import os
import cutPic
#定义图像预处理的整体函数
def Change_Image(Docu_Name,Dist):
    im = Handle_Image(Docu_Name,Dist,40,10)
    X_Value=cutPic.Cut_X(im)
    Y_Value=cutPic.Cut_Y(im)
#    print X_Value
#    print Y_Value
    ims = []
    Image_Value=[]
    Image_Values=[]
    Image_Value_Row=[]
    for k in range(4):
        im1= im.crop((X_Value[(2*k)],Y_Value[(2*k)],(X_Value[(2*k+1)]+1),(Y_Value[(2*k+1)]+1))) #切割图像为4个子图像
        ims.append(im1)    
        for j in range(Y_Value[(2*k)],(Y_Value[(2*k+1)]+1)):
            for i in range(X_Value[(2*k)],(X_Value[(2*k+1)]+1)):
                if im.getpixel((i,j))==0:#黑色像素的值是0
                    Image_Value_Row.append(1)
                else:
                    Image_Value_Row.append(0)
            Image_Value.append(Image_Value_Row)#
            Image_Value_Row=[]#
           
        Image_Values.append(Image_Value)
        Image_Value=[]
    
    return Image_Values #返回切割后各个图像对应的黑白像素的0-1值所存储在其中的三维数组。
   
#处理图片以便后续的0-1二值化
def Handle_Image(Docu_Name,Dist,w_box,h_box):
    im = Image.open('%s.jpg'%(Dist+Docu_Name)) #打开对应目录的png格式的验证码图片
    im=im.convert('RGB')
    w,h = im.size
    im = resize(w, h, w_box, h_box, im)
    for j in range(im.size[1]):
        for i in range(im.size[0]):            
            Gray = Change_Gray(im.getpixel((i,j)))  #灰度化
            im.putpixel([i,j],(Gray,Gray,Gray))
            # if i==0 or i==(im.size[0]-1): #将图片的第一行和最后一行设为白色。
                # im.putpixel([i,j],(255,255,255))
            # if j==0 or j==(im.size[1]-1):#将图片的第一列和最后一列设为白色。
                # im.putpixel([i,j],(255,255,255))
    enhancer = ImageEnhance.Contrast(im) #增加对比对
    im = enhancer.enhance(2)
    enhancer = ImageEnhance.Sharpness(im) #锐化
    im = enhancer.enhance(2)
    enhancer = ImageEnhance.Brightness(im) #增加亮度
    im = enhancer.enhance(2)
    #im=im.convert('L').filter(ImageFilter.DETAIL) #滤镜效果
    im = im.convert('1') #转为黑白图片
    
    im = Clear_Point(im) #清除周围8个像素都是白色的孤立噪点
    #im = Clear_Point_Twice(im) #清除两个孤立的噪点：周围8个像素中有7个是白色，而唯一的黑色像素对应的他的邻域（他周围的8个像素）中唯一的黑色像素是自身。
    #im = Clear_Point_Third(im) #清除第三种噪点：左右都是3个（含）以上的空白列，自身相邻的3个列上的X值投影不大于3.
    w,h = im.size
    im = resize(w, h, w_box, h_box, im)
    return im
    
def Handle_Image1(image,w_box,h_box):
    im = image
    im=im.convert('RGB')
    w,h = im.size
    im = resize(w, h, w_box, h_box, im)
    for j in range(im.size[1]):
        for i in range(im.size[0]):            
            Gray = Change_Gray(im.getpixel((i,j)))  #灰度化
            im.putpixel([i,j],(Gray,Gray,Gray))
            # if i==0 or i==(im.size[0]-1): #将图片的第一行和最后一行设为白色。
                # im.putpixel([i,j],(255,255,255))
            # if j==0 or j==(im.size[1]-1):#将图片的第一列和最后一列设为白色。
                # im.putpixel([i,j],(255,255,255))
    enhancer = ImageEnhance.Contrast(im) #增加对比对
    im = enhancer.enhance(5)
#    enhancer = ImageEnhance.Sharpness(im) #锐化
#    im = enhancer.enhance(2)
#    enhancer = ImageEnhance.Brightness(im) #增加亮度
#    im = enhancer.enhance(2)
    #im=im.convert('L').filter(ImageFilter.DETAIL) #滤镜效果
    im = im.convert('1') #转为黑白图片
    
    im = Clear_Point(im) #清除周围8个像素都是白色的孤立噪点
    #im = Clear_Point_Twice(im) #清除两个孤立的噪点：周围8个像素中有7个是白色，而唯一的黑色像素对应的他的邻域（他周围的8个像素）中唯一的黑色像素是自身。
    #im = Clear_Point_Third(im) #清除第三种噪点：左右都是3个（含）以上的空白列，自身相邻的3个列上的X值投影不大于3.
    w,h = im.size
    im = resize(w, h, w_box, h_box, im)
    return im
#改变灰度，查文献后发现据说按照下面的R，G，B数值的比例进行调整，图像的灰度最合适。
def Change_Gray(RGB_Value):
    Gray = int((RGB_Value[0]*299+RGB_Value[1]*587+RGB_Value[2]*114)/1000)
    return Gray
 
#图像处理的关键是后续的清楚噪点，也就是所谓的孤立点
 
#清除单个孤立点
def Clear_Point(im):
    for j in range(1,(im.size[1]-1)):
        for i in range(1,(im.size[0]-1)):
            if im.getpixel((i,j))==0 and im.getpixel(((i-1),(j-1)))==255  and im.getpixel((i,(j-1)))==255  and im.getpixel(((i+1),(j-1)))==255  and im.getpixel(((i-1),j))==255  and im.getpixel(((i+1),j))==255  and im.getpixel(((i-1),(j+1)))==255  and im.getpixel((i,(j+1)))==255  and im.getpixel(((i+1),(j+1)))==255:
                im.putpixel([i,j],255)
    return im
 
#清除只有2个的孤立点
def Clear_Point_Twice(im):
    for j in range(1,(im.size[1]-1)):
        for i in range(1,(im.size[0]-1)):
            if im.getpixel((i,j))==0 and ( im.getpixel(((i-1),(j-1)))+im.getpixel((i,(j-1)))+im.getpixel(((i+1),(j-1)))+im.getpixel(((i-1),j))+im.getpixel(((i+1),j))+im.getpixel(((i-1),(j+1)))+im.getpixel((i,(j+1)))+im.getpixel(((i+1),(j +1)))) == 255*7:
                if im.getpixel(((i+1),j))==0: #因为扫描的顺序是从上到下，从左到右，噪点只能是在自身像素的后面和下面，也就是只有4个可能性而已，而不是8个，可以减少一半的代码。
                    m=i+1
                    n=j
                    if ( im.getpixel(((m-1),(n-1)))+im.getpixel((m,(n-1)))+im.getpixel(((m+1),(n-1)))+im.getpixel(((m-1),n))+im.getpixel(((m+ 1),n))+im.getpixel(((m-1),(n+ 1)))+im.getpixel((m,(n+ 1)))+im.getpixel(((m +1),(n+ 1)))) == 255*7:
                       im.putpixel([i,j],255)
                       im.putpixel([m,n],255)
                elif im.getpixel(((i-1),(j+1)))==0:
                    m=i-1
                    n=j+1
                    if ( im.getpixel(((m-1),(n-1)))+im.getpixel((m,(n-1)))+im.getpixel(((m+1),(n-1)))+im.getpixel(((m-1),n))+im.getpixel(((m +1),n))+im.getpixel(((m-1),(n +1)))+im.getpixel((m,(n+ 1)))+im.getpixel(((m+ 1),(n +1)))) == 255*7:
                       im.putpixel([i,j],255)
                       im.putpixel([m,n],255)
                elif im.getpixel((i,(j+1)))==0:
                    m=i
                    n=j+1
                    if ( im.getpixel(((m-1),(n-1)))+im.getpixel((m,(n-1)))+im.getpixel(((m+1),(n-1)))+im.getpixel(((m-1),n))+im.getpixel(((m+ 1),n))+im.getpixel(((m-1),(n+ 1)))+im.getpixel((m,(n +1)))+im.getpixel(((m+ 1),(n +1)))) == 255*7:
                       im.putpixel([i,j],255)
                       im.putpixel([m,n],255)
                elif im.getpixel(((i+1),(j+1)))==0:
                    m=i+1
                    n=j+1
                    if ( im.getpixel(((m-1),(n-1)))+im.getpixel((m,(n-1)))+im.getpixel(((m+1),(n-1)))+im.getpixel(((m-1),n))+im.getpixel(((m+ 1),n))+im.getpixel(((m-1),(n +1)))+im.getpixel((m,(n +1)))+im.getpixel(((m +1),(n +1)))) == 255*7:
                       im.putpixel([i,j],255)
                       im.putpixel([m,n],255)
    return im
 
#清楚第三种噪点比较麻烦，需要计算图像的0-1值在X轴的投影后，才能判断。
#依据图片像素颜色计算X轴投影
def Caculate_X(im):
    Image_Value=[]
    for i in range(im.size[0]):
        Y_pixel=0
        for j in range(im.size[1]):
            if im.getpixel((i,j))==0:
                temp_value=1
            else:
                temp_value=0
            Y_pixel = Y_pixel+temp_value
        Image_Value.append(Y_pixel)
    return Image_Value
 
#逐次将多列设为全白
def Set_White_Y(im,List_Black):
    for j in range(im.size[1]):
        for i in range(List_Black[0],(List_Black[(len(List_Black)-1)]+1)):
            im.putpixel([i,j],255)
    return im
 
#清除第三种残余的孤立点
def Clear_Point_Third(im):
    Image_Value = Caculate_X(im)
    List01=[]
    List_Black=[]
    List03=[]
    for i in range(len(Image_Value)): #从左到右扫描
        if Image_Value[i] ==0 and len(List_Black) == 0 : #X轴投影是0，说明是空白列，黑色列的列表是空值，说明当前列是黑色列的左侧
            List01.append(i)
        elif  Image_Value[i] >0 : #X周投影大于0的列，即扫描到了黑色列
            List_Black.append(i)
        elif Image_Value[i] ==0 and len(List_Black)>0 and len(List_Black)<=3:# 黑色列的列表的长度大于0，不大于3个空白字符，现在的X轴投影为0，说明现在扫描到了孤立噪点所在的黑色列右侧的空白列
            List03.append(i)
            if len(List03)==3:#空白列为3列
                    im = Set_White_Y(im,List_Black) #逐次将多列设为全白
                    List01=[]
                    List_Black=[]
                    List03=[]
        elif Image_Value[i] ==0 and len(List_Black)>3: #当前是空白列，黑色列的数量大于3，说明扫描到了数字所在部分（不是噪点）的右侧空白列。
            List01=[]
            List_Black=[]
            List03=[]
            List01.append(i)
    return im
    
def resize(w, h, w_box, h_box, pil_image):
    '''
    resize a pil_image object so it will fit into
    a box of size w_box times h_box, but retain aspect ratio
    '''
    f1 = 1.0*w_box/w # 1.0 forces float division in Python2
    f2 = 1.0*h_box/h
    factor = min([f1, f2])
    #print(f1, f2, factor) # test
    # use best down-sizing filter
    width = int(w*factor)
    height = int(h*factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)
#切割完毕后，将4个子图片的像素颜色信息写入到文本文件
def Write_ImageFile(Docu_Name,Image_Value,Dist):   
    f=open('%s' % (Dist+Docu_Name+'.txt'),'w')
    f.write(str(Image_Value))
    f.close()
    
def Write_Txt(Dist):
    ''' txt文件的写入格式
    #文件名
    a=[]
    Image_Libs.append(a)
    '''
    #Big_Txt = Document_Name()
    fw = open ('%s' %(Dist+'code1'+'.txt'),'a')
    Array=[]
    for item in os.listdir(Dist): # 遍历指定目录
        if os.path.isfile(Dist+item) and item.endswith('.txt') and len(item)<20: # 判断是否为.txt文件
            f = open((Dist+item),'r') # 打开文件
            line=f.readline()
            for i in range(2): #为了保障比对时的成功率，每个素材要写入2次，这样可以提高比对成功率。
                fw.write('#')
                fw.write(item)
                fw.write('\n')
                fw.write('a=')
                fw.write(line)
                fw.write('\nImage_Libs.append(a)\n')
                Array.append(int(item[0]))
            f.close()           
    a={}
    for i in Array:
        if Array.count(i)>0:
            a[i]=Array.count(i)
    b=[]
    b.append(a[0])
    for i in range(1,len(a)):
        b.append(( a[i]+b[i-1] )) #生成一个从0到9，依次存储各个数字的累积个数的数组。
    b=[0]+b
    fw.write('\n')
    fw.write('Rank_Index=')
    fw.write(str(b))
    fw.close() 
#if __name__ == '__main__':
#    #Change_Image('code345','E:\\USTCEPCCode\\')
#    Dist = 'E:\USTCEPCCode\\'
#    Image_Value = Change_Image('code300',Dist)
#    Write_ImageFile('code',Image_Value,Dist)
#    Write_Txt(Dist)
    # for i in range (300,310):
        # imgry = Handle_Image('code'+str(i),'E:\USTCEPCCode\\')
        # imgry.save('e:\\code%s.tiff'%(str(i)))  
    
    
    
    
    