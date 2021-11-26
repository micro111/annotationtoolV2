import cv2 #モーショントラッキング
import os
import base64
import json
import random
import time
import numpy as np
import shutil
import sys
import datetime
import glob
from xml.etree.ElementTree import * #アノテーションした画像をxml形式に格納する
from werkzeug.utils import secure_filename
from flask import *
#中間画像の作成をするか　する:1　しない:0
debug =0

#flask web
UPLOAD_FOLDER = './uploads'     #送られてくる動画を保存するもの
ALLOWED_EXTENSIONS = set(['mp4', 'mov','avi', 'm4a'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def start():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0",port=8888)  #なぜか好む8080で開放

def allwed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS     # 拡張子を.以降取り出し、許可された拡張子か確認

def frame_resize(frame, n=2):
    #画面に収めるため縮小　（座標ずれに注意）
    return cv2.resize(frame, (int(frame.shape[1]*0.25), int(frame.shape[0]*0.25)))


def xmwrite(p,filename,posx,posy,posx2,posy2):   #tree構造を気合で・・（´・ω・｀）
    an=Element('annotation')
    filen=SubElement(an,'filename')
    filen.text=filename
    source=SubElement(an,'source')
    db=SubElement(source,'database')
    db.text="original"
    at=SubElement(source,'annotation')
    at.text="original"
    im=SubElement(source,'image')
    im.text="XXX"
    fl=SubElement(source,'flickrid')
    fl.text="0"
    ow=SubElement(an,'owner')
    fli=SubElement(ow,'flickrid')
    fli.text="0"
    nm=SubElement(ow,'name')
    nm.text="?"
    sz=SubElement(an,'size')
    wi=SubElement(sz,'width')
    wi.text="480"
    he=SubElement(sz,'height')
    he.text="480"
    dp=SubElement(sz,'depth')
    dp.text="3"
    ob=SubElement(an,'object')
    na=SubElement(ob,'name')
    na.text=str(cnumber)
    po=SubElement(ob,'pose')
    po.text="Unspecified"
    tr=SubElement(ob,'truncated')
    tr.text="1"
    di=SubElement(ob,'difficult')
    di.text="0"
    bn=SubElement(ob,'bndbox')
    x1=SubElement(bn,'xmin')
    x1.text=posx
    y1=SubElement(bn,'ymin')
    y1.text=posy
    x2=SubElement(bn,'xmax')
    x2.text=posx2
    y2=SubElement(bn,'ymax')
    y2.text=posy2
    tree=ElementTree(an)
    tree.write(p)

@app.route('/', methods=['GET', 'POST'])#GET,POSTを許可
def uploads_file():
    global lab,filename
    if request.method == 'POST': #POSTrec
        # ファイルがなかった場合の処理
        if 'file' not in request.files:     #ファイル選択
            print('ファイルがありません')
            return redirect(request.url)
        # データの取り出し
        file = request.files['file']
        # ファイル名がなかった時の処理
        if file.filename == '':
            print('ファイルがありません')
            return redirect(request.url)
        # ファイルのチェック
        if file and allwed_file(file.filename):
            # サニタイズ処理
            filename = secure_filename(file.filename)
            lab = request.form['label']
            # ファイルの保存
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # アップロード後のページに転送
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('index.html')

@app.route('/check', methods=['GET', 'POST'])#GET,POSTを許可
def posdata():
    global xd,yd,xxd,yyd
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        return redirect('trackstart')#tへ推移させる

@app.route('/trackstart', methods=['GET', 'POST'])#GET,POSTを許可

def up():
    global lab,cnumber,filename
    currentpath = os.getcwd()
    filename = os.path.basename(filename).split(".")[0]
    classfile=open(dir+"/classes.names",'a+') #新規作成するclass.names
    cnumber = 0#現在入っているclass.namesの行数をカウント
    with open('./data/class.names') as f: #元となるclass.namesを取得
        for line in f:
            cnumber += 1
            classfile.write(line) #元の中身をついでにコピー
    classfile.write(str(lab))
    classfile.close()
    customconfig = open("config/"+"custom"+date+".data",'a+')#カスタムデータの作成
    customconfig.write("classes="+str(cnumber)+"\n"+"train="+currentpath+"/"+dir+"/train.txt\nvalid="+currentpath+"/"+dir+"/valid.txt\nnames="+currentpath+"/"+dir+"/classes.names")#クラス数、trainリスト、validリストを生成
    customconfig.close()
    valid = open(dir+"/valid.txt", 'a')
    train = open(dir+"/train.txt", 'a')
    list  = open(dir+"/opencvlist.txt", 'a')

    os.makedirs(dir, exist_ok=True)
    if debug:
        os.makedirs(dir+"/gray", exist_ok=True)
        os.makedirs(dir+"/bin", exist_ok=True)
        os.makedirs(dir+"/mask", exist_ok=True)
        os.makedirs(dir+"/masked", exist_ok=True)
    backgroundimg=glob.glob('./back/*')
    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    n = 0
    while True:
        ret, frame = cap.read()
        if n==0:
            t=frame.copy()
        if ret:
            frame=cv2.resize(frame,dsize=(480,480))

            #グレースケール
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 2値化する。
            ret, binary = cv2.threshold(gray, 80, 255,cv2.THRESH_BINARY_INV)


            #境界線を取得
            contours, hierarchy = cv2.findContours(binary,
                                                   cv2.RETR_LIST,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            #境界線から面積が一番でかくなる（物体全域を囲う）境界を抽出
            contour = max(contours, key=lambda x: cv2.contourArea(x))

            #囲う
            img_contour = cv2.drawContours(frame, contour, -1, (0, 255, 0), 5)

            #背景を透過した画像を生成する。
            mask = np.zeros_like(frame)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)

            #グレースケールにマスク画像を重ねて透過する
            toukaimg = cv2.bitwise_and(gray,mask)

            #自動で背景を重ね合わせるため、ランダムで読み出しリサイズする
            back=cv2.imread(backgroundimg[random.randint(0,len(backgroundimg)-1)])
            back=cv2.cvtColor(back,cv2.COLOR_BGR2GRAY)
            back=cv2.resize(back,dsize=(480,480))

            #マスクと重ねる
            mas2=cv2.bitwise_and(back,cv2.bitwise_not(mask))
            #gousei = cv2.bitwise_or(mas2,toukaimg)
            gousei=toukaimg
            if debug:
                cv2.imwrite(dir+"/gray/"+str(n)+".jpg", gray)
                cv2.imwrite(dir+"/bin/"+str(n)+".jpg", binary)
                cv2.imwrite(dir+"/result/"+str(n)+".jpg",img_contour)
                cv2.imwrite(dir+"/mask/"+str(n)+".jpg",mask)
                cv2.imwrite(dir+"/masked/"+str(n)+".jpg",toukaimg)

            #bbox用に値を取り出しておく
            x1,y1,x2,y2 = cv2.boundingRect(contour)

            #取り出した値を各種ファイルに書き込む
            pos=" 1 "+str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2)
            p1 = x1,y1
            p2 = x2,y2
            pathnumber=str(n).zfill(digit)
            xmlfullpath=dir+"/"+filename+'-'+pathnumber+".xml"

            #アノテーションをtxt形式で書き出す。
            anotxt = open(dir+"/labels/"+filename+'-'+pathnumber+".txt", 'a')
            anotxt.write(str(cnumber)+" "+str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2)+"\n")
            anotxt.close()

            #画像を保存する(未加工ver)
            cv2.imwrite(os.path.join(dir+"/images",'{}-{}.{}'.format(filename,pathnumber,ext)),gousei)

            #画像を保存する(bboxを書いた加工ver)
            cv2.rectangle(gousei,(x1,y1),(x1+x2,y1+y2),(255),10)
            cv2.imwrite(os.path.join(dir+"/bbox",'{}-{}.{}'.format(filename,pathnumber,ext)),gousei)

            #train.txtとopencvのlist.txtを書く
            train.write(currentpath+"/"+dir+"/images/"+filename+"-"+pathnumber+".jpg"'\n')
            list.write(currentpath+"/"+dir+"/images/"+filename+'-'+pathnumber+ext+pos+'\n')
            jpgfullname=filename+'-'+pathnumber+".jpg"
            xmwrite(xmlfullpath,jpgfullname,str(x1),str(y1),str(x2),str(y2))

            n += 1


        else:
            break
    print("You're Code! (Yolov3-PyTorch)")
    print("bash ./config/create_custom_model.sh "+str(cnumber))
    print("sudo python3 train.py -d"+currentpath+"/config/"+"custom"+date+".data -e 200 --n_cpu 2 -m config/yolov3-custom.cfg")
    print("sudo python3 detect.py -i <TestimageDirPath> -w <checkpoint.pth> -c "+currentpath+"/"+dir+"/classes.names" )
    print("Done:!!!!")
    cap.release()
    cv2.destroyAllWindows()
    train.close()
    list.close()
    return redirect('/')

@app.route('/uploads/<filename>')
# ファイルを表示する
def uploaded_file(filename):
    global dir,ext,train,list,currentpath,cap,date
    filename = os.path.basename(filename).split(".")[0]
    print(filename)
    vpath='uploads/'+filename+'.mp4'
    ext='jpg'
    cap = cv2.VideoCapture(vpath)
    date=str(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    dir="data/custom/image/"+date
    os.makedirs(dir+"/bbox", exist_ok=True)
    os.makedirs(dir+"/images", exist_ok=True)
    os.makedirs(dir+"/labels", exist_ok=True)
    os.makedirs('images', exist_ok=True)
    ret, frame = cap.read()
    frame = frame_resize(frame)
    cv2.imwrite(os.path.join('images','{}{}.{}'.format(filename,'base','jpg')),frame)
    response = []
    with open(os.path.join('images','{}{}.{}'.format(filename,'base','jpg')), "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
        height, width, channels = frame.shape[:3]
    return render_template('res.html',data= img_base64,h=height,w=width)

if __name__=='__main__':
    start()
