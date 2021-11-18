import cv2 #モーショントラッキング
import os
import base64
import json
import random
import time
import shutil
import sys
from xml.etree.ElementTree import * #アノテーションした画像をxml形式に格納する
from werkzeug.utils import secure_filename
from flask import *
import cuttoxml

classcount=80       #すでにあるクラス数
UPLOAD_FOLDER = './uploads'     #送られてくる動画を保存するもの
ALLOWED_EXTENSIONS = set(['mp4', 'mov','avi', 'm4a'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ur="http://localhost:8080/" #公開サイト
def start():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(port=8080)  #なぜか好む8080で開放

def allwed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS     # 拡張子を.以降取り出し、許可された拡張子か確認

def xmwrite(p,fai,posx,posy,posx2,posy2):   #tree構造を気合で・・（´・ω・｀）
    an=Element('annotation')
    fn=SubElement(an,'filename')
    fn.text=fai
    sou=SubElement(an,'source')
    db=SubElement(sou,'database')
    db.text="original"
    at=SubElement(sou,'annotation')
    at.text="original"
    im=SubElement(sou,'image')
    im.text="XXX"
    fl=SubElement(sou,'flickrid')
    fl.text="0"
    ow=SubElement(an,'owner')
    fli=SubElement(ow,'flickrid')
    fli.text="0"
    nm=SubElement(ow,'name')
    nm.text="?"
    sz=SubElement(an,'size')
    wi=SubElement(sz,'width')
    wi.text="540"
    he=SubElement(sz,'height')
    he.text="960"
    dp=SubElement(sz,'depth')
    dp.text="3"
    ob=SubElement(an,'object')
    na=SubElement(ob,'name')
    na.text=str(lab)
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

def frame_resize(frame, n=2):
    #画面に収めるため縮小　（座標ずれに注意）
    return cv2.resize(frame, (int(frame.shape[1]*0.25), int(frame.shape[0]*0.25)))



@app.route('/', methods=['GET', 'POST'])#GET,POSTを許可
def uploads_file():
    global fn,lab
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
            fn=file.filename
            lab = request.form['label']
            # ファイルの保存
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # アップロード後のページに転送
            return redirect(url_for('uploaded_file', filename=filename))
    return render_template('index.html')

@app.route('/ret', methods=['GET', 'POST'])#GET,POSTを許可
def posdata():
    global xd,yd,xxd,yyd
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        print('in')
        # データの取り出し
        xd = request.form['x']#1フレーム目のbbox 1つ目x
        yd = request.form['y']#1フレーム目のbbox 1つ目y
        xxd = request.form['xx']#1フレーム目のbbox もう片方のx（差分）
        yyd = request.form['yy']#1フレーム目のbbox もう片方のy　(差分)
        print(xd,yd,xxd,yyd)
        return redirect('t')#tへ推移させる

@app.route('/t', methods=['GET', 'POST'])#GET,POSTを許可
def up():
    global lab
    tracker = cv2.TrackerMIL_create()#トラッキングのインスタンスを作成
    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))#フレーム桁数をカウント
    n=0
    src = './data/coco.names'#現在のクラス名を取得
    copy = dir+"/classes.names"#クラスの名前生成
    shutil.copyfile(src,copy)#フレーム桁数をカウント
    cf = open("config/"+"custom"+rad+".data",'a+')#カスタムデータの作成
    cf.write("classes="+str(classcount)+"\n"+"train="+dir+"/train.txt\nvalid="+dir+"/valid.txt\nnames="+dir+"/classes.names")#クラス数、trainリスト、validリストを生成
    cf.close()
    valid = open(dir+"/valid.txt", 'a')
    train = open(dir+"/train.txt", 'a')
    list  = open(dir+"/opencvlist.txt", 'a')
    winp = "C:/Users/misever/AppData/Local/Packages/CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc/LocalState/rootfss/home/cou/python/discord/PyTorch-YOLOv3/"
    print("a")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        print("ok")
        frame = frame_resize(frame)
        fr=frame
        p1=int(float(xd)),int(float(yd))
        p2=int(float(xxd)),int(float(yyd))
        bbox = (int(float(xd)),int(float(yd)),int(float(xxd)),int(float(yyd)))
        print(bbox)
        #bbox = cv2.selectROI(frame, False)
        #print(bbox)
        cv2.imshow("Tracking", frame)
        ok = tracker.init(frame, bbox)
        if not bbox==(0,0,0,0):
            break;
    print("ok")
    while True:
        ret, frame = cap.read() #1フレーム読み込み
        if not ret:
            break;
        frame = frame_resize(frame)
        height, width, channels = frame.shape[:3]
        # Start timer
        timer = cv2.getTickCount()

        # トラッカーをアップデートする
        track, bbox = tracker.update(frame)

        # FPSを計算する
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # 検出した場所に四角を書く
        if track:
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = (int(bbox[0] + bbox[2]))
            y2 = (int(bbox[1] + bbox[3]))
            pos=" 1 "+str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2)
            p1 = x1,y1
            p2 = x2,y2
            a=str(n).zfill(digit)
            tmp=dir+"/"+fpa+'-'+a+".xml"
            print(tmp)
            ntxt = open(dir+"/image/"+fpa+'-'+a+".txt", 'a')
            cv2.imwrite(os.path.join(dir+"/image",'{}-{}.{}'.format(fpa,a,ext)),frame)
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)
            cv2.imshow("Tracking", frame)
            cv2.imwrite(os.path.join(dir+"/bbox",'{}-{}.{}'.format(fpa,a,ext)),frame)
            train.write(rt+dir+"/image/"+fpa+"-"+a+".jpg"'\n')
            list.write(winp+dir+"/image/"+fpa+'-'+a+ext+pos+'\n')
            ff=fpa+'-'+a+".jpg"
            print(ff,tmp)
            xmwrite(tmp,ff,str(x1),str(y1),str(x2),str(y2))
            x1=float((x1+x2)/(2*width))
            y1=float((y1+y2)/(2*height))
            x2=float((x2-x1)/(2*width))
            y2=float((y2-y1)/(2*height))
            print(x1,y1,x1,y2)
            ntxt.write(str(lab)+" "+str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2)+"\n")
            n += 1
            ntxt.close()


        # キー入力を1ms待って、k が27（ESC）だったらBreakする
        k = cv2.waitKey(1)
        if k == 27 :
            break
    cap.release()
    cv2.destroyAllWindows()
    train.close()
    list.close()
    print("please move to valid.txt in from train.txt a few text(笑)")
    print("please run to cmd :")
    print("chmod 777 config/* && chmod 777 "+dir+"/*")
    print("bash config/create_custom_model.sh <num-classes>")
    print("python3 train.py --model_def config/yolov3-custom.cfg --data_config config/custom"+rad+".data")
    cuttoxml.start()


@app.route('/uploads/<filename>')
# ファイルを表示する
def uploaded_file(filename):
    print(filename)
    global dir,ext,digit,jpath,train,list,winp,cap,fpa,rad,rt
    filename = os.path.basename(filename).split(".")[0]
    vpath='uploads/'+filename+'.mp4'
    fpa=filename
    rt="/home/cou/python/discord/PyTorch-YOLOv3/"
    ext='jpg'
    jpath='images/'+filename+ext
    cap=""
    cap = cv2.VideoCapture(vpath)
    rad=str(random.randint(0,100))
    dir="data/custom/image/"+filename+rad
    os.makedirs(dir+"/bbox", exist_ok=True)
    os.makedirs(dir+"/image", exist_ok=True)
    os.makedirs('images', exist_ok=True)
    print(dir)
    ret, frame = cap.read()
    frame = frame_resize(frame)
    cv2.imwrite(os.path.join('images','{}{}.{}'.format(filename,'base','jpg')),frame)
    response = []
    # 保存したファイルに対してエンコード
    print(os.path.join('images','{}{}.{}'.format(filename,'base','jpg')))
    with open(os.path.join('images','{}{}.{}'.format(filename,'base','jpg')), "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('utf-8')
        height, width, channels = frame.shape[:3]
    # レスポンスのjsonに箱詰め
    return render_template('res.html',data= img_base64,h=height,w=width)
    #return jsonify({"language": img_base64})
if __name__=='__main__':
    cuttoxml.start()
