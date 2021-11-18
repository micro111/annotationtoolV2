import os
# request フォームから送信した情報を扱うためのモジュール
# redirect  ページの移動
# url_for アドレス遷移
from flask import Flask, request, render_template,redirect, url_for
# ファイル名をチェックする関数
from werkzeug.utils import secure_filename
# 画像のダウンロード
from flask import send_from_directory
# 画像のアップロード先のディレクトリ
UPLOAD_FOLDER = './uploads'
# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['mp4', 'mov','avi', 'm4a'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




def allwed_file(filename):
    # .があるかどうかのチェックと、拡張子の確認
    # OKなら１、だめなら0
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=['GET', 'POST'])
def uploads_file():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'file' not in request.files:
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
            # 危険な文字を削除（サニタイズ処理）
            filename = secure_filename(file.filename)
            # ファイルの保存
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # アップロード後のページに転送
            #return redirect(url_for('uploaded_file', filename=filename))
    return render_template('index.html')

@app.route('/uploads/<filename>')
# ファイルを表示する
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/image/<filename>", methods=["POST"])
def post():
    """
    画像をグレースケールに変換する
    """

    response = []

    for json in request.json:

        # Imageをデコード
        img_stream = base64.b64decode(json['Image'])

        # 配列に変換
        img_array = np.asarray(bytearray(img_stream), dtype=np.uint8)

        # open-cv でグレースケール
        img_gray = cv2.imdecode(img_array, 0)

        # 変換結果を保存
        cv2.imwrite('result.png', img_before)

        # 保存したファイルに対してエンコード
        with open('result.png', "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')

        # レスポンスのjsonに箱詰め
        response.append({'id':json['id'], 'result' : img_base64})

    return jsonify(response)

if __name__ == "__main__":
    app.run(port=8080)
