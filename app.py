from flask import Flask,render_template,url_for,redirect,request,jsonify,flash
from torch_utils import transform_image,Predict
app=Flask(__name__)

ALLOWED_EXTENSIONS={'png','jpg','jpeg'}
def allowed_file(filename):
      return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file=request.files.get('file')
        if file is None or file.filename=="":
            error='No file uploaded'
            return render_template('index.html',messages=error)
        if not allowed_file(file.filename):
             error='Format Not Supported'
             return render_template('index.html',messages=error)
        try:
            img_bytes=file.read()
            tensor=transform_image(img_bytes)
            labels=Predict(tensor)
            return render_template('result.html',labels=labels)
        except:
            error='Prediction Error'
            return render_template('index.html',messages=error)
    return render_template('index.html')
if __name__=='__main__':
    app.run()
