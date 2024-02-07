import os
from flask import Flask, request, render_template
import sqlite3
import json
import logging
from PIL import Image
import io
import pandas as pd
from torchvision.transforms import Resize
from apscheduler.schedulers.background import BackgroundScheduler
from predict import make_predictions

# schedule nightly predictions. 3:30 at night so it is not influenced by changing between daylight saving time and regualar time.
scheduler = BackgroundScheduler()
scheduler.add_job(func=make_predictions, trigger="cron", hour=3, minute=30)
scheduler.start()

logging.basicConfig(level="DEBUG") # for general logging purposes

config = json.load(open("config.json", "r")) # loading the config for naming uploaded images
allowed_file_extentions = {"png", "jpg", "jpeg"} # to later ensure that only images are uploaded

# setting up the Flask app
app = Flask(__name__)
app.config["DEBUG"] = True
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024 # 100 MB

def check_database():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("CREATE TABLE if not exists images (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT NOT NULL, status TEXT NOT NULL, classification TEXT, confidence FLOAT)")
    conn.commit()
    conn.close()

# main page
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

# upload page
@app.route('/upload', methods=['POST'])
def upload():
    # connecting to database
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    # Create the table if it does not exist yet
    check_database()

    files = request.files.getlist('inputFile')
    ids = []
    for file in files:
        print(file)
        # check that file is an of allowed_file_extentions
        if file.filename.split(".")[-1].lower() not in allowed_file_extentions:
            return "File type not allowed", 400

        # turn the image into .jpg
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        resize = Resize((80,60))
        img = resize(img)
        if img.format != "JPEG":
            img = img.convert("RGB")
            png_img_bytes = io.BytesIO()
            img.save(png_img_bytes, format="JPEG")
            img_bytes = png_img_bytes.getvalue()

        # save the image to the uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'image_{config["image_id"]}.jpeg')
        with open(filepath, "wb") as f:
            f.write(img_bytes)

        # add image entry to the database
        c.execute("INSERT INTO images (path, id, status) VALUES (?,?,?)", (filepath, config["image_id"], "in progress"))

        # increment id by 1
        ids.append(config["image_id"])
        config["image_id"] += 1
    # save the new id to the config file
    with open("config.json", "w") as f:
        json.dump(config, f)
        
    conn.commit()
    conn.close()

    return f'File uploaded successfully. View database for uploaded images. Uploaded ids:\n{ids}'

@app.route("/database", methods=["GET"])
def database():
    conn = sqlite3.connect('database.db')
    check_database()

    df = pd.read_sql_query("SELECT * FROM images", conn)

    return df.to_html()

@app.route("/clear_db", methods=["GET"])
def clear_df():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("DROP TABLE images")
    conn.commit()
    conn.close()

    return "The database has been cleared!"

@app.route("/status", methods=["GET"])
def status():
    if "image_id" not in request.args:
        return "Please provide image_id"
    image_id = request.args.get("image_id")

    try:
        image_id = int(image_id)
    except Exception as e:
        logging.error(Exception)
        return "Please provide an integer as image_id"
    
    if image_id < 0:
        return "Please provide valid image_id"
    
    check_database()
    conn = sqlite3.connect("database.db")
    df = pd.read_sql_query("SELECT * FROM images", conn)

    if image_id in df["id"].unique():
        classification = f' | Classification: {df[df["id"] == image_id]["classification"].item()} | Confidence: {df[df["id"] == image_id]["confidence"].item()}' if df[df["id"] == image_id] is not None else ""
        return f'The current status of the uploaded image with the id {image_id} is: {df[df["id"] == image_id]["status"].unique()[0]}{classification}'

    return f"Oops! Couldn't find this data entry."
    

if __name__ == "__main__":
    app.run()