# READ ME
Welcome to the READ ME file for this project. Here, you'll find some extra explanation and information.
This is the coding part for an assignment in the course "From Model to Production" by IU.

### The Assignment / Task 2
An online shopping platform for sustainable products is growing rapidly. Despite the desirable development of
the company, this also comes with new challenges. As one example, as more and more products are ordered via
the platform, the number of items sent back for refund also increases. The workforce that is necessary to
categorize the re-shipped products has increased in the last month beyond a viable amount. As the newly
employed data scientist of the company, you are assigned the task of developing a machine learning model that
automatically classifies the refund items into categories based on pictures of the items. Your model should run as
a service that can be triggered in batches overnight. This project hopes to reduce the workforce and costs for
manually sorting items. Instead, incoming goods will automatically be categorized daily.
1. Design a conceptual architecture of your system. By doing so, consider data ingestion, storing,
processing, and handling requests to the prediction model as a service. Draft a visual overview of your
architectural design, showing which data and processes are handed over by which application to the
next. This will also guide you through the next steps of your project.
2. Identify an open data source of images of arbitrary items. A good starting point for your research are open
data science competition websites, such as Kaggle. Ensure that there are enough images of each category
on which you would like to train your model.
3. Connect to your image base in Python and train a machine learning model that classifies the images into
categories. Do not put too much effort into building this model, and just make sure that it performs the
task at hand and check basic statistical measures.
4. Package your model in a way that it can take (serialized) image data over a standardized RESTful API and
respond with probabilities for classes of items. You might want to use Python libraries for this, such as
Flask or mflow. A good starting point for this step is the documentation of the respective Python libraries.
Keep in mind that you put your model to production in a way bet it can handle batches of data.
5. Set up the system to automatically perform predictions in batches on all new data in the image storage
every night. At this stage, you have various options: You can work with mlflow, MLOps platform forms,
such as Azure DevOps, Jenkins, or GitHub actions, or you can rely on cronjobs if you like and if it fits the
requirements of your system.
6. It is unnecessary to implement your system in the cloud to obtain the highest grade for your project.
Implementing to the cloud is a little more elaborate, though you might want to challenge yourself and
gain some extra points for this effort.

### General Setup
I decided to implement the system using a combination of `Flask`, `SQLite3`, and `<Machine Learning Library>`.
`Flask` is a web framework for Python, which is used to create the API. Through this API, the users will communicate with the system. This way, users can upload new images or retrieve data of previous uploads.
`SQLite3` is a database management system, which is used to store the data of the images. This way, the data can be retrieved later on.
`<Machine Learning Library>` is a machine learning library, which is used to train the model and to make predictions.

### Dataset
To train the model, a [dataset from Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset?resource=download) was used. This dataset includes 44,000 images of different clothing categories. These images will be used for training and testing purposes. The dataaset was chosen for the following reasons:
1. All images have a clear separation between the object (i.e., piece of clothing) and the background (white background).
2. The images are of high quality with close to no noise.
3. In a real life scenario, when taking images of returned products, the images may be taken in a similar setting as those of the dataset. This way, the model will likely function as intended.

Additionally, a [smaller version of the dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) was used as well, for faster model training and quicker iterations. 

There were some entries where there were too many fields. This happened because some fields have regular text in them. Some of them have a comma, which is used as the separator for this .csv file. This lead to a parsing Error. To ignore these entries, the following code was used:
```python
df = pd.read_csv("training_data_small/styles.csv", error_bad_lines=False) # instead of the regular df = pd.read_csv("training_data_small/styles.csv")
```
There are 143 different `articleType`s in the dataset. These are the different categories of clothing. 

## Quick Usage Tipps:
1. Always use the venv for running code (`python3 -m venv venv``)
2. Install all requirements using `pip install requirements.txt`
3. Run the clear_db.py file to clear the entire database for resets. This does not reset the image id index, nor does it delete uploaded files
4. Run predict.py if you don't want to wait for automated schedule to run the code/predict the newly uploaded images.
5. Change the prediction model in config.json for quick model swapping. This allows for simultaneous training and deployment.