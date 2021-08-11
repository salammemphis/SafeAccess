'''************************************************
 Copyright (c) 2020 the university of Memphis to present
 All right reserved
 Author: Shahinur Alam
 Email:salam@memphis.edu
 **************************************************
This Flask Webservice facilitate personal profile creation and communicate with SafeAccess 
Android app. This webservice retrieve data from users (android app) and performs following task
 =============
1. Receive user information (demographics and pictures)
2. Creates a folder in file system and insert a record in DB if that person not in
    the database/profile and returns a unique ID
3. Process the received face images and save in file system
3. Update/train the person recognition model
4. Perform model and data versioning 
 
 Webservice: Flask Python has been used to create webservice,
 basic os related packages for file operation: os glob, pickle
 image processing: opencv cv2
 encoding:base64
 logger: logging package
 database: MySql
 model: will be saved as .yml
 How to run: Python3 add_person.py
'''
from flask import Flask, request, Response,jsonify
import cv2, sys, pickle, struct, zlib, os,time
import numpy as np
import argparse, jsonpickle
import imutils, shutil, glob
from PIL import Image
import fnmatch
import json
import mysql.connector
from mysql.connector import Error
import base64
#to store server log
import logging
#read config parameter from file
import configparser

#Create Instance of Flask class and pass module name

app = Flask(__name__)
logging.basicConfig(filename='server.log',format='%(asctime)s  %(name)s  %(levelname)s: %(message)s',level=logging.DEBUG)
logging.info("application has started")
#parse configuration parameters for database access and others
config = configparser.ConfigParser()
config.read('config.env')
DB_HOST=config.get('DB','DB_HOST')
DB_DATABASE=config.get('DB','DB_DATABASE')
DB_USERNAME=config.get('DB','DB_USERNAME')
DB_PASSWORD=config.get('DB','DB_PASSWORD')
CASCADE_PATH =config.get('DB','cascade_Path')
model_folder=config.get('DB','model_folder')
IMAGE_WIDTH=config.get('DB','IMAGE_WIDTH')
IMAGE_HEIGHT=config.get('DB','IMAGE_HEIGHT') 
directory=config.get('DB','directory')
#directory where training image will be stored
train_src=config.get('DB','train_src')
#directory where model will be stored
model_folder=config.get('DB','model_folder')
APP_PORT=config.get('DB','APP_PORT')

#auxiliary variables 
#used as a received image counter to name and save images to profile
pic_indx=0
#name of the latest model stored in the folder
latest_models=""

# For face detection we will use the Haar Cascade provided by OpenCV.
#create instance and initilize 
recognizer = cv2.face.LBPHFaceRecognizer_create()

#read all models and picked latest one
files_path = os.path.join(model_folder, '*')
model_files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True) 
if (len(model_files)>0):
    latest_models=model_files[0]
    recognizer.read(latest_models)
else:
    logging.info("No model. This will be the first model")


def adjust_gamma(image, gamma=1.0):
    ''' correct contrast of an image.
        build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
        param: image
        patam: gamma -smoothing factor
        return: corrected image
    '''
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def get_images_and_labels(path):
    '''
    read all images received from users to train model. files are saved in .jpg and filter them
    param: path- where the recived images are saved. its person specific
    return: images and associated labels
    '''
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    indx=0
    for image_path in glob.glob(path+"/*.jpg"):
        # Read images and convert to grayscale
        image = cv2.imread(image_path,0)
        #resize image to optimize training
        image =cv2.resize(image, (int(IMAGE_HEIGHT),int (IMAGE_WIDTH)), interpolation = cv2.INTER_LINEAR)   
        image = np.array(image, 'uint8')
        #get number associated with a particular person. it will be used as label for training model
#subject17.1604191077281.1106_0.jpg
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        #adjust gamma
        image =adjust_gamma(image ,1.3)  
        #append to the list
        images.append(image)
        labels.append(nbr)
    logging.info('Face Recognizer has been trained')
    return images, labels

def star_training(path, modelname):
    ''' The method is used train face recognizer model
        param: path - where the images are stored. will be passed from the caller method
        patam: modelname -model name will be generated dynamically in the caller method
        return: status of training
    '''
    global recognizer
    global model_files
    # Call the get_images_and_labels function and get the face images and the 
    # corresponding labels
    images, labels = get_images_and_labels(path)
    # Perform the tranining
    #if there exists a model then just update it otherwise train a new model
    if (len(model_files)>0):
        recognizer.update(images, np.array(labels))
        logging.info('Face recognizer updating END....')
    else:
        recognizer.train(images, np.array(labels))
        logging.info('Face recognizer training END....')
    recognizer.write(modelname)
    
    return "done"

def get_person_info(personinfo):
    ''' The method is to retrieve person information stored in database
        param: personinfo - JSON objects contains name and phone to query
        return: JSON object contains person name, id etc
    '''
    #default value will be returned if person is not found in DB
    person_id=-88888
    try:
        #create connection object
        connection = mysql.connector.connect(host=DB_HOST, database=DB_DATABASE, user=DB_USERNAME,password=DB_PASSWORD)
        if connection.is_connected():
            #create query
            query="select * from  personinfo where name= %s and phone= %s" 
            values=(str (personinfo["name"]),str (personinfo["phone"]),)
            #create cursor
            cursor_cnt = connection.cursor()
            cursor_cnt.execute(query,values)
            #fetch cursor
            records = cursor_cnt.fetchall()
            #read records
            for row in records:
                person_id=row[0]
            cursor_cnt.close()
    except Error as e:
        logging.error("Error while connecting to MySQL", exc_info=True)
    finally:
        if (connection.is_connected()):
            connection.close()#clean up connection
    return  person_id

@app.route('/api/ownerinfo', methods=['POST'])
def get_owner_info():
    ''' The webservice is to check and retrieve home owner information stored in database
        param: personinfo - JSON objects contains Phone id as owner id/name and its unique
        return: JSON object contains owner id 
    '''
    #recive person information from POST method
    personinfo=request.json
    
    #default value if owner is not found
    person_id=88888
    try:    
        connection = mysql.connector.connect(host=DB_HOST, database=DB_DATABASE, user=DB_USERNAME,password=DB_PASSWORD)
        if connection.is_connected():
            #get owner information with unique phone id used as owner name
            query="select * from  personinfo where owner_name =%s"
            values=(str (personinfo["owner_name"]),)
            print(values)
            cursor_cnt = connection.cursor()
            cursor_cnt.execute(query,values)
            records = cursor_cnt.fetchall()
            for row in records:
                person_id=row[0]
            cursor_cnt.close()
    except Error as e:
        logging.error("Error while connecting to MySQL", exc_info=True)
    finally:
        if (connection.is_connected()):
            connection.close()
    result={"message": str(person_id)}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(result)
    #result=jsonify(result)
    return Response(response=response_pickled, status=200, mimetype="application/json")   


@app.route('/api/addperson', methods=['POST'])
def insert_person_to_db():
    '''The webservice is to insert a person in DB if not exists and create a folder for that person
         in the file system
        param: personinfo - JSON objects contains person information
        return: JSON object contains person id generated by DB
    '''
    #recive person information from POST method
    personinfo=request.json
    result=""
    person_id=-999999
    try:    
        connection = mysql.connector.connect(host=DB_HOST, database=DB_DATABASE, user=DB_USERNAME,password=DB_PASSWORD)
        if connection.is_connected():
            #check if already name and phone exists
            query="select count(1) from  personinfo where name= %s and phone= %s" 
            values=(str (personinfo["name"]),str (personinfo["phone"]),)
            cursor_cnt = connection.cursor()
            cursor_cnt.execute(query,values)
            records = cursor_cnt.fetchall()
            num_val=-999
            for row in records:
                num_val=row[0]
            #if a person does not exist then insert a record
            if(num_val)<=0:
                query="insert into personinfo (name,email,phone,phone_carier,relation,owner_name) values (%s, %s, %s, %s, %s, %s)"
                values=(str (personinfo["name"]),str(personinfo["email"]),str(personinfo["phone"]),str(personinfo["phone_carier"]),str(personinfo["relation"]),str(personinfo["owner_name"]))
                cursor = connection.cursor()
                #execute query
                cursor.execute(query,values)
                #commit the changes 
                connection.commit()
                #get id generated by DB
                person_id=cursor.lastrowid
                cursor.close()
            else:
                person_id=get_person_info(personinfo)
            
    except Error as e:
        logging.error("Error while connecting to MySQL",  exc_info=True)
    finally:
        if (connection.is_connected()):
            connection.close()
    # create a folder to store images of that person        
    if os.path.exists(directory+"/"+personinfo["name"]):
        #claen up previous images since already trained model with them
        shutil.rmtree(directory+"/"+personinfo["name"])
    os.makedirs(directory+"/"+personinfo["name"])
    result={"message": str(person_id)}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(result)
    #result=jsonify(result)
    return Response(response=response_pickled, status=200, mimetype="application/json")

    

@app.route('/api/deleteperson', methods=['POST'])
def delete_person_from_db():
    '''The webservice is to delete a person from DB. It performs soft delete 
        param: personinfo - JSON objects contains person information
        return: deletion status
    '''
    #recive person information from POST method
    personinfo=request.json
    result=""
    deleted_row=-999999
    try:    
        connection = mysql.connector.connect(host=DB_HOST, database=DB_DATABASE, user=DB_USERNAME,password=DB_PASSWORD)
        if connection.is_connected():
            #perform soft delete from DB
            query="update  personinfo set isactive=1 where name= %s and phone= %s" 
            values=(str(personinfo["name"]),str(personinfo["phone"]),)
            cursor_cnt = connection.cursor()
            cursor_cnt.execute(query,values)
            connection.commit() 
            deleted_row=cursor_cnt.rowcount
            cursor_cnt.close()
    except Error as e:
        logging.error("Error while connecting to MySQL", exc_info=True)
    finally:
        if (connection.is_connected()):

            connection.close()
            
    result={"message": str(deleted_row)}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(result)
    #result=jsonify(result)
    return Response(response=response_pickled, status=200, mimetype="application/json")




@app.route('/api/getpicture', methods=['POST'])
def get_picture():
    '''The webservice is to recive images of a person.
        param: personinfo - JSON objects contains person information and image frame
        return: frame receiving status
    '''
    global directory
    request_recived=request.json
    #receives person name from post method
    person_name= request_recived.get("name")
    person_id=request_recived.get("person_id")
    #receives person image from post method
    pic=request_recived.get("pic")
    #decode image base64
    frame = base64.b64decode(pic)
    #generate image name to save in file system
    current_milli_time = time.time() * 1000
    file_name=directory+"/"+person_name+"/"+"subject"+str(person_id)+"."+str(current_milli_time)+"_"+str(pic_indx)+".jpg"
    with open(file_name, 'wb') as f:
        f.write(frame )
    result="frame received"
    result={"message": result}
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(result)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/api/trainmodel', methods=['POST'])
def train_model():
    '''The webservice call perform model training after receving all images.
        param: personinfo - JSON objects contains name
        return: training status
    '''
    global directory
    global latest_models
    global train_src
    
     #receives person name from post method
    request_received=request.json
    person_name= request_received.get("name")
   
    #src directory where images were saved
    src=directory+"/"+person_name
    dates=time.time()
    dest=directory+"/train"+str(dates)
    logging.info('Face recognizer training starts....')
    
    #generate new model name to be saved
    latest_models=directory+"/models/lbp_"+str(round(dates))+".yml"
    #start trainign model and pass the source image directory and name
    star_training(src,latest_models)
    message=""
    if os.path.exists(latest_models):
        message="Training is complete"
    else:
        message="Training failed"
        latest_models=""
    
    result={"message": message,"modelpath":latest_models}
    response_pickled = jsonpickle.encode(result)
    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/api/latestmodel' , methods=['POST'])
def get_latest_models_path():
    
    '''The webservice call is to get latest model path after model training.
        param: 
        return: latest model path
    '''
    global latest_models
    #read all model files
    files_path = os.path.join(model_folder, '*')
    #sort them by date to get latest one
    model_files = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True) 
    latest_models=model_files[0]
    result={"models": latest_models}
    response_pickled = jsonpickle.encode(result)
    return Response(response=response_pickled, status=200, mimetype="application/json")

####################Face Recognition################

#start the Flask application 
app.run(host="0.0.0.0", port=APP_PORT)
    

    