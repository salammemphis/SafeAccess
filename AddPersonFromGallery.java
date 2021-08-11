/************************************************
 Copyright (c) 2020 the university of Memphis to present
 All right reserved
 Author: Shahinur Alam
 Email:salam@memphis.edu
 **************************************************
This is an utility activity class  to facilitate personal profile creation. This class is called
 after making sure connection to the server has been established from Main activity. The personal profile is
 created from users demographic information (name, contact) and face images. This class facilitates
 collecting demographics and allows users to browse a recorded video of their face image from
 smartphone gallery. The collected information and images are sent to a server to train face
 recognition model via webservice call.
 Task performs:
 1. Collect user information
 2. Allow users to browse their face images video from smartphone gallery
 3. Read the video file and detect faces
 4. compress the image and encode with base64
 5. Transfer data asynchronously
 6. Initiate model training
 7. Initiate model and data versioning
 8. Replace old model with new one.

 Note: Volley package has been used to make asynchronous communication easy
 */

package edu.memphis.com.safeaccess;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonObjectRequest;
import com.android.volley.toolbox.Volley;
import com.github.hiteshsondhi88.libffmpeg.FFmpeg;
import com.github.hiteshsondhi88.libffmpeg.FFmpegLoadBinaryResponseHandler;
import com.github.hiteshsondhi88.libffmpeg.exceptions.FFmpegNotSupportedException;
import org.json.JSONObject;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import wseemann.media.FFmpegMediaMetadataRetriever; //external packages to read and convert multimedia files
public class AddPersonFromGallery extends AppCompatActivity { //main class
    private CascadeClassifier      mJavaDetector; //Helper class used to detect faces from the images
    private static final String    TAG= "OCVSample::Activity"; // Tag to filter logs
    private Text2Speech txt2Speech;// used to convert text to speech for reading out feedback
    private static String SERVER_URL = "";//server url to post data. Picked from config file
    private static String MODEL_URL ="";//server url where model is running. picked from config file
    private static int RESIZE_WIDTH=0;//width resize large face image. Picked from config file
    private static int RESIZE_HEIGHT=0;// resize large face image. Picked from config file
    private volatile boolean  hasSent= false;// flag to check whether previous frame has received by server
    private JSONObject personinfo=null;//json object to send demographic and image to server via POST method
    private String latestModelPath="";//Path of the latest model after updating it with new images.
    //volatile to make changes visible to all threads
    private volatile int numberOfimageSent=0;// count how many frame has been sent
    private volatile boolean hasTrained=false; //flag to check whether model training has finished or not
    private static  int PERSON_ID; //Unique Id created by Database
    private String uploatPath=""; //path of the recorded video picked by user
    private static int REQUEST_TAKE_GALLERY_VIDEO=2; // Intent specific value
    private FFmpeg fFmpeg; //external package to facilitate reading all format of video file


    /**
     This method is to start the activity.
     parameter:Bundle, to save the instance state so that if phone change orientation user dont lose data
     return: None
     exception: none
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);// call parent class constructor
        setContentView(R.layout.activity_add_person_from_gallery);//select which xml will be rendered
        Button btnPicvideo=findViewById(R.id.btnPickVideo);// action button to pick video
        Button btnSubmit =findViewById(R.id.btnSubmit);// action button to submit data to server
        final TextView txtName=findViewById(R.id.txtName); // Text field to enter Name
        final TextView txtPhone=findViewById(R.id.txtPhone); //Text field to enter Phone
        //getConfigParameters method of Utility class reads values from config file
        SERVER_URL=Utility.getConfigParameters(this,"SERVER_URL"); //read parameter from config file
        MODEL_URL=Utility.getConfigParameters(this,"MODEL_URL"); //read parameter from config file
        RESIZE_WIDTH=Integer.parseInt(Utility.getConfigParameters(this,"RESIZE_WIDTH")); //read parameter from config file
        RESIZE_HEIGHT=Integer.parseInt(Utility.getConfigParameters(this,"RESIZE_HEIGHT")); //read parameter from config file
        txt2Speech = new Text2Speech(getBaseContext()); //instantiate txt2Speech object

        txt2Speech = new Text2Speech(getBaseContext()); //instantiate txt2Speech object
        try {
            loadFFMpeg();// load FFMpeg library
        } catch (FFmpegNotSupportedException e) {
            Log.i("REST","Can not load FFMpeg");
        }

        //fires the event when user press Pick video button
        btnPicvideo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //validates the required fields
                if (txtName.getText().toString().trim()=="" || txtName.getText().toString().trim().contains("Enter Person Name")){
                    txtName.setError( "Required filed!! Enter Person Name" );
                    txtName.setHint("Enter Person Name");
                }else{
                    //Receives information from MainActivity class passed through Intent
                    String recPesroninfo = getIntent().getStringExtra("pesroninfo");
                    String message = getIntent().getStringExtra("message");
                    try{
                        //includes user information in the json object
                        personinfo=new JSONObject(recPesroninfo);
                        personinfo.put("name",txtName.getText().toString().trim().toLowerCase());
                        personinfo.put("email","salam@memphis.edu");
                        personinfo.put("phone",txtPhone.getText().toString().trim());
                        personinfo.put("phone_carier","cricket");
                        personinfo.put("relation","junior");
                        Log.i("REST","Person info: "+personinfo.toString());
                        checkPersonExists();// check whether the person is already in database or new. If new enter a record and return ID

                    }catch (Exception ex){
                    }
                    //Invokes native intent to allow selecting video from gallery
                    Intent intent = new Intent();
                    intent.setType("video/*");
                    intent.setAction(Intent.ACTION_PICK);
                    startActivityForResult(Intent.createChooser(intent,"Select Video"),REQUEST_TAKE_GALLERY_VIDEO);

                }
            }
        });
        //fires the event when user press submit button to post data
        btnSubmit.setOnClickListener(new View.OnClickListener() {
             @Override
            public void onClick(View v) {
                 if (PERSON_ID>0 ) { //validate whether person ID is null or not. If null look at required parameter
                     Log.i("REST","Person id: "+PERSON_ID);
                     try {
                         personinfo.put("person_id", PERSON_ID);
                     } catch (Exception ex) {
                         Log.i("REST","failed to add person ID");
                     }
                     uploadVideo(uploatPath);//Post data to server

                 }else{
                     Toast.makeText(getApplicationContext(), "Check required field", Toast.LENGTH_LONG).show();
                 }
            }
        });

    }

    /**
     This method is to check whether the entered person is already in database/profile or not. if not the a associated
     webservice call will enter a new record in DB and return the unique ID
     parameter: None. Class member variable "personinfo" which has information entered by user
     return:None. update Class member variable PERSON_ID with returned unique identifier generated by DB
     exception: Exception
     */
    public void checkPersonExists(){
        //Volley is an external package to make asynchronous communication easy
        RequestQueue queue = Volley.newRequestQueue(AddPersonFromGallery.this);
        String submitURL=SERVER_URL+"addperson";
        try{
            //invoke the webservice and post data
            JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(Request.Method.POST, submitURL, personinfo,
                    new Response.Listener<JSONObject>() {
                        @Override
                        public void onResponse(JSONObject response) {//data received from webservice
                            try{
                                PERSON_ID= Integer.parseInt(response.get("message").toString());
                                Log.i("REST",personinfo.toString());
                            }catch (Exception ex){
                                Log.i("REST",ex.toString());
                            }
                        }
                    }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Log.i("REST",error.toString());
                }
            });
            queue.add(jsonObjectRequest);


        }catch(Exception ex){
            ex.printStackTrace();
        }
    }

    /**
     this method is used to get the path of selected video image file from asynchronous call
     parameter: requestCode, resultCode, data
     return:None. update class member variable uploatPath
     exception: none
     */
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {//check whether result has returned
            if (requestCode == REQUEST_TAKE_GALLERY_VIDEO) {
                Uri selectedImageUri = data.getData();//get URI of that file
                String filemanagerstring = selectedImageUri.getPath();
                String selectedImagePath = getPath(selectedImageUri);
                if (selectedImagePath != null) {
                    uploatPath=selectedImagePath;

                   Log.i("REST",selectedImagePath);

                }
            }
        }
    }
    /**
     this helper method is used to retrieve all selected files and absolute paths. Make sure you have
     set up read permission for external storage in AndroidManifest.xml
     parameter: URI, all selected files location
     return:absolute path of selected file
     exception: none
     */
    public String getPath(Uri uri) {
        String[] projection = { MediaStore.Images.Media.DATA };
        Cursor cursor = getContentResolver().query(uri, projection, null, null, null);
        if (cursor == null) return null;
        int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
        cursor.moveToFirst();
        String s=cursor.getString(column_index);
        cursor.close();
        return s;
    }

    /**
     this methods is used to send the data to the server to train the model
     parameter: Path of the selected video file
     return:None.
     exception: none
     */
    public void uploadVideo(String path){

        File camfile=new File(path); //checks whether the file exists
        if (! camfile.exists()){
            Log.i("REST","file not found "+uploatPath);
        }else{
            Log.i("REST","file exists "+uploatPath);
        }
        FFmpegMediaMetadataRetriever mmr = new FFmpegMediaMetadataRetriever(); //external package to read video file
        try {
            mmr.setDataSource(uploatPath);// set data source path
        }catch (Exception e) {
            System.out.println("Exception= "+e);
        }
        //duration of the video file
        long duration = mmr.getMetadata().getLong("duration");
        int numberOfFrame =50;  //number of frame with face will be sent
        long frameRate= duration/numberOfFrame;
        String imgname=uploatPath.replace(".mp4",".jpg");
        if (PERSON_ID > 0) {//double chaek person id has valid value
            try {
                personinfo.put("person_id", PERSON_ID);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
            //Log.i("REST", personinfo.toString());
        }
        //read frames from video file and send it to server
        for (int frameIterator = 0; frameIterator < numberOfFrame; frameIterator++) {

            //if (hasSent == false) {
                //read video frame as bitmap image
                Bitmap b = mmr.getFrameAtTime((long) frameRate * frameIterator*1000, FFmpegMediaMetadataRetriever.OPTION_CLOSEST);
                try {
                    //convert bitmap image to Mat so that we can use opencv to process it
                    Mat imgOr = new Mat (b.getWidth(), b.getHeight(), CvType.CV_8UC1);
                    Utils.bitmapToMat(b, imgOr);
                    //convert RGB image to Gray
                    Imgproc.cvtColor(imgOr, imgOr, Imgproc.COLOR_RGB2GRAY);
                    //correct orientation of image
                    Core.rotate(imgOr, imgOr, Core.ROTATE_90_COUNTERCLOCKWISE);
                    Imgcodecs.imwrite(imgname +(int)(frameIterator%10)+ "_1.jpg", imgOr);
                    Log.i("REST",""+hasSent +" "+ numberOfFrame +" "+frameRate);
                    Log.i("REST", "sending.." + frameRate * frameIterator);
                    processFrame(imgOr);// Process each frame and send to server
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
             if (hasSent){//give a pause for the acknowledge for current frame
                 SystemClock.sleep(100);
             }
        }
        if( hasTrained==false){// if all frames are sent call for training model
            trainModel();// call training webservice
            hasTrained=true;
        }
    }

    /**
     this method is used to process each frame. it finds face, crop faces
     parameter: Frame
     return:None.
     exception: none
     */
    public void processFrame(Mat frame) {
        MatOfRect faces = new MatOfRect();
        if (mJavaDetector != null) { //make sure face detector is not null
            mJavaDetector.detectMultiScale(frame, faces); //detect face
            Log.i(TAG, "called face detector");
        }
        Rect[] facesArray = faces.toArray();//get all detected faces
        Log.i("REST","number of face "+facesArray.length);
        String sonifyMessage = "";
        for (int faceIterator = 0; faceIterator < facesArray.length; faceIterator++) {// iterate through each face
            //if (hasSent == false) {
                hasSent = true;
                Mat mcrop=frame.submat(facesArray[faceIterator]);
                if (mcrop.rows()>RESIZE_WIDTH || mcrop.cols()>RESIZE_HEIGHT){//resize face if it is big
                    Imgproc.resize(mcrop,mcrop, new Size((int)mcrop.rows()/2,(int)mcrop.cols()/2));
                }

                sendImage(mcrop);// send it to server
        }
    }

    /**
     this method is used to transfer each frame. It compress the image to utilize the network
     bandwith, performs base64 encoding to obfuscate the data
     parameter: Frame
     return:None.
     exception: none
     */
    public void sendImage(Mat img){
        RequestQueue queue = Volley.newRequestQueue(AddPersonFromGallery.this);
        String submitURL=SERVER_URL+"getpicture";
        boolean isConnected=false;
        try{
            //Mat resimg=new  Mat();
            Mat resimg=img;
            MatOfByte mb=new MatOfByte();
            MatOfInt  params90 = new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 90);// compress the image
            Imgcodecs.imencode(".jpg",resimg,mb, params90);
            byte[] byteArray = mb.toArray();// convert to a byte array
            //String encodedImage = Base64.encode(byteArray);
            //String imgString=new String(byteArray);
            String imgString=Base64.encodeToString(byteArray, Base64.DEFAULT); // perform base64 encoding
            personinfo.put("pic",imgString);// add picture to the json data
            JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(Request.Method.POST, submitURL, personinfo,
                    new Response.Listener<JSONObject>() {
                        @Override
                        public void onResponse(JSONObject response) {
                            try{

                                ///personinfo.put("person_id",response.get("message"));
                                //personId= Integer.parseInt(response.get("message").toString());
                                if(response.get("message").toString().contains("received")){//acknowledgement from server
                                    hasSent=false;
                                    numberOfimageSent++;

                                }
                                Log.i("REST",response.get("message").toString());
                                hasSent=false;
                                //Log.i("REST",personinfo.toString());
                            }catch (Exception ex){

                            }
                        }
                    }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Log.i("REST",error.toString());
                }
            });
            queue.add(jsonObjectRequest);
        }catch(Exception ex){
            ex.printStackTrace();
        }
    }

    /**
     this method is used to invoke the webservice to start training the model
     parameter: None.
     return:None. update class member latestModelPath
     exception: none
     */
    public void trainModel(){
        RequestQueue queue = Volley.newRequestQueue(AddPersonFromGallery.this);
        String submitURL=SERVER_URL+"trainmodel";

        boolean isConnected=false;
        try{
            JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(Request.Method.POST, submitURL, personinfo,
                    new Response.Listener<JSONObject>() {
                        @Override
                        public void onResponse(JSONObject response) {
                            try{

                                ///personinfo.put("person_id",response.get("message"));
                                if (response.get("message").toString().contains("Training")){
                                    txt2Speech.sonify("Model has been trained with new images");
                                    latestModelPath=response.get("modelpath").toString();
                                    Log.i("REST",latestModelPath);
                                    useLatestModel();
                                }
                                Log.i("REST",personinfo.toString());
                            }catch (Exception ex){

                            }
                        }
                    }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Log.i("REST",error.toString());
                }
            });
            queue.add(jsonObjectRequest);


        }catch(Exception ex){
            ex.printStackTrace();
        }
    }

    /**
     this methods is used to notify that model has been updated and ready to use
     parameter: None. Class member latestModelPath
     return:None.
     exception: none
     */
    public void useLatestModel(){
        RequestQueue queue = Volley.newRequestQueue(AddPersonFromGallery.this);

        try{
            personinfo.put("modelpath",latestModelPath);
            JsonObjectRequest jsonObjectRequest = new JsonObjectRequest(Request.Method.POST, MODEL_URL, personinfo,
                    new Response.Listener<JSONObject>() {
                        @Override
                        public void onResponse(JSONObject response) {
                            try{
                                ///personinfo.put("person_id",response.get("message"));
                                if (response.get("message").toString().contains("latest")){
                                    txt2Speech.sonify("New model will be used soon");
                                    finish();
                                }
                                Log.i("REST",personinfo.toString());
                            }catch (Exception ex){
                            }
                        }
                    }, new Response.ErrorListener() {
                @Override
                public void onErrorResponse(VolleyError error) {
                    Log.i("REST",error.toString());
                }
            });
            queue.add(jsonObjectRequest);
        }catch(Exception ex){
            ex.printStackTrace();
        }
    }

    /**
     This method load and initialize FFmpeg instance which is used to read and convert video files
     parameter:none
     return: None
     exception: FFmpegNotSupportedException
     */
    private void loadFFMpeg() throws FFmpegNotSupportedException {
        if (fFmpeg==null){
            fFmpeg=FFmpeg.getInstance(this);
            fFmpeg.loadBinary(new FFmpegLoadBinaryResponseHandler() {
                @Override
                public void onFailure() {
                    Log.i("REST","Failed LOADED");
                }

                @Override
                public void onSuccess() {
                    Log.i("REST","LOADED");
                }

                @Override
                public void onStart() {

                }

                @Override
                public void onFinish() {

                }
            });

        }
    }

    /**
     This method load and initialize opencv library
     parameter:none
     return: None
     exception: FFmpegNotSupportedException
     */
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    try {
                        InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt_tree);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();
                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());


                    } catch (IOException e) {

                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }


                } break;

                default:
                {
                    super.onManagerConnected(status);
                    Log.e(TAG, "Failed to load opencv");

                } break;
            }
        }
    };

    @Override
    public void onPause() {
        super.onPause();
        txt2Speech.close();
        //System.exit(0);
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }
    @Override
    public void onDestroy() {
        super.onDestroy();
        txt2Speech.close();
        System.exit(0);
    }
}
