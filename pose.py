# -*- coding: utf-8 -*-


from dataclasses import dataclass
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication,  QWidget, QTableWidget, QTableWidgetItem, QVBoxLayout
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import sklearn
from threading import *
import sys
import time

class ThreadedCamera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture("testvideo.mp4")
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)

        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.mp_drawing=mp.solutions.drawing_utils
        self.mp_pose=mp.solutions.pose
        self.scaler=joblib.load('C:\\Users\\ASUS\\Desktop\\Project work\\major project\\UI files\\scaler.save')
        self.lr=joblib.load('C:\\Users\\ASUS\\Desktop\\Project work\\major project\\UI files\\logistic.h5')
       
        
    
    def calculate_angle(self,A,B):
        unit_A=A/np.linalg.norm(A)
        unit_B=B/np.linalg.norm(B)

        return np.arccos(np.dot(unit_A,unit_B))


    def joint_angle(self,df):
        chest_vector=np.transpose(np.array([df['LEFT_SHOULDER_x']-df['RIGHT_SHOULDER_x'],df['LEFT_SHOULDER_y']-df['RIGHT_SHOULDER_y'],df['LEFT_SHOULDER_z']-df['RIGHT_SHOULDER_z']]))
        left_upper_arm_vector=np.transpose(np.array([df['LEFT_ELBOW_x']-df['LEFT_SHOULDER_x'],df['LEFT_ELBOW_y']-df['LEFT_SHOULDER_y'],df['LEFT_ELBOW_z']-df['LEFT_SHOULDER_z']]))
        right_upper_arm_vector=np.transpose(np.array([df['RIGHT_ELBOW_x']-df['RIGHT_SHOULDER_x'],df['RIGHT_ELBOW_y']-df['RIGHT_SHOULDER_y'],df['RIGHT_ELBOW_z']-df['RIGHT_SHOULDER_z']]))
        left_lower_arm_vector=np.transpose(np.array([df['LEFT_WRIST_x']-df['LEFT_ELBOW_x'],df['LEFT_WRIST_y']-df['LEFT_ELBOW_y'],df['LEFT_WRIST_z']-df['LEFT_ELBOW_z']]))
        right_lower_arm_vector=np.transpose(np.array([df['RIGHT_WRIST_x']-df['RIGHT_ELBOW_x'],df['RIGHT_WRIST_y']-df['RIGHT_ELBOW_y'],df['RIGHT_WRIST_z']-df['RIGHT_ELBOW_z']]))
        lowerbody_vector=np.transpose(np.array([df['LEFT_HIP_x']-df['RIGHT_HIP_x'],df['LEFT_HIP_y']-df['RIGHT_HIP_y'],df['LEFT_HIP_z']-df['RIGHT_HIP_z']]))
        left_upper_foot_vector=np.transpose(np.array([df['LEFT_KNEE_x']-df['LEFT_HIP_x'],df['LEFT_KNEE_y']-df['LEFT_HIP_y'],df['LEFT_KNEE_z']-df['LEFT_HIP_z']]))
        right_upper_foot_vector=np.transpose(np.array([df['RIGHT_KNEE_x']-df['RIGHT_HIP_x'],df['RIGHT_KNEE_y']-df['RIGHT_HIP_y'],df['RIGHT_KNEE_z']-df['RIGHT_HIP_z']]))
        left_lower_foot_vector=np.transpose(np.array([df['LEFT_ANKLE_x']-df['LEFT_KNEE_x'],df['LEFT_ANKLE_y']-df['LEFT_KNEE_y'],df['LEFT_ANKLE_z']-df['LEFT_KNEE_z']]))
        right_lower_foot_vector=np.transpose(np.array([df['RIGHT_ANKLE_x']-df['RIGHT_KNEE_x'],df['RIGHT_ANKLE_y']-df['RIGHT_KNEE_y'],df['RIGHT_ANKLE_z']-df['RIGHT_KNEE_z']]))
        mouth_vector=np.transpose(np.array([df['MOUTH_LEFT_x']-df['MOUTH_RIGHT_x'],df['MOUTH_LEFT_y']-df['MOUTH_RIGHT_y'],df['MOUTH_LEFT_z']-df['MOUTH_RIGHT_z']]))
        
        
        df['head_angle']=([self.calculate_angle(mouth_vector[i],(chest_vector[i])) for i in range(len(mouth_vector))])
        df['left_shoulder_angle']=([self.calculate_angle(chest_vector[i],(left_upper_arm_vector[i])) for i in range(len(chest_vector))])
        df['right_shoulder_angle']=([self.calculate_angle(chest_vector[i],(right_upper_arm_vector[i])) for i in range(len(chest_vector))])
        df['left_elbow_angle']=([self.calculate_angle(left_upper_arm_vector[i],(left_lower_arm_vector[i])) for i in range(len(left_upper_arm_vector))])
        df['right_elbow_angle']=([self.calculate_angle(right_upper_arm_vector[i],(right_lower_arm_vector[i])) for i in range(len(right_upper_arm_vector))])
        df['body_angle']=([self.calculate_angle(chest_vector[i],(lowerbody_vector[i])) for i in range(len(chest_vector))])
        df['left_upper_leg_angle']=([self.calculate_angle(lowerbody_vector[i],(left_upper_foot_vector[i])) for i in range(len(lowerbody_vector))])
        df['right_upper_leg_angle']=([self.calculate_angle(lowerbody_vector[i],(right_upper_foot_vector[i]))for i in range(len(lowerbody_vector))])
        df['left_lower_leg_angle']=([self.calculate_angle(left_upper_foot_vector[i],(left_lower_foot_vector[i])) for i in range(len(left_upper_foot_vector))])
        df['right_lower_leg_angle']=([self.calculate_angle(right_upper_foot_vector[i],(right_lower_foot_vector[i])) for i in range(len(right_upper_foot_vector))])
        df['body_angle_two']=([self.calculate_angle(mouth_vector[i],(lowerbody_vector[i])) for i in range(len(lowerbody_vector))])

    def collect_data(self,landmarks,insertiondata):
        curr=[]
        for i in self.mp_pose.PoseLandmark:
            val=i.value
            try:
                currData=[landmarks[val].x,landmarks[val].y,landmarks[val].z,landmarks[val].visibility]
            except:
                currData= [landmarks[val]["x"],landmarks[val]["y"],landmarks[val]["z"],landmarks[val]["visibility"]]
            curr.append(currData[0])
            curr.append(currData[1])
            curr.append(currData[2])
            curr.append(currData[3])
        insertiondata.append(curr)
    
    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)

    def show_frame(self):
        # cv2.imshow('data',self.frame)
        

        global poses
        global pos
        global totalcount
        global correctcount
        
        correctpos=poses[pos]

        with self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
                   
                 
                                    # Recolor image to RGB
            totalcount=totalcount+1
            # cv2.imshow("data",self.frame)
            image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

                                    # Make detection
            results = pose.process(image)

                                    # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                                    # Render detections
            if(results is None or results.pose_landmarks is None or results.pose_landmarks.landmark is None):
                                x,y,z,visibility="x","y","z","visibility"
                                landmarks=[{x: 0.5117800235748291
,y: 0.2662951648235321
,z: -0.12836523354053497
,visibility: 0.9982413053512573}
, {x: 0.5158652067184448
,y: 0.25395065546035767
,z: -0.1065572053194046
,visibility: 0.9971885085105896}
, {x: 0.5186954140663147
,y: 0.254347562789917
,z: -0.10658024251461029
,visibility: 0.99599289894104}
, {x: 0.5212198495864868
,y: 0.2544401288032532
,z: -0.10661140829324722
,visibility: 0.9973624348640442}
, {x: 0.5070594549179077
,y: 0.2536393105983734
,z: -0.1065383106470108
,visibility: 0.9960306286811829}
, {x: 0.5039318203926086
,y: 0.2539697587490082
,z: -0.10661356151103973
,visibility: 0.9944849610328674}
, {x: 0.5009992122650146
,y: 0.254224956035614
,z: -0.10665525496006012
,visibility: 0.9965745210647583}
, {x: 0.5246399641036987
,y: 0.26236534118652344
,z: -0.01909729652106762
,visibility: 0.993423581123352}
, {x: 0.4980285167694092
,y: 0.2643604874610901
,z: -0.016453541815280914
,visibility: 0.9971707463264465}
, {x: 0.5173282027244568
,y: 0.28398412466049194
,z: -0.09823139011859894
,visibility: 0.9968583583831787}
,{ x: 0.5067944526672363
,y: 0.28217560052871704
,z: -0.09756196290254593
,visibility: 0.9964589476585388}
, {x: 0.5463482141494751
,y: 0.3268822431564331
,z: -0.039033107459545135
,visibility: 0.9976558685302734}
, {x: 0.4835125803947449
,y: 0.3221513032913208
,z: -0.0278254933655262
,visibility: 0.997441291809082}
, {x: 0.5406409502029419
,y: 0.22643402218818665
,z: -0.12595845758914948
,visibility: 0.9856775999069214}
,{ x: 0.48346978425979614
,y: 0.2157876193523407
,z: -0.11655652523040771
,visibility: 0.9907386302947998}
, {x: 0.5213525891304016
,y: 0.12515538930892944
,z: -0.14444078505039215
,visibility: 0.9788314700126648}
, {x: 0.5018875598907471
,y: 0.11396202445030212
,z: -0.14616577327251434
,visibility: 0.9844316840171814}
, {x: 0.5178598761558533
,y: 0.09826380759477615
,z: -0.17037004232406616
,visibility: 0.9464482069015503}
, {x: 0.5095193386077881
,y: 0.08864033222198486
,z: -0.1788129210472107
,visibility: 0.9568019509315491}
, {x: 0.5173537731170654
,y: 0.09696957468986511
,z: -0.15952077507972717
,visibility: 0.9399663209915161}
,{ x: 0.5097643136978149
,y: 0.08705857396125793
,z: -0.16766071319580078
,visibility: 0.950014591217041}
, {x: 0.5172666907310486
,y: 0.10724005103111267
,z: -0.14514100551605225
,visibility: 0.9359948039054871}
, {x: 0.5083553194999695
,y: 0.09778067469596863
,z: -0.14835789799690247
,visibility: 0.9530620574951172}
, {x: 0.5207171440124512
,y: 0.5666654109954834
,z: -0.014246786013245583
,visibility: 0.9988834261894226}
, {x: 0.4755555987358093
,y: 0.5494391322135925
,z: 0.014349048025906086
,visibility: 0.9990936517715454}
, {x: 0.5134727358818054
,y: 0.741663932800293
,z: -0.06579648703336716
,visibility: 0.9610702991485596}
, {x: 0.42615318298339844
,y: 0.6123049855232239
,z: -0.29858019948005676
,visibility: 0.9834790229797363}
, {x: 0.5024677515029907
,y: 0.8953871726989746
,z: 0.03195176273584366
,visibility: 0.9762340188026428}
, {x: 0.4843992292881012
,y: 0.6898603439331055
,z: -0.08224476873874664
,visibility: 0.8482667803764343}
, {x: 0.4974837899208069
,y: 0.9128468632698059
,z: 0.03618386387825012
,visibility: 0.8173412680625916}
,{ x: 0.4918200969696045
,y: 0.6857270002365112
,z: -0.06163928657770157
,visibility: 0.7411646246910095}
, {x: 0.5039628744125366
,y: 0.9504320621490479
,z: -0.07508417963981628
,visibility: 0.9465537071228027
}, {x: 0.48542967438697815
,y: 0.7671421766281128
,z: -0.12287615239620209
,visibility: 0.8403297662734985}
]
            else:       
                landmarks = results.pose_landmarks.landmark
                                # print(landmarks)
                            # if(landmarks is None):
                            #     landmarks=np.zeros(132)
            data=[]
            columns=[]
            for landmrk in self.mp_pose.PoseLandmark:
                val=str(landmrk).split(".")[1]
                columns.append(val+"_x")
                columns.append(val+"_y")
                columns.append(val+"_z")
                columns.append(val+"_visiblity")
                                    

            self.collect_data(landmarks,data)
            test_frame=pd.DataFrame(data=data,columns=columns)
            self.joint_angle(test_frame)
            test_frame=self.scaler.transform(test_frame)
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                )               


            prediction=self.lr.predict(test_frame)[0]
            y_pred_prob = self.lr.predict_proba(test_frame)
            ix = y_pred_prob.argmax(1).item()
                            
            value=y_pred_prob[0,ix]
            print(value)
            threshold=0.9700
            if(value<threshold):
                prediction="No pose detected"
                                    
                                    
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (0,20)
            fontScale              = 1
            fontColor              = (255,0,0)
            thickness              = 3
            lineType               = 2
                                    
            cv2.putText(image,prediction, 
                            bottomLeftCornerOfText, 
                                    font, 
                                    fontScale,
                                    fontColor,
                                    thickness,
                                    lineType)
            cv2.imshow('Mediapipe Feed', image)
            if(correctpos==prediction):
                correctcount=correctcount+1
                print("Correct Count ",correctcount)
            cv2.waitKey(self.FPS_MS)
            

            print(f'predicted class = {prediction} and confidence = {y_pred_prob[0,ix]:.2%}')
                    
    

class QtCapture(QtWidgets.QWidget):
    def __init__(self, *args):
        super(QtWidgets.QWidget, self).__init__()

        self.fps = 10
        self.poses=["bhujangasana","padamasana","shavasana","tadasana","trikonasana","vrikashasana"]
        self.pos=-1
        self.totalcount=0
        self.correctcount=0
        self.data=[]
        self.cap = cv2.VideoCapture("trikon.mp4")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.time=0
        self.video_frame = QtWidgets.QLabel()
        self.mp_drawing=mp.solutions.drawing_utils
        self.mp_pose=mp.solutions.pose
        self.scaler=joblib.load('C:\\Users\\ASUS\\Desktop\\Project work\\major project\\UI files\\scaler.save')
        self.lr=joblib.load('C:\\Users\\ASUS\\Desktop\\Project work\\major project\\UI files\\logistic.h5')
        lay = QtWidgets.QVBoxLayout()
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.video_frame)
        self.setLayout(lay)
        self.videothread=Thread(target=self.mloperation)
        self.videothreadstopper=False
        self.cam=None
        self.camStopper=True
        # ------ Modification ------ #
        self.isCapturing = False
        self.ith_frame = 1
        # ------ Modification ------ #

    def setFPS(self, fps):
        self.fps = fps

    def timerval(self):
        global data
        global totalcount
        global correctcount
        global pos
        global poses        
        if(self.time%15==0 and self.time!=0):
            if(pos==len(poses)-1):
                global capturer
                capturer=False
                self.timer2.stop()
                return
            data.append({"name":poses[pos],"total count":totalcount,"correct count":correctcount,"accuracy":(correctcount/totalcount)*100})
            pos=int(self.time/15)
            
            totalcount=0
            correctcount=0
        self.time=self.time+1

    def calculate_angle(self,A,B):
        unit_A=A/np.linalg.norm(A)
        unit_B=B/np.linalg.norm(B)

        return np.arccos(np.dot(unit_A,unit_B))


    def joint_angle(self,df):
        chest_vector=np.transpose(np.array([df['LEFT_SHOULDER_x']-df['RIGHT_SHOULDER_x'],df['LEFT_SHOULDER_y']-df['RIGHT_SHOULDER_y'],df['LEFT_SHOULDER_z']-df['RIGHT_SHOULDER_z']]))
        left_upper_arm_vector=np.transpose(np.array([df['LEFT_ELBOW_x']-df['LEFT_SHOULDER_x'],df['LEFT_ELBOW_y']-df['LEFT_SHOULDER_y'],df['LEFT_ELBOW_z']-df['LEFT_SHOULDER_z']]))
        right_upper_arm_vector=np.transpose(np.array([df['RIGHT_ELBOW_x']-df['RIGHT_SHOULDER_x'],df['RIGHT_ELBOW_y']-df['RIGHT_SHOULDER_y'],df['RIGHT_ELBOW_z']-df['RIGHT_SHOULDER_z']]))
        left_lower_arm_vector=np.transpose(np.array([df['LEFT_WRIST_x']-df['LEFT_ELBOW_x'],df['LEFT_WRIST_y']-df['LEFT_ELBOW_y'],df['LEFT_WRIST_z']-df['LEFT_ELBOW_z']]))
        right_lower_arm_vector=np.transpose(np.array([df['RIGHT_WRIST_x']-df['RIGHT_ELBOW_x'],df['RIGHT_WRIST_y']-df['RIGHT_ELBOW_y'],df['RIGHT_WRIST_z']-df['RIGHT_ELBOW_z']]))
        lowerbody_vector=np.transpose(np.array([df['LEFT_HIP_x']-df['RIGHT_HIP_x'],df['LEFT_HIP_y']-df['RIGHT_HIP_y'],df['LEFT_HIP_z']-df['RIGHT_HIP_z']]))
        left_upper_foot_vector=np.transpose(np.array([df['LEFT_KNEE_x']-df['LEFT_HIP_x'],df['LEFT_KNEE_y']-df['LEFT_HIP_y'],df['LEFT_KNEE_z']-df['LEFT_HIP_z']]))
        right_upper_foot_vector=np.transpose(np.array([df['RIGHT_KNEE_x']-df['RIGHT_HIP_x'],df['RIGHT_KNEE_y']-df['RIGHT_HIP_y'],df['RIGHT_KNEE_z']-df['RIGHT_HIP_z']]))
        left_lower_foot_vector=np.transpose(np.array([df['LEFT_ANKLE_x']-df['LEFT_KNEE_x'],df['LEFT_ANKLE_y']-df['LEFT_KNEE_y'],df['LEFT_ANKLE_z']-df['LEFT_KNEE_z']]))
        right_lower_foot_vector=np.transpose(np.array([df['RIGHT_ANKLE_x']-df['RIGHT_KNEE_x'],df['RIGHT_ANKLE_y']-df['RIGHT_KNEE_y'],df['RIGHT_ANKLE_z']-df['RIGHT_KNEE_z']]))
        mouth_vector=np.transpose(np.array([df['MOUTH_LEFT_x']-df['MOUTH_RIGHT_x'],df['MOUTH_LEFT_y']-df['MOUTH_RIGHT_y'],df['MOUTH_LEFT_z']-df['MOUTH_RIGHT_z']]))
        
        
        df['head_angle']=([self.calculate_angle(mouth_vector[i],(chest_vector[i])) for i in range(len(mouth_vector))])
        df['left_shoulder_angle']=([self.calculate_angle(chest_vector[i],(left_upper_arm_vector[i])) for i in range(len(chest_vector))])
        df['right_shoulder_angle']=([self.calculate_angle(chest_vector[i],(right_upper_arm_vector[i])) for i in range(len(chest_vector))])
        df['left_elbow_angle']=([self.calculate_angle(left_upper_arm_vector[i],(left_lower_arm_vector[i])) for i in range(len(left_upper_arm_vector))])
        df['right_elbow_angle']=([self.calculate_angle(right_upper_arm_vector[i],(right_lower_arm_vector[i])) for i in range(len(right_upper_arm_vector))])
        df['body_angle']=([self.calculate_angle(chest_vector[i],(lowerbody_vector[i])) for i in range(len(chest_vector))])
        df['left_upper_leg_angle']=([self.calculate_angle(lowerbody_vector[i],(left_upper_foot_vector[i])) for i in range(len(lowerbody_vector))])
        df['right_upper_leg_angle']=([self.calculate_angle(lowerbody_vector[i],(right_upper_foot_vector[i]))for i in range(len(lowerbody_vector))])
        df['left_lower_leg_angle']=([self.calculate_angle(left_upper_foot_vector[i],(left_lower_foot_vector[i])) for i in range(len(left_upper_foot_vector))])
        df['right_lower_leg_angle']=([self.calculate_angle(right_upper_foot_vector[i],(right_lower_foot_vector[i])) for i in range(len(right_upper_foot_vector))])
        df['body_angle_two']=([self.calculate_angle(mouth_vector[i],(lowerbody_vector[i])) for i in range(len(lowerbody_vector))])

    def collect_data(self,landmarks,data):
        curr=[]
        for i in self.mp_pose.PoseLandmark:
            val=i.value
            try:
                currData=[landmarks[val].x,landmarks[val].y,landmarks[val].z,landmarks[val].visibility]
            except:
                currData= [landmarks[val]["x"],landmarks[val]["y"],landmarks[val]["z"],landmarks[val]["visibility"]]
            curr.append(currData[0])
            curr.append(currData[1])
            curr.append(currData[2])
            curr.append(currData[3])
        data.append(curr)
    

    def mloperation(self):
        while(1):
            print(self.time)
            self.pos=int(self.time/15)
            if(self.pos==len(self.poses)):
                self.timer2.stop()
                break
            # if(self.pos==len(self.poses)):
            #     break
            # if(self.time%15 == 0):
            #     if(self.pos==-1):
            #         self.pos=self.pos+1
            #     else:
            #         print({"name":self.poses[self.pos],"total count":self.totalcount,"correct count":self.correctcount,"accuracy":(self.correctcount/self.totalcount)*100})
            #         self.pos=self.pos+1
            if(self.videothreadstopper==False):
                with self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
                    ret, frame = self.cap.read()
                    if ret==True:
                                    # Recolor image to RGB
                            self.totalcount=self.totalcount+1
                            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image.flags.writeable = False

                                    # Make detection
                            results = pose.process(image)

                                    # Recolor back to BGR
                            image.flags.writeable = True
                            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                                    # Render detections
                            if(results is None or results.pose_landmarks is None or results.pose_landmarks.landmark is None):
                                x,y,z,visibility="x","y","z","visibility"
                                landmarks=[{x: 0.5117800235748291
,y: 0.2662951648235321
,z: -0.12836523354053497
,visibility: 0.9982413053512573}
, {x: 0.5158652067184448
,y: 0.25395065546035767
,z: -0.1065572053194046
,visibility: 0.9971885085105896}
, {x: 0.5186954140663147
,y: 0.254347562789917
,z: -0.10658024251461029
,visibility: 0.99599289894104}
, {x: 0.5212198495864868
,y: 0.2544401288032532
,z: -0.10661140829324722
,visibility: 0.9973624348640442}
, {x: 0.5070594549179077
,y: 0.2536393105983734
,z: -0.1065383106470108
,visibility: 0.9960306286811829}
, {x: 0.5039318203926086
,y: 0.2539697587490082
,z: -0.10661356151103973
,visibility: 0.9944849610328674}
, {x: 0.5009992122650146
,y: 0.254224956035614
,z: -0.10665525496006012
,visibility: 0.9965745210647583}
, {x: 0.5246399641036987
,y: 0.26236534118652344
,z: -0.01909729652106762
,visibility: 0.993423581123352}
, {x: 0.4980285167694092
,y: 0.2643604874610901
,z: -0.016453541815280914
,visibility: 0.9971707463264465}
, {x: 0.5173282027244568
,y: 0.28398412466049194
,z: -0.09823139011859894
,visibility: 0.9968583583831787}
,{ x: 0.5067944526672363
,y: 0.28217560052871704
,z: -0.09756196290254593
,visibility: 0.9964589476585388}
, {x: 0.5463482141494751
,y: 0.3268822431564331
,z: -0.039033107459545135
,visibility: 0.9976558685302734}
, {x: 0.4835125803947449
,y: 0.3221513032913208
,z: -0.0278254933655262
,visibility: 0.997441291809082}
, {x: 0.5406409502029419
,y: 0.22643402218818665
,z: -0.12595845758914948
,visibility: 0.9856775999069214}
,{ x: 0.48346978425979614
,y: 0.2157876193523407
,z: -0.11655652523040771
,visibility: 0.9907386302947998}
, {x: 0.5213525891304016
,y: 0.12515538930892944
,z: -0.14444078505039215
,visibility: 0.9788314700126648}
, {x: 0.5018875598907471
,y: 0.11396202445030212
,z: -0.14616577327251434
,visibility: 0.9844316840171814}
, {x: 0.5178598761558533
,y: 0.09826380759477615
,z: -0.17037004232406616
,visibility: 0.9464482069015503}
, {x: 0.5095193386077881
,y: 0.08864033222198486
,z: -0.1788129210472107
,visibility: 0.9568019509315491}
, {x: 0.5173537731170654
,y: 0.09696957468986511
,z: -0.15952077507972717
,visibility: 0.9399663209915161}
,{ x: 0.5097643136978149
,y: 0.08705857396125793
,z: -0.16766071319580078
,visibility: 0.950014591217041}
, {x: 0.5172666907310486
,y: 0.10724005103111267
,z: -0.14514100551605225
,visibility: 0.9359948039054871}
, {x: 0.5083553194999695
,y: 0.09778067469596863
,z: -0.14835789799690247
,visibility: 0.9530620574951172}
, {x: 0.5207171440124512
,y: 0.5666654109954834
,z: -0.014246786013245583
,visibility: 0.9988834261894226}
, {x: 0.4755555987358093
,y: 0.5494391322135925
,z: 0.014349048025906086
,visibility: 0.9990936517715454}
, {x: 0.5134727358818054
,y: 0.741663932800293
,z: -0.06579648703336716
,visibility: 0.9610702991485596}
, {x: 0.42615318298339844
,y: 0.6123049855232239
,z: -0.29858019948005676
,visibility: 0.9834790229797363}
, {x: 0.5024677515029907
,y: 0.8953871726989746
,z: 0.03195176273584366
,visibility: 0.9762340188026428}
, {x: 0.4843992292881012
,y: 0.6898603439331055
,z: -0.08224476873874664
,visibility: 0.8482667803764343}
, {x: 0.4974837899208069
,y: 0.9128468632698059
,z: 0.03618386387825012
,visibility: 0.8173412680625916}
,{ x: 0.4918200969696045
,y: 0.6857270002365112
,z: -0.06163928657770157
,visibility: 0.7411646246910095}
, {x: 0.5039628744125366
,y: 0.9504320621490479
,z: -0.07508417963981628
,visibility: 0.9465537071228027
}, {x: 0.48542967438697815
,y: 0.7671421766281128
,z: -0.12287615239620209
,visibility: 0.8403297662734985}
]
                            else:       
                                landmarks = results.pose_landmarks.landmark
                                # print(landmarks)
                            # if(landmarks is None):
                            #     landmarks=np.zeros(132)
                            data=[]
                            columns=[]
                            for landmrk in self.mp_pose.PoseLandmark:
                                val=str(landmrk).split(".")[1]
                                columns.append(val+"_x")
                                columns.append(val+"_y")
                                columns.append(val+"_z")
                                columns.append(val+"_visiblity")
                                    

                            self.collect_data(landmarks,data)
                            test_frame=pd.DataFrame(data=data,columns=columns)
                            self.joint_angle(test_frame)
                            test_frame=self.scaler.transform(test_frame)
                            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                )               


                            prediction=self.lr.predict(test_frame)[0]
                            y_pred_prob = self.lr.predict_proba(test_frame)
                            ix = y_pred_prob.argmax(1).item()
                            
                            value=y_pred_prob[0,ix]
                            print(value)
                            threshold=0.9700
                            if(value<threshold):
                                prediction="No pose detected"
                                    
                                    
                            font                   = cv2.FONT_HERSHEY_SIMPLEX
                            bottomLeftCornerOfText = (0,20)
                            fontScale              = 1
                            fontColor              = (255,0,0)
                            thickness              = 3
                            lineType               = 2
                                    
                            cv2.putText(image,prediction, 
                                bottomLeftCornerOfText, 
                                    font, 
                                    fontScale,
                                    fontColor,
                                    thickness,
                                    lineType)
                            cv2.imshow('Mediapipe Feed', image)
                            if(self.poses[self.pos]==prediction):
                                self.correctcount=self.correctcount+1
                                print("Correct Count Value ", self.correctcount)

                            print(f'predicted class = {prediction} and confidence = {y_pred_prob[0,ix]:.2%}')
                            if cv2.waitKey(10) & 0xFF == ord('q'):
                                break
                    else:
                        break
                    
            else:
                break

    def nextFrameSlot(self):

        ret, frame = self.cap.read()

        # # ------ Modification ------ #
        # # Save images if isCapturing
        # if self.isCapturing:
        #     print(self.isCapturing)    
            
        # # ------ Modification ------ #

        # # My webcam yields frames in BGR format
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(frame, 
        #         str(self.time), 
        #         (50, 50), 
        #         font, 1, 
        #         (0, 255, 255), 
        #         2, 
        #         cv2.LINE_4)
        # img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
        # pix = QtGui.QPixmap.fromImage(img)
        # self.video_frame.setPixmap(pix)

        
        with self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
            
                
            if ret==True:
                        # Recolor image to RGB
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                        # Make detection
                results = pose.process(image)

                        # Recolor back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        # Render detections
                landmarks = results.pose_landmarks.landmark
                if(landmarks is None):
                    pass
                data=[]
                columns=[]
                for landmrk in self.mp_pose.PoseLandmark:
                    val=str(landmrk).split(".")[1]
                    columns.append(val+"_x")
                    columns.append(val+"_y")
                    columns.append(val+"_z")
                    columns.append(val+"_visiblity")
                        

                self.collect_data(landmarks,data)
                test_frame=pd.DataFrame(data=data,columns=columns)
                self.joint_angle(test_frame)
                test_frame=self.scaler.transform(test_frame)
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                    self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                    )               


                prediction=self.lr.predict(test_frame)[0]
                y_pred_prob = self.lr.predict_proba(test_frame)
                ix = y_pred_prob.argmax(1).item()
                
                value=y_pred_prob[0,ix]
                print(value)
                threshold=0.9700
                if(value<threshold):
                    prediction="No pose detected"
                        
                        
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (0,20)
                fontScale              = 1
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
                        
                cv2.putText(image,prediction, 
                    bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)
                cv2.imshow('Mediapipe Feed', image)
                        
                print(f'predicted class = {prediction} and confidence = {y_pred_prob[0,ix]:.2%}')

          


    def start(self):
        self.timer = QtCore.QTimer()
        self.timer2=QtCore.QTimer()
        self.timer2.timeout.connect(self.timerval)
        # self.timer.timeout.connect(self.nextFrameSlot)
        # self.timer.start(1000./self.fps)
        self.timer2.start(1000)
        # self.videothread.start()

    def stop(self):
        self.timer.stop()
        # self.videothreadstopper=True

    # ------ Modification ------ #
    def capture(self):
        if not self.isCapturing:
            self.isCapturing = True
        else:
            self.isCapturing = False
    # ------ Modification ------ #

    def deleteLater(self):
        self.cap.release()
        # self.videothread.terminate()
        super(QtWidgets.QWidget, self).deleteLater()




class Ui_MainWindow(object):
        def setupUi(self, MainWindow):
                MainWindow.setObjectName("MainWindow")
                MainWindow.resize(1180, 980)
                MainWindow.setMinimumSize(QtCore.QSize(1180, 980))
                MainWindow.setMaximumSize(QtCore.QSize(1180, 980))
                self.centralwidget = QtWidgets.QWidget(MainWindow)
                self.centralwidget.setObjectName("centralwidget")
                self.StartButton = QtWidgets.QPushButton(self.centralwidget)
                self.StartButton.setGeometry(QtCore.QRect(470, 160, 311, 61))
                self.StartButton.setStyleSheet("color:red;\n"
        "font-size:30px;")
                self.StartButton.setObjectName("StartButton")
                self.StopButton = QtWidgets.QPushButton(self.centralwidget)
                self.StopButton.setGeometry(QtCore.QRect(470, 280, 311, 61))
                self.StopButton.setStyleSheet("color:red;\n"
        "font-size:30px;")
                
                
                global data
                global poses
                global pos
                global totalcount
                global correctcount
                global capturer
                capturer=True
                data=[]
                poses=["bhujangasana","padamasana","shavasana","tadasana","trikonasana","vrikashasana"]
                pos=0
                totalcount=0
                correctcount=0



                self.StopButton.setObjectName("StopButton")
                self.EndButton = QtWidgets.QPushButton(self.centralwidget)
                self.EndButton.setGeometry(QtCore.QRect(460, 390, 311, 61))
                self.EndButton.setStyleSheet("color:red;\n"
        "font-size:30px;")
                self.EndButton.setObjectName("EndButton")

                self.StartButton.clicked.connect(self.startCapture)
                self.EndButton.clicked.connect(self.endCapture)
                self.StopButton.clicked.connect(self.stopCapture)
                self.capture = None

                self.Heading = QtWidgets.QLabel(self.centralwidget)
                self.Heading.setGeometry(QtCore.QRect(200, 10, 851, 111))
                self.Heading.setObjectName("Heading")
                self.scrollArea = QtWidgets.QScrollArea(self.centralwidget)
                self.scrollArea.setGeometry(QtCore.QRect(100, 510, 1041, 401))
                self.scrollArea.setWidgetResizable(True)
                self.scrollArea.setObjectName("scrollArea")
                self.scrollAreaWidgetContents = QtWidgets.QWidget()
                self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 1039, 399))
                self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
                self.scrollArea.setWidget(self.scrollAreaWidgetContents)
                MainWindow.setCentralWidget(self.centralwidget)
                self.statusbar = QtWidgets.QStatusBar(MainWindow)
                self.statusbar.setObjectName("statusbar")
                MainWindow.setStatusBar(self.statusbar)
                # self.creatingTables()
                self.retranslateUi(MainWindow)
                QtCore.QMetaObject.connectSlotsByName(MainWindow)

        def retranslateUi(self, MainWindow):
                _translate = QtCore.QCoreApplication.translate
                MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
                self.StartButton.setText(_translate("MainWindow", "Start"))
                self.StopButton.setText(_translate("MainWindow", "Stop/Show"))
                self.EndButton.setText(_translate("MainWindow", "End"))
                self.Heading.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:26pt; font-weight:600;\">Watch Out Your Pose </span></p></body></html>"))
        
        def creatingTables(self,data):
                        print("data obtained is",data)
                        self.tableWidget = QTableWidget()
                        self.containerTableWidget=QWidget()
                        self.tableWidget.setRowCount(len(data)+1)
                        self.tableWidget.setColumnCount(4)
                
                        self.tableWidget.setItem(0,0, QTableWidgetItem("Asana"))
                        self.tableWidget.setItem(0,1, QTableWidgetItem("Total Frames Count"))
                        self.tableWidget.setItem(0, 2 , QTableWidgetItem("Correct Frames Count"))
                        self.tableWidget.setItem(0, 3 , QTableWidgetItem("Accuracy"))

                        for i in range(0,len(data)):
                                # print(data[i])
                                # self.tableWidget.setItem(i+1,0, QTableWidgetItem("Asana"))
                                # self.tableWidget.setItem(i+1,1, QTableWidgetItem("Total Frames Count"))
                                # self.tableWidget.setItem(i+1, 2 , QTableWidgetItem("Correct Frames Count"))
                                # self.tableWidget.setItem(i+1, 3 , QTableWidgetItem("Accuracy"))
                                self.tableWidget.setItem(int(i)+1,0, QTableWidgetItem(data[i]['name']))
                                self.tableWidget.setItem(int(i)+1,1, QTableWidgetItem(str(data[i]['total count'])))
                                self.tableWidget.setItem(int(i)+1, 2 , QTableWidgetItem(str(data[i]['correct count'])))
                                self.tableWidget.setItem(int(i)+1, 3 , QTableWidgetItem(str(data[i]['accuracy'])))
                
                        # self.tableWidget.setItem(1,0, QTableWidgetItem("Parwiz"))
                        # self.tableWidget.setItem(1,1, QTableWidgetItem("parwiz@gmail.com"))
                        # self.tableWidget.setItem(1,2, QTableWidgetItem("845845845"))
                        # self.tableWidget.setColumnWidth(1, 200)
                
                        # self.tableWidget.setItem(2, 0, QTableWidgetItem("Ahmad"))
                        # self.tableWidget.setItem(2, 1, QTableWidgetItem("ahmad@gmail.com"))
                        # self.tableWidget.setItem(2, 2, QTableWidgetItem("2232324"))
                
                        # self.tableWidget.setItem(3, 0, QTableWidgetItem("John"))
                        # self.tableWidget.setItem(3, 1, QTableWidgetItem("john@gmail.com"))
                        # self.tableWidget.setItem(3, 2, QTableWidgetItem("2236786782324"))
                
                        # self.tableWidget.setItem(4, 0, QTableWidgetItem("Doe"))
                        # self.tableWidget.setItem(4, 1, QTableWidgetItem("Doe@gmail.com"))
                        # self.tableWidget.setItem(4, 2, QTableWidgetItem("12343445"))

                        # self.tableWidget.setItem(5, 0, QTableWidgetItem("Doe"))
                        # self.tableWidget.setItem(5, 1, QTableWidgetItem("Doe@gmail.com"))
                        # self.tableWidget.setItem(5, 2, QTableWidgetItem("12343445"))


                        # self.tableWidget.setItem(6, 0, QTableWidgetItem("Doe"))
                        # self.tableWidget.setItem(6, 1, QTableWidgetItem("Doe@gmail.com"))
                        # self.tableWidget.setItem(6, 2, QTableWidgetItem("12343445"))


                        # self.tableWidget.setItem(7, 0, QTableWidgetItem("Doe"))
                        # self.tableWidget.setItem(7, 1, QTableWidgetItem("Doe@gmail.com"))
                        # self.tableWidget.setItem(7, 2, QTableWidgetItem("12343445"))


                        # self.tableWidget.setItem(8, 0, QTableWidgetItem("Doe"))
                        # self.tableWidget.setItem(8, 1, QTableWidgetItem("Doe@gmail.com"))
                        # self.tableWidget.setItem(8, 2, QTableWidgetItem("12343445"))


                        # self.tableWidget.setItem(9, 0, QTableWidgetItem("Doe"))
                        # self.tableWidget.setItem(9, 1, QTableWidgetItem("Doe@gmail.com"))
                        # self.tableWidget.setItem(9, 2, QTableWidgetItem("12343445"))


                        # self.tableWidget.setItem(10, 0, QTableWidgetItem("Doe"))
                        # self.tableWidget.setItem(10, 1, QTableWidgetItem("Doe@gmail.com"))
                        # self.tableWidget.setItem(10, 2, QTableWidgetItem("12343445"))


                        # self.tableWidget.setItem(11, 0, QTableWidgetItem("Doe"))
                        # self.tableWidget.setItem(11, 1, QTableWidgetItem("Doe@gmail.com"))
                        # self.tableWidget.setItem(11, 2, QTableWidgetItem("12343445"))
                
                        self.vBoxLayout = QVBoxLayout()
                        self.vBoxLayout.addWidget(self.tableWidget)
                        self.containerTableWidget.setLayout(self.vBoxLayout)
                        # self.scrollArea.setWidget(self.containerTableWidget)
                        self.scrollArea.setWidget(self.tableWidget)

        def startCapture(self):
                if not self.capture:
                        self.capture = QtCapture(0)
                        self.EndButton.clicked.connect(self.capture.stop)
                        # self.capture.setFPS(1)
                        # self.capture.setParent(self)
                        self.capture.setWindowFlags(QtCore.Qt.Tool)
                self.capture.start()
                self.capture.show()

                self.cam=ThreadedCamera()
                global capturer
                while(capturer):
                    try:
                        self.cam.show_frame()
                    except AttributeError:
                        pass



        def stopCapture(self):
                global data
                global capturer
                self.capture.stop()
                self.creatingTables(data)
                self.capture.deleteLater()
                capturer=False

        def endCapture(self):
                self.capture.deleteLater()
                
                self.capture = None

        # ------ Modification ------ #
        def saveCapture(self):
                if self.capture:
                        self.capture.capture()
        # ------ Modification ------ #

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

