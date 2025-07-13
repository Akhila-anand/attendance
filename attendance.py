import tkinter as tk
import cv2,os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
from tkinter import messagebox

window = tk.Tk()
window.title("Attendance System")
window.configure(background='pink')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

x_cord = 75;
y_cord = 20;
checker=0;

message = tk.Label(window, text="DIT UNIVERSITY" ,bg="white"  ,fg="black"  ,width=20  ,height=2,font=('Times New Roman', 25, 'bold')) 
message.place(x=950, y=660)

message = tk.Label(window, text="ATTENDANCE MANAGEMENT PORTAL" ,bg="pink"  ,fg="black"  ,width=40  ,height=1,font=('Times New Roman', 35, 'bold underline')) 
message.place(x=100, y=10)

lbl = tk.Label(window, text="Enter Your College ID",width=20  ,height=2  ,fg="black"  ,bg="Pink" ,font=('Times New Roman', 25, ' bold ') ) 
lbl.place(x=100-x_cord, y=100-y_cord)

txt = tk.Entry(window,width=30,bg="white" ,fg="blue",font=('Times New Roman', 15, ' bold '))
txt.place(x=150-x_cord, y=200-y_cord)

lbl2 = tk.Label(window, text="Enter Your Name",width=20  ,fg="black"  ,bg="pink"    ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
lbl2.place(x=500-x_cord, y=100-y_cord)

txt2 = tk.Entry(window,width=30  ,bg="white"  ,fg="blue",font=('Times New Roman', 15, ' bold ')  )
txt2.place(x=550-x_cord, y=200-y_cord)

lbl3 = tk.Label(window, text="NOTIFICATION",width=20  ,fg="black"  ,bg="pink"  ,height=2 ,font=('Times New Roman', 25, ' bold ')) 
lbl3.place(x=960-x_cord, y=100-y_cord)

message = tk.Label(window, text="" ,bg="white"  ,fg="blue"  ,width=30  ,height=1, activebackground = "white" ,font=('Times New Roman', 15, ' bold ')) 
message.place(x=975-x_cord, y=200-y_cord)

lbl3 = tk.Label(window, text="ATTENDANCE",width=20  ,fg="white"  ,bg="lightgreen"  ,height=2 ,font=('Times New Roman', 30, ' bold ')) 
lbl3.place(x=60, y=470-y_cord)

message2 = tk.Label(window, text="" ,fg="red"   ,bg="yellow",activeforeground = "green",width=60  ,height=4  ,font=('times', 15, ' bold ')) 
message2.place(x=600, y=470-y_cord)

lbl4 = tk.Label(window, text="STEP 1",width=20  ,fg="green"  ,bg="pink"  ,height=2 ,font=('Times New Roman', 20, ' bold '))
lbl4.place(x=140-x_cord, y=275-y_cord)

lbl5 = tk.Label(window, text="STEP 2",width=20  ,fg="green"  ,bg="pink"  ,height=2 ,font=('Times New Roman', 20, ' bold ')) 
lbl5.place(x=545-x_cord, y=275-y_cord)

lbl6 = tk.Label(window, text="STEP 3",width=20  ,fg="green"  ,bg="pink"  ,height=2 ,font=('Times New Roman', 20, ' bold ')) 
lbl6.place(x=900-x_cord, y=262-y_cord)
 
def clear1():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id = txt.get()
    name = txt2.get()
    if not Id:
        res = "Please enter Id"
        message.configure(text=res)
        tk.messagebox.showwarning("Warning", "Please enter roll number properly.")
        return
    elif not name:
        res = "Please enter Name"
        message.configure(text=res)
        tk.messagebox.showwarning("Warning", "Please enter your name properly.")
        return
    
    if is_number(Id) and name.isalpha():
        harcascadePath = r"haarcascade_frontalface_default.xml"
        if not os.path.exists(harcascadePath):
            tk.messagebox.showerror("Error", "Haarcascade file not found.")
            return
        
        os.makedirs("TrainingImage", exist_ok=True)
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            tk.messagebox.showerror("Error", "Cannot access camera.")
            return
        
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while True:
            ret, img = cam.read()
            if not ret:
                tk.messagebox.showerror("Error", "Failed to capture image.")
                break
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                cv2.imwrite(f"TrainingImage/{name}.{Id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
                cv2.imshow('frame', img)
            
            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 60:
                break
        
        cam.release()
        cv2.destroyAllWindows()
        res = f"Images Saved for ID: {Id}, Name: {name}"
        with open('StudentDetails/StudentDetails.csv', 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([Id, name])
        message.configure(text=res)
    else:
        if is_number(Id):
            res = "Enter Alphabetical Name"
            message.configure(text=res)
        if name.isalpha():
            res = "Enter Numeric Id"
            message.configure(text=res)
            
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    res = "Image Trained"
    clear1();
    clear2();
    message.configure(text= res)
    tk.messagebox.showinfo('Completed','Your model has been trained successfully!!')
    
def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    faces=[]
    Ids=[]
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails/StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown/Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance/Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    res=attendance
    message2.configure(text= res)
    res = "Attendance Taken"
    message.configure(text= res)
    tk.messagebox.showinfo('Completed','Congratulations ! Your attendance has been marked successfully for the day!!')
    
def quit_window():
   MsgBox = tk.messagebox.askquestion ('Exit Application','Are you sure you want to exit the application',icon = 'warning')
   if MsgBox == 'yes':
       tk.messagebox.showinfo("Greetings", "Thank You very much for using our software. Have a nice day ahead!!")
       window.destroy()
    
takeImg = tk.Button(window, text="IMAGE CAPTURE BUTTON", command=TakeImages  ,fg="white"  ,bg="blue"  ,width=25  ,height=2, activebackground = "pink" ,font=('Times New Roman', 15, ' bold '))
takeImg.place(x=145-x_cord, y=325-y_cord)
trainImg = tk.Button(window, text="MODEL TRAINING BUTTON", command=TrainImages  ,fg="white"  ,bg="blue"  ,width=25  ,height=2, activebackground = "pink" ,font=('Times New Roman', 15, ' bold '))
trainImg.place(x=545-x_cord, y=325-y_cord)
trackImg = tk.Button(window, text="ATTENDANCE MARKING BUTTON", command=TrackImages  ,fg="white"  ,bg="red"  ,width=30  ,height=3, activebackground = "pink" ,font=('Times New Roman', 15, ' bold '))
trackImg.place(x=975-x_cord, y=312-y_cord)
quitWindow = tk.Button(window, text="QUIT", command=quit_window  ,fg="white"  ,bg="red"  ,width=10  ,height=2, activebackground = "pink" ,font=('Times New Roman', 15, ' bold '))
quitWindow.place(x=400, y=600-y_cord)

window.mainloop()