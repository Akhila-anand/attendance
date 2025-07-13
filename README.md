# Student Attendance System using Face Recognition

This is a Python-based attendance system that uses face recognition to automate the attendance process for students. It captures student faces, trains a model, and marks attendance by detecting faces in real-time using OpenCV.

##  Features

-  Face Detection using Haar Cascade
-  Face Recognition using OpenCV LBPH
-  Attendance tracking saved as CSV
-  Student data management
-  Image-based training system

## Technologies Used

- Python 3
- OpenCV
- Tkinter (GUI)
- NumPy
- CSV (for data logging)
- 
## 📦 Folder Structure

attendance_system/
│
├── Attendance/              # Stores daily attendance CSVs
├── StudentDetails/          # Stores student data
├── TrainingImageLabel/      # Stores trained model file
├── haarcascade_frontalface_default.xml  # Face detection model
├── attendance.py            # Main app script
├── cwd.py                   # Additional logic
└── README.md                # Project documentation

## 🧑‍💻 How to Run

1. Clone the repo:

   git clone https://github.com/Akhila-anand/attendance.git
   cd attendance
  
2. Install dependencies:

   pip install opencv-python numpy
   
3. Run the project:

   python attendance.py
   

