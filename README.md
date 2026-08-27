# Face-recognition-system
An award-winning facial recognition access control system built with Python and SQL. Developed as a collaborative Erasmus+ project, it won the university's best course project award.

The system simulates a strict security checkpoint: to enter a "restricted laboratory," a user must enter a password and pass a real-time face verification check via a webcam.

**Note: For a detailed overview, system architecture, workflow, and screenshots, please check out the presentation PDF in the `docs/` folder!**

## Built with
* **Python** (core logic)
* **OpenCV** (Haar Cascade for face detection & LBPH for recognition)
* **NumPy** & **Pygame** (for data handling and audio feedback)
* **SQL** (database management)

##  Quick Start
If you want to test the system locally, make sure you have the requirements installed, and use these main commands:

**1. Capture facial data (create your profile):**
```bash
python main.py capture-name "Captain Demming" --user-id 1-samples 30
