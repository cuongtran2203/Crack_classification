from tkinter import *
import cv2 
import time
import imageio
from core_ai import *
from PIL import Image, ImageTk
import customtkinter
from customtkinter import *
count=0
count2=0

def update_frame():
    global canvas1, photo,count,count2
    # Doc tu camera
    ret, frame = cap.read()
    frame,count23,count34=circles_detection(frame,video_path)
    count+=count23
    count2+=count34
    label_count1.configure(text=str(count))
    label_count2.configure(text=str(count2))
    # Ressize
    frame = cv2.resize(frame, (640,480))
    
    # Chuyen he mau
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert hanh image TK
    photo =ImageTk.PhotoImage(Image.fromarray(frame))
    # Show
    canvas1.create_image(0,0, image = photo, anchor=NW)

    app.after(20, update_frame)
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")
app=customtkinter.CTk()
app.geometry("800x480")
def restart_counter():
    global count,count2
    count=0
    count2=0
    label_count1.configure(text=str(count))
    label_count2.configure(text=str(count2))
def start():
    pass
def stop():
    pass
if __name__ == '__main__': 

    #Video Streaming
    video_path="video/video_5.mp4"
    cap=cv2.VideoCapture(video_path)
    
    canvas1=CTkCanvas(app,width=640,height=480)
    canvas1.place(x=0,y=0)

    canvas2=CTkCanvas(app,width=160,height=480)
    canvas2.place(x=640,y=0)
    

    label_ok=CTkLabel(canvas2,text="OK",font =('Verdana', 30),text_color="#3ADF00")
    label_ok.place(x=20,y=10)
    label_count1=CTkLabel(canvas2,text="0",font =('Verdana', 30,"bold"),text_color="#3ADF00")
    label_count1.place(x=90,y=10)
    label_NG=CTkLabel(canvas2,text="NG",font =('Verdana', 30),text_color="#D35B58")
    label_NG.place(x=20,y=60)
    label_count2=CTkLabel(canvas2,text="0",font=('Verdana', 30,"bold"),text_color="#D35B58")
    label_count2.place(x=90,y=60)
    # create Rectangle for canvas
    canvas2.create_rectangle(5,5,150,100)
    # photo=CTkImage(Image.open("button2_resize.jpg"),size=(50,50))
    button_start=CTkButton(canvas2,text="START",font =('Verdana', 20),command=start)
    button_start.place(x=10,y=150)
    button_stop=CTkButton(canvas2,text="STOP",command=stop,font =('Verdana', 20))
    button_stop.place(x=10,y=220)
    button_restart=CTkButton(canvas2,text="RESET",command=restart_counter,font =('Verdana', 20))
    button_restart.place(x=10,y=290)
    update_frame()









    app.mainloop()
