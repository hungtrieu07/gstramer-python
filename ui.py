from tkinter import *
from tkinter import messagebox
import cv2
import numpy as np
from math import sqrt
import sys

first_frames = []

class FrameProcess:
    @staticmethod
    def read_video(path):
        videos = []
        for p in path:
            cap = cv2.VideoCapture(p)
            videos.append(cap)
        return videos 
    
    @staticmethod
    def nearest_square_number(n):
        return int(sqrt(n))+1

    @staticmethod
    def square_number(n):
        sqr = sqrt(n)
        if (sqr * sqr) == n:
            return True
        else:
            return False
        
    @staticmethod
    def process_window(image_list):
        image_length = len(image_list)
        if FrameProcess.square_number(image_length):
            stream_tile_number = int(sqrt(image_length))
            new_width = int(1920 / stream_tile_number)
            new_height = int(1080 / stream_tile_number)
            resized_image = [cv2.resize(image, (new_width, new_height)) for image in image_list]
            output_image = cv2.vconcat([cv2.hconcat(resized_image[i:i+stream_tile_number]) for i in range(0, image_length, stream_tile_number)])
        else:
            stream_tile_number = FrameProcess.nearest_square_number(image_length)
            expected_image_length = stream_tile_number**2
            blank_tile_number = expected_image_length - image_length
            new_width = int(1920 / stream_tile_number)
            new_height = int(1080 / stream_tile_number)
            blank_image = np.zeros((new_height, new_width, 3), np.uint8)
            image_list.extend([blank_image] * blank_tile_number)
            resized_image = [cv2.resize(image, (new_width, new_height)) for image in image_list]
            output_image = cv2.vconcat([cv2.hconcat(resized_image[i:i+stream_tile_number]) for i in range(0, expected_image_length, stream_tile_number)])
        return output_image
    
    @staticmethod
    def choose_coord(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            coor.append((x, y))
           
    @staticmethod
    def crop_window(image, coords):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pts = np.array([coords], np.int32)
        cv2.fillPoly(mask, pts, (255, 255, 255))
        crop = cv2.bitwise_and(image, image, mask=mask)
        rect = cv2.boundingRect(pts)
        crop = crop[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        return crop

class App(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent, background="white")
        self.parent = parent
        self.links = []
        self.initUI()
        
    def initUI(self):
        self.parent.title("Window")
        self.pack(fill=BOTH, expand=1)
        
        AddContent = Label(self, text="Link")
        AddContent.pack(side=LEFT)
        AddContent.place(x=10, y=10)
        
        self.link_entry = Entry(self, bd=2)
        self.link_entry.pack(side=LEFT)
        self.link_entry.place(x=50, y=10)
        
        AddButton = Button(self, text="Add")
        AddButton.place(x=200, y=5)
        AddButton.bind("<Button-1>", self.add)
        
        SubmitButton = Button(self, text="Submit")
        SubmitButton.place(x=300, y=5)
        SubmitButton.bind("<Button-1>", lambda event: self.submit_link())
        
    def add(self, event):
        link = self.link_entry.get().split(",")
        print("Link added: ", link)
        self.links.extend(link)
        self.link_entry.delete(0, END)
    
    def submit_link(self):
        global coor
        coor = []
        links = self.links
        
        for link in links:
            if(link == ''):
                print("Link thu %d bi rong".format(link))
                sys.exit(1)
            print("Submitted links: ", links)
            
        video_frames = FrameProcess.read_video(links)
        cropped_frames = []
        
        for i, cap in enumerate(video_frames):
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read first frame from video %d" % i)
                continue
            cv2.imshow("Video %d" % i, frame)
            cv2.setMouseCallback('Video %d' % i, FrameProcess.choose_coord)
        
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyWindow('Video %d' % i)
            
            cropped_frame = FrameProcess.crop_window(frame, coor)
            cropped_frames.append(cropped_frame)
            coor = []
        
        for cap in video_frames:
            cap.release()
        cv2.destroyAllWindows()
    
        for i, cropped_frame in enumerate(cropped_frames):
            window_name = 'Cropped frame %d' % i
            cv2.imshow(window_name, cropped_frame)
            cv2.waitKey()
            cv2.destroyWindow(window_name)
    
        self.link_entry.delete(0, END)


root = Tk()
root.geometry("1920x1080")
app = App(root)
root.mainloop()