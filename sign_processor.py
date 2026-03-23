import cv2
import math
import numpy as np
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import enchant
import pyttsx3
import threading
import time

model = load_model("cnn8grps_rad1_model.h5")
ddd = enchant.Dict("en-US")
# Enable up to 2 hands for better detection and complex gestures
hd = HandDetector(detectionCon=0.1, maxHands=2)

class SignProcessor:
    def __init__(self):
        self.offset = 29
        self.str = " "
        self.word = " "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "
        self.current_symbol = ""
        self.prev_char = ""
        self.count = -1
        self.ten_prev_char = [" "] * 10
        self.pts = []
        self.model = model
        self.white_bg = np.ones((400, 400, 3), dtype=np.uint8)*255
        self.stable_char = ""
        self.stable_count = 0
        self.last_appended_char = ""
        self.no_hand_frames = 0
        self.custom_gestures = {} # {word: [pts1, pts2, ...]}
        self.capture_mode = False
        self.capture_word = ""
        self.capture_frames = []
        self.capture_count = 0
        self.is_active = False
        self.last_commit_time = 0
        self.commit_interval = 3.0 # seconds
        self.current_candidate = ""
        self.tally = {}



    def start_capture(self, word):
        self.capture_mode = True
        self.capture_word = word
        self.capture_frames = []
        self.capture_count = 0

    def cancel_capture(self):
        self.capture_mode = False
        self.capture_word = ""
        self.capture_frames = []
        self.capture_count = 0

    def add_custom_word(self, new_word):
        ddd.add(new_word)

    def speak_text(self):
        def _speak(text):
            engine = pyttsx3.init()
            engine.setProperty("rate", 100)
            engine.say(text)
            engine.runAndWait()
        threading.Thread(target=_speak, args=(self.str,)).start()

    def clear(self):
        self.str = " "
        self.word1 = self.word2 = self.word3 = self.word4 = " "
        self.word = " "

    def backspace(self):
        if len(self.str) > 1:
            self.str = self.str[:-1]
        elif len(self.str) == 1:
            self.str = " "

    def space(self):
        if self.str[-1] != " ":
            self.str += " "

    def toggle_active(self, val):
        self.is_active = val
        if val:
            self.last_commit_time = time.time()
            self.tally = {}
            self.current_candidate = ""

    def process_frame(self, frame):
        # Mirrored view
        cv2image = cv2.flip(frame, 1)
        
        # Panel 1: Original
        p1 = cv2image.copy()
        
        # Panel 2: AR (Original + Skeleton)
        p2 = cv2image.copy()
        hands, p2 = hd.findHands(p2, draw=True, flipType=False)
        
        # Panel 3: White (White background + Skeleton)
        p3 = self.white_bg.copy()
        
        if hands:
            self.no_hand_frames = 0
            
            # Combine skeletons for the White View (centered group)
            min_x = min(h['bbox'][0] for h in hands)
            min_y = min(h['bbox'][1] for h in hands)
            max_x = max(h['bbox'][0] + h['bbox'][2] for h in hands)
            max_y = max(h['bbox'][1] + h['bbox'][3] for h in hands)
            
            w_total = max(1, max_x - min_x)
            h_total = max(1, max_y - min_y)
            os_x = (400 - w_total) // 2
            os_y = (400 - h_total) // 2
            
            for hand in hands:
                h_pts = hand['lmList']
                p_centered = []
                for pt in h_pts:
                    px = (pt[0] - min_x) + os_x
                    py = (pt[1] - min_y) + os_y
                    p_centered.append((px, py))

                # Draw skeleton lines
                for t in range(0, 4, 1): cv2.line(p3, p_centered[t], p_centered[t+1], (0, 255, 0), 3)
                for t in range(5, 8, 1): cv2.line(p3, p_centered[t], p_centered[t+1], (0, 255, 0), 3)
                for t in range(9, 12, 1): cv2.line(p3, p_centered[t], p_centered[t+1], (0, 255, 0), 3)
                for t in range(13, 16, 1): cv2.line(p3, p_centered[t], p_centered[t+1], (0, 255, 0), 3)
                for t in range(17, 20, 1): cv2.line(p3, p_centered[t], p_centered[t+1], (0, 255, 0), 3)
                cv2.line(p3, p_centered[5], p_centered[9], (0, 255, 0), 3)
                cv2.line(p3, p_centered[9], p_centered[13], (0, 255, 0), 3)
                cv2.line(p3, p_centered[13], p_centered[17], (0, 255, 0), 3)
                cv2.line(p3, p_centered[0], p_centered[5], (0, 255, 0), 3)
                cv2.line(p3, p_centered[0], p_centered[17], (0, 255, 0), 3)
                for pt in p_centered: cv2.circle(p3, pt, 2, (0, 0, 255), 1)

            # --- ALWAYS PREDICT FOR INSTANT FEEDBACK ---
            self.pts = hands[0]['lmList'] # set current primary points
            self.predict(p3, w_total, h_total)
            
            if self.capture_mode:
                if self.capture_count < 5:
                    # Capture and normalize landmarks for the first hand detected
                    h0 = hands[0]
                    h_pts = h0['lmList']
                    bx, by, bw, bh = h0['bbox']
                    
                    # Normalize relative to wrist and bounding box size
                    hx, hy = h_pts[0][0], h_pts[0][1]
                    norm_pts = [((p[0]-hx)/max(1,bw), (p[1]-hy)/max(1,bh)) for p in h_pts]
                    
                    self.capture_frames.append(norm_pts)
                    self.capture_count += 1
                    
                    if self.capture_count == 5:
                        if self.capture_word not in self.custom_gestures:
                            self.custom_gestures[self.capture_word] = []
                        self.custom_gestures[self.capture_word].extend(self.capture_frames)
                        self.capture_mode = False
                        print(f"TRAINING COMPLETE for: {self.capture_word}")
                
                # Visual counting on the AR panel
                cv2.putText(p2, f"Capturing: {self.capture_count}/5", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
        else:
            self.no_hand_frames += 1
            if self.no_hand_frames > 30 and len(self.str) > 0 and self.str[-1] != " ":
                self.str += " "
        
        # Consistent Resizing (All panels to 400x400 for combined list)
        p1_res = cv2.resize(p1, (400, 400))
        p2_res = cv2.resize(p2, (400, 400))
        
        # UI overlays
        cv2.putText(p1_res, "CAMERA", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(p2_res, "AR MODE", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(p3, "WHITE VIEW", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Draw instant prediction (small letter) on panels
        if self.current_candidate:
            char_disp = self.current_candidate.lower()
            # Draw on AR View
            cv2.putText(p2_res, char_disp, (350, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            # Draw on White View
            cv2.putText(p3, char_disp, (350, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        combined = np.hstack((p1_res, p2_res, p3))
        return combined

    def distance(self,x,y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    def action1(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word1.upper()


    def action2(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str=self.str[:idx_word]
        self.str=self.str+self.word2.upper()
        #self.str[idx_word:last_idx] = self.word2


    def action3(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word3.upper()



    def action4(self):
        idx_space = self.str.rfind(" ")
        idx_word = self.str.find(self.word, idx_space)
        last_idx = len(self.str)
        self.str = self.str[:idx_word]
        self.str = self.str + self.word4.upper()


    def speak_fun(self):
        self.speak_engine.say(self.str)
        self.speak_engine.runAndWait()


    def clear_fun(self):
        self.str=" "
        self.word1 = " "
        self.word2 = " "
        self.word3 = " "
        self.word4 = " "

    def predict(self, test_image, w, h):
        white=test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 0
                # print("00000")

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0
                print("++++++++++++++++++")
                # print("00000")

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][
                0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2
                # print("22222")

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2
                # print("22222")


        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]

        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][
                0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3
                print("33333c")



        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3
                print("33333b")

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3
                print("33333a")

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4
                # print("44444")

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 4
                # print("44444")

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4
                # print("44444")

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4
                # print("44444")

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4
                # print("44444")

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5
                print("55555b")

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][
                1] + 17 > self.pts[20][1]:
                ch1 = 5
                print("55555a")

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5
                # print("55555")

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5
                # print("55555")

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7
                # print("77777")

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7
                # print("77777")

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7
                # print("77777")

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6
                print("666661")


        # condition for [yj][x]
        print("2222  ch1=+++++++++++++++++", ch1, ",", ch2)
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6
                print("666662")

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6
                print("666663")

        # con for [l][x]

        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6
                print("666664")

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6
                print("666665")

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 1
                print("111111")

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1
                print("111112")

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and
                    self.pts[18][1] > self.pts[20][1]):
                ch1 = 1
                print("111112")

        # con for [d][pqz]
        fg = 19
        # print("_________________ch1=",ch1," ch2=",ch2)
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1
                print("111113")

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 1
                print("1111993")

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1
                print("1111mmm3")

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1
                print("1111140")

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] > self.pts[20][1])):
                ch1 = 1
                print("111114")

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
             self.pts[18][1] > self.pts[20][1])):
                ch1 = 7
                print("111114lll;;p")

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and
                 self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1
                print("111115")

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and
                    self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][
                0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1
                print("111116")

        # con for [w]

        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1
                print("1117")

        # -------------------------condn for 8 groups  ends

        # -------------------------condn for subgroups  starts
        #
        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][
                0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][
                1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][
                1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'

            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] <
                    self.pts[20][1]):
                ch1 = 'R'


        # Custom Gesture Matching
        if len(self.custom_gestures) > 0:
            hx, hy = self.pts[0][0], self.pts[0][1]
            norm_pts = [((p[0]-hx)/w, (p[1]-hy)/h) for p in self.pts]
            
            min_dist = float('inf')
            best_match = None
            for word, frames in self.custom_gestures.items():
                for f_pts in frames:
                    dist = sum(math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) for a, b in zip(norm_pts, f_pts))
                    if dist < min_dist:
                        min_dist = dist
                        best_match = word
            
            if min_dist < 2.8:
                ch1 = best_match
                
        # --- SENSING (Always happens for instant feedback) ---
        # Find current most sensed char from current frame's vote
        self.tally[ch1] = self.tally.get(ch1, 0) + 1
        leader = max(self.tally, key=self.tally.get)
        self.current_candidate = str(leader).lower()

        # Commitment logic (only happens if is_active is True)
        valid_commit = False
        if self.is_active and isinstance(ch1, str) and len(ch1.strip()) > 0:
            valid_commit = True
        
        if valid_commit:
            curr_time = time.time()
            elapsed = curr_time - self.last_commit_time
            
            if elapsed >= self.commit_interval:
                # 3-Second Window Complete
                print(f"VOTED COMMIT: {self.current_candidate}")
                self.str += self.current_candidate
                self.last_commit_time = curr_time
                self.tally = {} # Reset for next 3s window
                self.current_candidate = ""
            
            # UI display for progress
            progress = min(100, int((elapsed / self.commit_interval) * 100))
            self.current_symbol = f"{self.current_candidate} ({progress}%)"
        elif not self.is_active:
            # If start is OFF, we just show what we sense instantly
            self.current_symbol = f"Sensed: {self.current_candidate}"
            # Keep tally low so it reacts fast to hand changes while OFF
            if sum(self.tally.values()) > 5:
                self.tally = {ch1: 1} if ch1 else {}
        else:
            self.current_symbol = f"Adjusting... ({self.current_candidate})" if self.current_candidate else "Waiting..."
            if not ch1:
                self.stable_char = ""
                self.stable_count = 0

        # Pyenchant Suggestions
        if len(self.str.strip()) != 0:
            st = self.str.rfind(" ")
            word = self.str[st+1:]
            self.word = word
            if len(word.strip()) != 0:
                try:
                    ddd.check(word)
                    suggests = ddd.suggest(word)
                    lenn = len(suggests)
                    self.word4 = suggests[3] if lenn >= 4 else " "
                    self.word3 = suggests[2] if lenn >= 3 else " "
                    self.word2 = suggests[1] if lenn >= 2 else " "
                    self.word1 = suggests[0] if lenn >= 1 else " "
                except:
                    pass
            else:
                self.word1 = self.word2 = self.word3 = self.word4 = " "
