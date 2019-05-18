import numpy as np
import cv2
from matplotlib import pyplot as plt

class Seminar():

    def __init__(self):
        # init
        print("running sem")    

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,800)
        self.cap.set(4,800)
        ret, self.frame = self.cap.read()
        
        r,h,c,w = 250,125,400,125  # dimenzije
        self.track_window = (c,r,w,h)
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

    def adjust_gamma(self, workframe, gamma=0.4):
      
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")	  
        frame_gamma = cv2.LUT(workframe, table)
        return frame_gamma
    
    def histogram(self, workFrame):
        
        hist = cv2.calcHist([workFrame],[0],None,[256],[0,256])
        cdf = hist.cumsum();
        cdf_norm = cdf * hist.max()/cdf.max()

        #plt.figure(0)
        #plt.clf()
        #plt.subplot(121)
        #plt.plot(cdf_norm, color = 'b')
        #plt.plot(hist)
        #plt.show(block = False)
        #plt.title('Original')

        
        cdf_m = np.ma.masked_equal(cdf,0)
        cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
        cdf = np.ma.filled(cdf_m,0).astype('uint8')
        frame_eq = cdf[workFrame]
        hist_eq = cv2.calcHist([frame_eq],[0],None,[256],[0,256])
        
        #plt.subplot(122)
        #plt.plot(cdf_m, color = 'b')
        #plt.plot(hist_eq)
        #plt.show(block = False)
        #plt.title('Equalised')

        plt.pause(0.001)

        return frame_eq;

    def thresholding(self, workframe):
        ret, gray_tresh = cv2.threshold(workframe,0,15,cv2.THRESH_BINARY_INV) 
        return gray_tresh

    def filter(self, workframe):
        gray_filt = cv2.GaussianBlur(workframe,(5,5),0)
        return gray_filt

    def meanshift(self, workframe):       
        ret, self.track_window = cv2.meanShift(workframe, self.track_window, self.term_crit)
        
    def main(self):
        print("running cam")
        while(True):

  

            # Capture frame-by-frame
            ret, self.frame = self.cap.read()
            
            # Our operations on the frame come here
            self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            
            self.gray_gam = self.adjust_gamma(self.gray)
            self.gray_eq = self.histogram(self.gray_gam)
            self.gray_tresh = self.thresholding(self.gray_eq)
            self.gray_filt = self.filter(self.gray_tresh)
            # Display the resulting frame
            cv2.imshow('frame', self.gray)
            cv2.imshow('frame_eq', self.gray_eq)
            
            self.meanshift(self.gray_filt)

            # Draw it on image
            x,y,w,h = self.track_window
            img = cv2.rectangle(self.frame, (x,y), (x+w,y+h), 255,2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,str(x + w/2) + ' ' + str(y + h/2),(10,40), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(img,'+',(int(x+w/2-5),int(y+h/2+5)), font, 1,(255,255,255),2,cv2.LINE_AA)
            #img_cut = self.gray_eq[y:y+h, x:x+w]
            cv2.imshow('final',img)
            #cv2.imshow('frame_cut',img_cut)
            cv2.imshow('frame', self.gray)
            cv2.imshow('frame_eq', self.gray_gam)


            #plt.figure(1)
            #plt.clf()
            #plt.subplot(121)    
            #plt.imshow(self.gray_tresh,'gray')
            #plt.title('thresholded')
            #plt.show(block = False)
           
            #plt.subplot(122)      
            #plt.title('thresholded & filtered')
            #plt.imshow(self.gray_filt,'gray')
            #plt.show(block = False)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    sem = Seminar()
    sem.main()
    


