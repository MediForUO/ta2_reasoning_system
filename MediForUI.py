import sys, os
#from PyQt5 import QtCore, QtGui, QtWidgets, QtGui
from PyQt5.QtCore import *
from PyQt5.QtGui import * #import QIcon, QPixmap
from PyQt5.QtWidgets import *
from PIL import Image
from PIL.ImageQt import ImageQt

#from subprocess import call
import pandas as pd
import generate_Medifor_bn_model_10 as reasoning_system

import io

import numpy as np
from scipy import ndimage
from scipy.misc import imread, imsave

from medifor import fileutil, processing
from medifor.resources import Resource


class mainClass(QWidget):
    def __init__(self, parent = None):
        super().__init__()
        
        self.mydict = { }
        #Train the initial model
        #self.myArray = generate_Medifor_bn_model_10.train_model()
        
        #Initilize all file paths as NA (nothing selected)
        self.fname1 = self.fname2 = self.fname3 = self.fname4 = self.fname5 = self.fname6 = self.fname7 = self.fname8 = self.fname9 = self.fname10 =str("")
        self.fname0="nan"
        #This is the layout
        layout = QVBoxLayout(self)
        #Add the two layouts for the scroll boxes
        imageLayout = QVBoxLayout()
        scoreLayout = QVBoxLayout()
        heatMapResultsLayout = QVBoxLayout()
        
        everythingLayout = QVBoxLayout()
        
		
        #Intro message
        self.instructionText = QLabel("Step 1: Select the heat maps from the file system, leave blank if no values\nStep 2: Once the heat maps are loaded, select the TA1 algorithm scores\nStep 3: Run inference \nStep 4: Edit the results in the text box, then save the file")
        layout.addWidget(self.instructionText)
      
        self.selectOriginalImageButton = QPushButton("Select Original Image")
        #self.selectOriginalImageButton.setMaximumWidth(200)
        self.selectOriginalImageButton.clicked.connect(self.getOriginalImage)
        layout.addWidget(self.selectOriginalImageButton)
        self.imageDisplay0 = QLabel("No Image Loaded")
        layout.addWidget(self.imageDisplay0)


        #Header for heap maps output
        self.headerHeatMapsResults = QLabel("\nHeat Maps are posted below")
        self.headerHeatMapsResults.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.headerHeatMapsResults)
        
        #Select Image Button / call for all heat maps
        #Also all initial heat map set upts
        self.displayHeatMapResult = QPushButton("Display Results")
        #self.displayHeatMapResult.setMaximumWidth(200)
        #self.displayHeatMapResult.setAlignment(Qt.AlignCenter)
        self.displayHeatMapResult.clicked.connect(self.displayHeatMapResults)
        
        self.headerHeatMapsResults1 = QLabel("\ncopyclone confidence:")
        self.headerHeatMapsResults1.setAlignment(Qt.AlignLeft)
        heatMapResultsLayout.addWidget(self.headerHeatMapsResults1)
        
        self.heatmapDisaply1 = QLabel("No Image copyclone")
        heatMapResultsLayout.addWidget(self.heatmapDisaply1)
        
        self.headerHeatMapsResults2 = QLabel("\nremoval confidence:")
        self.headerHeatMapsResults2.setAlignment(Qt.AlignLeft)
        heatMapResultsLayout.addWidget(self.headerHeatMapsResults2)
        
        self.heatmapDisaply2 = QLabel("No Image removal")
        heatMapResultsLayout.addWidget(self.heatmapDisaply2)
        
        self.headerHeatMapsResults3 = QLabel("\nHeat splice confidence:")
        self.headerHeatMapsResults3.setAlignment(Qt.AlignLeft)
        heatMapResultsLayout.addWidget(self.headerHeatMapsResults3)
        
        self.heatmapDisaply3 = QLabel("No Image splice")
        heatMapResultsLayout.addWidget(self.heatmapDisaply3)
        
        
        #Header for heap maps input
        self.headerHeatMaps = QLabel("\nSelect the heat map(s) in the directory")
        self.headerHeatMaps.setAlignment(Qt.AlignCenter)
        imageLayout.addWidget(self.headerHeatMaps)
          
        #Select Image Button / call for all heat maps
        #Also all initial heat map set upts
        self.selectImageButton = QPushButton("Select block01")
        self.selectImageButton.setMaximumWidth(200)
        self.selectImageButton.clicked.connect(self.getHeatMap1)
        imageLayout.addWidget(self.selectImageButton)
        self.imageDisplay1 = QLabel("No Image Loaded")
        imageLayout.addWidget(self.imageDisplay1)
        
        self.selectImageButton2 = QPushButton("Select block02")
        self.selectImageButton2.setMaximumWidth(200)
        self.selectImageButton2.clicked.connect(self.getHeatMap2)
        imageLayout.addWidget(self.selectImageButton2)
        self.imageDisplay2 = QLabel("No Image Loaded")
        imageLayout.addWidget(self.imageDisplay2)
  
        
        self.selectImageButton3 = QPushButton("Select copymove01")
        self.selectImageButton3.setMaximumWidth(200)
        self.selectImageButton3.clicked.connect(self.getHeatMap3)
        imageLayout.addWidget(self.selectImageButton3)
        self.imageDisplay3 = QLabel("No Image Loaded")
        imageLayout.addWidget(self.imageDisplay3)

        self.selectImageButton4 = QPushButton("Select dct01")
        self.selectImageButton4.setMaximumWidth(200)
        self.selectImageButton4.clicked.connect(self.getHeatMap4)
        imageLayout.addWidget(self.selectImageButton4)
        self.imageDisplay4 = QLabel("No Image Loaded")
        imageLayout.addWidget(self.imageDisplay4)


        self.selectImageButton5 = QPushButton("Select dc02")
        self.selectImageButton5.setMaximumWidth(200)
        self.selectImageButton5.clicked.connect(self.getHeatMap5)
        imageLayout.addWidget(self.selectImageButton5)
        self.imageDisplay5 = QLabel("No Image Loaded")
        imageLayout.addWidget(self.imageDisplay5)

        self.selectImageButton6 = QPushButton("Select dct03_A")
        self.selectImageButton6.setMaximumWidth(200)
        self.selectImageButton6.clicked.connect(self.getHeatMap6)
        imageLayout.addWidget(self.selectImageButton6)
        self.imageDisplay6 = QLabel("No Image Loaded")
        imageLayout.addWidget(self.imageDisplay6)

        self.selectImageButton7 = QPushButton("Select dct03_NA")
        self.selectImageButton7.setMaximumWidth(200)
        self.selectImageButton7.clicked.connect(self.getHeatMap7)
        imageLayout.addWidget(self.selectImageButton7)
        self.imageDisplay7 = QLabel("No Image Loaded")
        imageLayout.addWidget(self.imageDisplay7)

        self.selectImageButton8 = QPushButton("Select ela01")
        self.selectImageButton8.setMaximumWidth(200)
        self.selectImageButton8.clicked.connect(self.getHeatMap8)
        imageLayout.addWidget(self.selectImageButton8)
        self.imageDisplay8 = QLabel("No Image Loaded")
        imageLayout.addWidget(self.imageDisplay8)

        self.selectImageButton9 = QPushButton("Select noise01")
        self.selectImageButton9.setMaximumWidth(200)
        self.selectImageButton9.clicked.connect(self.getHeatMap9)
        imageLayout.addWidget(self.selectImageButton9)
        self.imageDisplay9 = QLabel("No Image Loaded")
        imageLayout.addWidget(self.imageDisplay9)

        self.selectImageButton10 = QPushButton("Select noise02")
        self.selectImageButton10.setMaximumWidth(200)
        self.selectImageButton10.clicked.connect(self.getHeatMap10)
        imageLayout.addWidget(self.selectImageButton10)
        self.imageDisplay10 = QLabel("No Image Loaded")
        imageLayout.addWidget(self.imageDisplay10)
        
        #End of heat map set ups



        #set up for all the score text fields and names
        self.newline1 = QLabel("\nEnter the float for block01")
        scoreLayout.addWidget(self.newline1)
        
        self.text1 = QLineEdit()
        self.text1.setMaximumWidth(100)
        self.text1.setText("nan")
        scoreLayout.addWidget(self.text1)
        
        
        self.newline2 = QLabel("\nEnter the float for block02")
        scoreLayout.addWidget(self.newline2)
        
        self.text2 = QLineEdit()
        self.text2.setMaximumWidth(100)
        self.text2.setText("nan")
        scoreLayout.addWidget(self.text2)
        

        self.newline3 = QLabel("\nEnter the float for copymove01")
        scoreLayout.addWidget(self.newline3)

        self.text3 = QLineEdit()
        self.text3.setMaximumWidth(100)
        self.text3.setText("nan")
        scoreLayout.addWidget(self.text3)


        self.newline4 = QLabel("\nEnter the float for dct01")
        scoreLayout.addWidget(self.newline4)

        
        self.text4 = QLineEdit()
        self.text4.setMaximumWidth(100)
        self.text4.setText("nan")
        scoreLayout.addWidget(self.text4)

        
        self.newline5 = QLabel("\nEnter the float for dct02")
        scoreLayout.addWidget(self.newline5)
        
        self.text5 = QLineEdit()
        self.text5.setMaximumWidth(100)
        self.text5.setText("nan")
        scoreLayout.addWidget(self.text5)
        
        
        self.newline6 = QLabel("\nEnter the float for dct03_A")
        scoreLayout.addWidget(self.newline6)
        
        self.text6 = QLineEdit()
        self.text6.setMaximumWidth(100)
        self.text6.setText("nan")
        scoreLayout.addWidget(self.text6)


        self.newline7 = QLabel("\nEnter the float for dct03_NA")
        scoreLayout.addWidget(self.newline7)

        self.text7 = QLineEdit()
        self.text7.setMaximumWidth(100)
        self.text7.setText("nan")
        scoreLayout.addWidget(self.text7)
        

        self.newline8 = QLabel("\nEnter the float for ela01")
        scoreLayout.addWidget(self.newline8)

        self.text8 = QLineEdit()
        self.text8.setMaximumWidth(100)
        self.text8.setText("nan")
        scoreLayout.addWidget(self.text8)
        

        self.newline9 = QLabel("\nEnter the float for noise01")
        scoreLayout.addWidget(self.newline9)

        self.text9 = QLineEdit()
        self.text9.setMaximumWidth(100)
        self.text9.setText("nan")
        scoreLayout.addWidget(self.text9)


        self.newline10 = QLabel("\nEnter the float for noise02")
        scoreLayout.addWidget(self.newline10)

        self.text10 = QLineEdit()
        self.text10.setMaximumWidth(100)
        self.text10.setText("nan")
        scoreLayout.addWidget(self.text10)
        
        #End scores set ups
        
        #Make group box, which is allowed to be scrolled

        mygroupbox =QGroupBox()
        mygroupbox.setLayout(imageLayout)
        scroll = QScrollArea()
        scroll.setWidget(mygroupbox)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(300)
        layout.addWidget(scroll)



        #Header for scores
        self.headerScores = QLabel("\nEnter a float value for each score below, or NAN for unknown")
        self.headerScores.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.headerScores)
        
        
        #Make second group box, which is allowed to be scrolled
        mygroupbox2 = QGroupBox()
        mygroupbox2.setLayout(scoreLayout)
        scroll2 = QScrollArea()
        scroll2.setWidget(mygroupbox2)
        scroll2.setWidgetResizable(True)
        scroll2.setFixedHeight(300)
        layout.addWidget(scroll2)
        
        
        
        
        self.runInferenceButton = QPushButton(" Inference")
        
        #Call inference
		#Later on it will change to pass a list of algorith(s) to run
        self.runInferenceButton.clicked.connect(self.inference)
        layout.addWidget(self.runInferenceButton)
        
        self.loading = QLabel("Inference is ready to run")
        layout.addWidget(self.loading)
      		
        #Save Results Button - to save results to text file from text box
        self.saveResultsButton = QPushButton("Save Confidence Scores")
        self.saveResultsButton.clicked.connect(self.saveResults)
		
        #Disaply Results in text box
        self.displayResultsButton= QPushButton("Display Results")
        #Maybe auto load files later on from inference?
        self.displayResultsButton.clicked.connect(self.displayResults)
        
        #Add results button after text box
        layout.addWidget(self.displayResultsButton)


        #Display Results in a text Field
        self.resultsTextEditBox = QTextEdit()
        layout.addWidget(self.resultsTextEditBox)
        
        #Add save results button after text box
        layout.addWidget(self.saveResultsButton)
        
        ##Add results button
        layout.addWidget(self.displayHeatMapResult)
        
        
        mygroupbox3 = QGroupBox()
        mygroupbox3.setLayout(heatMapResultsLayout)
        scroll3 = QScrollArea()
        scroll3.setWidget(mygroupbox3)
        scroll3.setWidgetResizable(True)
        scroll3.setFixedHeight(300)
        layout.addWidget(scroll3)
        #Inference Button / call
        
      
      


		
        
        #Add layout and widgets to window , set window size, name window
        self.setLayout(layout)
        
        mygroupbox4 = QGroupBox()
        mygroupbox4.setLayout(layout)
        scroll4 = QScrollArea()
        scroll4.setWidget(mygroupbox4)
        scroll4.setWidgetResizable(True)
        #scroll.setFixedHeight(800)
        everythingLayout.addWidget(scroll4)
        #Inference Button / call
        
        
        self.setLayout(everythingLayout)
        #Width,Height
        self.setMinimumSize(650, 700)
        self.setWindowTitle("MediFor reasoning")
    
    def inference(self):
    
        #Set status
        self.loading.setText("Calculating........")
        
        #train_model()
        '''
        NIST_rows = self.myArray[0]
        NIST_cols = self.myArray[1]
        NIST_df = self.myArray[2].copy()
        NIST_regional_manips = self.myArray[3][:]
        NIST_global_manips = self.myArray[4][:]
        NIST_manip_schema_dict = self.myArray[5].copy()
        NIST_TA1_algorithms = self.myArray[6][:]
        NIST_base_bn_lines = self.myArray[7][:]
        node_dict = self.myArray[8].copy()
        '''
        #query_dict = self.myArray[9].copy()
        
        query_dict = {}
        
        query_dict['block01'] = {'heatmap': str(self.fname1), 'score': str(self.text1.text())}
        query_dict['block02'] = {'heatmap': str(self.fname2), 'score': str(self.text2.text())}
        query_dict['copymove01'] = {'heatmap': str(self.fname3), 'score': str(self.text3.text())}
        query_dict['dct01'] = {'heatmap': str(self.fname4), 'score': str(self.text4.text())}
        query_dict['dct02'] = {'heatmap': str(self.fname5), 'score': str(self.text5.text())}
        query_dict['dct03_A'] = {'heatmap': str(self.fname6), 'score': str(self.text6.text())}
        query_dict['dct01_NA'] = {'heatmap': str(self.fname7), 'score': str(self.text7.text())}
        query_dict['ela01'] = {'heatmap': str(self.fname8), 'score': str(self.text8.text())}
        query_dict['noise01'] = {'heatmap': str(self.fname9), 'score': str(self.text9.text())}
        query_dict['noise02'] = {'heatmap': str(self.fname10), 'score': str(self.text10.text())}
        
        '''
        query_dict['block01']['heatmap'] = str(self.fname1)
        query_dict['block02']['heatmap'] = str(self.fname2)
        query_dict['copymove01']['heatmap'] = str(self.fname3)
        query_dict['dct01']['heatmap'] = str(self.fname4)
        query_dict['dct02']['heatmap'] = str(self.fname5)
        query_dict['dct03_A']['heatmap'] = str(self.fname6)
        query_dict['dct03_NA']['heatmap'] = str(self.fname7)
        query_dict['ela01']['heatmap'] = str(self.fname8)
        query_dict['noise01']['heatmap'] = str(self.fname9)
        query_dict['noise02']['heatmap'] = str(self.fname10)
        
       
        query_dict['block01']['score'] = str(self.text1.text())
        query_dict['block02']['score'] = str(self.text2.text())
        query_dict['copymove01']['score'] = str(self.text3.text())
        query_dict['dct01']['score'] = str(self.text4.text())
        query_dict['dct02']['score'] = str(self.text5.text())
        query_dict['dct03_A']['score'] = str(self.text6.text())
        query_dict['dct03_NA']['score'] = str(self.text7.text())
        query_dict['ela01']['score'] = str(self.text8.text())
        query_dict['noise01']['score'] = str(self.text9.text())
        query_dict['noise02']['score'] = str(self.text10.text())
        '''
        #Input for inference
        input = {}

        for algorithm in query_dict.keys():
            flag = False
            if((query_dict[algorithm]['heatmap']) != ""):
                heatmap_object = imread(query_dict[algorithm]['heatmap'])
                heatmap_resource = Resource('image',heatmap_object,'image/png')
                flag = True
            print (query_dict[algorithm]['score'])
            input[algorithm] = {}
            if(flag):
                input[algorithm]['heatmap'] = heatmap_resource
            input['score'] = query_dict[algorithm]['score']
                #input[algorithm] = {'score': query_dict[algorithm]['score'], 'heatmap': heatmap_resource}
        
        #Path to original image will be user selected main image
        if(self.fname0 == "nan"):
            self.loading.setText("Must Select original Image !!!!")
            print(self.fname0)
        else:
            image_object = imread(self.fname0)
            input['image'] = Resource('image', image_object, 'image/jpeg')
            input['algorithms'] = ['block01', 'block02', 'copymove01', 'dct01', 'dct02', 'dct03_A', 'dct03_NA', 'ela01', 'noise01', 'noise02']
            #print query_dict
            #Try to call inference
            try:
                self.loading.setText("Calculating........")
                self.mydict = reasoning_system.run_inference(input)
            #Else catch error
            except ValueError:
                self.loading.setText("Failed due to score value, please check values.")
                #except:
                #self.loading.setText("Failed Inference, please check inputs, ensure proper heat maps and proper score values.")
            self.loading.setText("Done")
            print (self.mydict)


    def saveResults(self):
        #Open the desktop, let the user select file path to save file
        fileName, _ = QFileDialog.getSaveFileName(self, 'save your results','inference_results',  'Text files (*.txt);;All files (*.*)', 'Text files (*.txt)')
        
        #Ensure has txt extention -- new safefilename so .txt option provided
        #fileName =  fileName+ ".txt"
        
        #Error Catching
        if(fileName=="" or fileName ==None):
            return None
        
        #Create file, or open file
        output_file = open(fileName, "w")
        
        #Text box contents moved to string
        tempText = self.resultsTextEditBox.toPlainText()
        
        #Save string into file
        output_file.write(tempText)
        output_file.close()
    
    
    def displayResults(self):
        #MediFor_inference_output.txt is where the calculations are saved, we open them to view/edit them
        #results_file = open("MediFor_inference_output.txt", "r")
        #data=results_file.read()
        #print (data)
        #Set the text box to display results
        self.resultsTextEditBox.setText("") #Clear previous results
        for key in self.mydict.keys():
            self.resultsTextEditBox.setText( key + " " + str(self.mydict[key].confidence)+ "\n" + self.resultsTextEditBox.toPlainText() )

    def dialog(self):
        #Open file system and get file path and file name
        temp, _ = QFileDialog.getOpenFileName(self, 'Open file', 'cos.path.dirname(__file__)',"Image files (*.jpg *.gif *png)")
        return temp


    #Heat Map read me#
    #
    # Each heat map has its own function. This way class variables can be accessed
    #
    #
    
    def displayHeatMapResults(self):
        #print(self.mydict)
        try:
            imageq = ImageQt(self.mydict['copyclone'].heatmap.data) #convert PIL image to a PIL.ImageQt object
            qimage = QImage(imageq)
            pm=(QPixmap(qimage).scaledToWidth(200))
            self.heatmapDisaply1.setPixmap(pm)
        except:
            self.heatmapDisaply1.setText("Copyclone results not detected")

        try:
            imageq2 = ImageQt(self.mydict['removal'].heatmap.data) #convert PIL image to a PIL.ImageQt object
            qimage2 = QImage(imageq)
            pm2=(QPixmap(qimage2).scaledToWidth(200))
            self.heatmapDisaply2.setPixmap(pm2)
        except:
            self.heatmapDisaply2.setText("Removal results not detected")
        
        try:
            imageq3 = ImageQt(self.mydict['splice'].heatmap.data) #convert PIL image to a PIL.ImageQt object
            qimage3 = QImage(imageq3)
            pm3=(QPixmap(qimage3).scaledToWidth(200))
            self.heatmapDisaply3.setPixmap(pm3)
        except:
            self.heatmapDisaply3.setText("Splice results not detected")
        
        #print(self.mydict)
        #self.mydict['copyclone'].heatmap.data.show()
        #self.mydict['removal'].heatmap.data.show()
        #self.mydict['splice'].heatmap.data.show()
        #Lighting is global, will be fixed later
        #self.mydict['lighting'].heatmap.data.show()
        #print(self.mydict['copyclone'].heatmap.data)
        #print(self.mydict['removal'].heatmap.data)
        #print(self.mydict['splice'].heatmap.data)
        #Lighting is global, will be fixed later
        #print(self.mydict['lighting'].heatmap.data)
        '''
        {'removal': <generate_Medifor_bn_model_10.Manipulation object at 0x112f71ef0>, 
        'lighting': <generate_Medifor_bn_model_10.Manipulation object at 0x112f71e10>, 
        'splice': <generate_Medifor_bn_model_10.Manipulation object at 0x112f71e80>, 
        'copyclone': <generate_Medifor_bn_model_10.Manipulation object at 0x112f71f60>}

        '''
   
    ###
    ##
    ###
    ##
    ###
    ##
    ###
    ##
    ###
    ##
    ###
    ##
    
    
    def getOriginalImage(self):
        #Open file system to select image to processes
        self.fname0 = self.dialog()
        if(self.fname0!=""):
            self.imageDisplay0.setPixmap(QPixmap(self.fname0).scaledToWidth(200))
        else:
            self.fname0 = "nan"
            self.imageDisplay0.setText("No Image Loaded")
    
    def getHeatMap1(self):
		#Open file system to select image to processes
        self.fname1 = self.dialog()
        if(self.fname1!=""):
            self.imageDisplay1.setPixmap(QPixmap(self.fname1).scaledToWidth(200))
        else:
            self.fname1 = ""
            self.imageDisplay1.setText("No Image Loaded")

    def getHeatMap2(self):
        #Open file system to select image to processes
        self.fname2 = self.dialog()
        print(self.fname2)
        if(self.fname2!=""):
            self.imageDisplay2.setPixmap(QPixmap(self.fname2).scaledToWidth(200))
        else:
            self.fname2 = ""
            self.imageDisplay2.setText("No Image Loaded")

    def getHeatMap3(self):
        #Open file system to select image to processes
        self.fname3 = self.dialog()
        if(self.fname3!=""):
            self.imageDisplay3.setPixmap(QPixmap(self.fname3).scaledToWidth(200))
        else:
            self.fname3 = ""
            self.imageDisplay3.setText("No Image Loaded")

    def getHeatMap4(self):
        #Open file system to select image to processes
        self.fname4 = self.dialog()
        if(self.fname4!=""):
            self.imageDisplay4.setPixmap(QPixmap(self.fname4).scaledToWidth(200))
        else:
            self.fname4 = ""
            self.imageDisplay4.setText("No Image Loaded")

    
    def getHeatMap5(self):
        #Open file system to select image to processes
        self.fname5 = self.dialog()
        if(self.fname5!=""):
            self.imageDisplay5.setPixmap(QPixmap(self.fname5).scaledToWidth(200))
        else:
            self.fname5 = ""
            self.imageDisplay5.setText("No Image Loaded")
    
    def getHeatMap6(self):
        #Open file system to select image to processes
        self.fname6 = self.dialog()
        if(self.fname6!=""):
            self.imageDisplay6.setPixmap(QPixmap(self.fname6).scaledToWidth(200))
        else:
            self.fname6 = ""
            self.imageDisplay6.setText("No Image Loaded")
    
    def getHeatMap7(self):
        #Open file system to select image to processes
        self.fname7 = self.dialog()
        if(self.fname7!=""):
            self.imageDisplay7.setPixmap(QPixmap(self.fname7).scaledToWidth(200))
        else:
            self.fname7 = ""
            self.imageDisplay7.setText("No Image Loaded")
            
    
    def getHeatMap8(self):
        #Open file system to select image to processes
        self.fname8 = self.dialog()
        if(self.fname8!=""):
            self.imageDisplay8.setPixmap(QPixmap(self.fname8).scaledToWidth(200))
        else:
            self.fname8 = ""
            self.imageDisplay8.setText("No Image Loaded")
    
    def getHeatMap9(self):
        #Open file system to select image to processes
        self.fname9 = self.dialog()
        if(self.fname9!=""):
            self.imageDisplay9.setPixmap(QPixmap(self.fname9).scaledToWidth(200))
        else:
            self.fname9 = ""
            self.imageDisplay9.setText("No Image Loaded")
    
    def getHeatMap10(self):
        #Open file system to select image to processes
        self.fname10 = self.dialog()
        if(self.fname10!=""):
            self.imageDisplay10.setPixmap(QPixmap(self.fname10).scaledToWidth(200))
        else:
            self.fname10 = ""
            self.imageDisplay10.setText("No Image Loaded")

    
		
def main():
    print("test")
    app = QApplication(sys.argv)
    ex = mainClass()
    ex.show()
    sys.exit(app.exec_())
	
if __name__ == '__main__':
    main()
