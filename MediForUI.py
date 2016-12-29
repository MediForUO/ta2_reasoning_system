import sys
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from subprocess import call
import pandas as pd
import generate_Medifor_bn_model_7

class mainClass(QWidget):
    def __init__(self, parent = None):
        super(mainClass, self).__init__(parent)

        #Train the initial model
        self.myArray = generate_Medifor_bn_model_7.train_model()
        
        #Initilize all file paths as NA (nothing selected)
        self.fname1 = str("")
        self.fname2 = str("")
        self.fname3 = str("")
        self.fname4 = str("")
        self.fname5 = str("")
        self.fname6 = str("")
        self.fname7 = str("")
        self.fname8 = str("")
        self.fname9 = str("")
        self.fname10 =str("")

        #This is the layout
        layout = QVBoxLayout(self)
        
        #Add the two layouts for the scroll boxes
        imageLayout = QVBoxLayout()
        scoreLayout = QVBoxLayout()
                                                                
		
        #Intro message
        self.instructionText = QLabel("Step 1: Select the heat maps from the file system, leave blank if no values\nStep 2: Once the heat maps are loaded, select the TA1 algorithm scores\nStep 3: Run inference \nStep 4: Edit the results in the text box, then save the file")
        layout.addWidget(self.instructionText)
      
      
        #Header for heap maps
        self.headerHeatMaps = QLabel("\nSelect the heat map in the directory")
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

        #Header for scores
        self.headerScores = QLabel("\nEnter a float value for each score below, or NAN for unknown")
        self.headerScores.setAlignment(Qt.AlignCenter)
        scoreLayout.addWidget(self.headerScores)

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
        #scroll.setFixedHeight(700)
        layout.addWidget(scroll)

        #Make second group box, which is allowed to be scrolled
        mygroupbox2 = QGroupBox()
        mygroupbox2.setLayout(scoreLayout)
        scroll = QScrollArea()
        scroll.setWidget(mygroupbox2)
        scroll.setWidgetResizable(True)
        #scroll.setFixedHeight(700)
        layout.addWidget(scroll)
        
        
      	#Inference Button / call
        self.runInferenceButton = QPushButton(" Inference")
        
        #Call inference
		#Later on it will change to pass a list of algorith(s) to run
        self.runInferenceButton.clicked.connect(self.inference)
        layout.addWidget(self.runInferenceButton)
        
        self.loading = QLabel("Inference is ready to run")
        layout.addWidget(self.loading)
      		
        #Save Results Button - to save results to text file from text box
        self.saveResultsButton = QPushButton("Save Results")
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
		
        
        #Add layout and widgets to window , set window size, name window
        self.setLayout(layout)
        #Width,Height
        self.setMinimumSize(650, 700)
        self.setWindowTitle("MediFor reasoning")
    
    def inference(self):
    
        #Set status
        self.loading.setText("Calculating........")
        
        #train_model()
        NIST_rows = self.myArray[0]
        NIST_cols = self.myArray[1]
        NIST_df = self.myArray[2].copy()
        NIST_regional_manips = self.myArray[3][:]
        NIST_global_manips = self.myArray[4][:]
        NIST_manip_schema_dict = self.myArray[5].copy()
        NIST_TA1_algorithms = self.myArray[6][:]
        NIST_base_bn_lines = self.myArray[7][:]
        node_dict = self.myArray[8].copy()
        query_dict = self.myArray[9].copy()
        
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

        #print query_dict
        
        #Try to call inference
        try:
            self.loading.setText("Calculating........")
            generate_Medifor_bn_model_7.run_inference(NIST_rows,NIST_cols , NIST_df,NIST_regional_manips, NIST_global_manips, NIST_manip_schema_dict, NIST_TA1_algorithms, NIST_base_bn_lines,node_dict,query_dict)
            self.loading.setText("Done")
        #Else catch error
        except ValueError:
            self.loading.setText("Failed due to score value, please check values.")
            #except:
        #self.loading.setText("Failed Inference, please check inputs, ensure proper heat maps and proper score values.")


    def saveResults(self):
        #Open the desktop, let the user select file path to save file
        fileName = QFileDialog.getSaveFileName(self, 'save your results','/Desktop/results',  selectedFilter='*.txt')
        #Ensure has txt extention
        fileName =  fileName+ ".txt"
        
        #Create file, or open file
        output_file = open(fileName, "w")
        #Text box contents moved to string
        tempText = self.resultsTextEditBox.toPlainText()
        #Save string into file
        output_file.write(tempText)
        output_file.close()
    
    def displayResults(self):
        #MediFor_inference_output.txt is where the calculations are saved, we open them to view/edit them
        results_file = open("MediFor_inference_output.txt", "r")
        data=results_file.read()
        
        #Set the text box to display results
        self.resultsTextEditBox.setText(data)

    def dialog(self):
        #Open file system and get file path and file name
        temp = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Image files (*.jpg *.gif *png)")
        return temp


    #Heat Map read me#
    #
    # Each heat map has its own function. This way class variables can be accessed
    #
    #
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
    app = QApplication(sys.argv)
    ex = mainClass()
    ex.show()
    sys.exit(app.exec_())
	
if __name__ == '__main__':
    main()
