#An ML based project on Diabetes Diagnosis System
#libraries and the modules 

import streamlit as st #pip install streamlit this line import streamlit library with the alias st and also streamlit is used for webapp framwork developement.
import pandas as pd #pip install pandas pandas is an powerfull library data analysis and manipulation in the python
from PIL import Image # it imports image class pillow imaging library it is used for image processing
#here this pil library there is no functionality but it can be used if the data given has any image or graphical data

from sklearn.metrics import accuracy_score   #pip install scikiet-learn from sklearn.metric module we import the function accuracy_score
# this is used to calculate the accuracy of the any machine learning model 

from sklearn.ensemble import RandomForestClassifier # from the sklearn.ensemble module we import the randomforestclassifier class
# this  class is used  for making randomforest machine learning model

from sklearn.model_selection import train_test_split # from sklearn.model_selection module we import the function train_test_split
# this function splits the data into the training and Testion Sets for model training and evaluation

import streamlit.components.v1 as com #this import component module from the streamlit library 
#here component module is used to embed the html,css,javascript functionality into your webapp

from streamlit_option_menu import option_menu #pip install streamlit_option_menu this imports the option menu function from thne streamlit_option_menu module
# this function allows you to add functionality like option menu to our web page created with the stream lit

from streamlit_extras.let_it_rain import rain #pip install streamlit_extras this imports the rain function from the custom let it run module 
# as the name suggest it is used in creation of custom effect like some thing falling like rain 




selected = option_menu( # here we are using the option menu function from stream lit library 
menu_title=None, #here the menu title is set to none 
options=['Home','Report','Contact Us','Feedback'], #This parameter defines the options available in the menu
icons=["house","book","envelope",'journals'], # This represent the icon associated with the option in the name 
default_index=0, # Here this does the work of selecting the Home option as default
orientation="horizontal", # orientation of the option menu 
styles={
    
    "icon":{"color":"orange","font-size":"25px"}, # size and the colour of the  options 
    "nav-link-selected":{"background-color":"black"}, # background colour when it is selected
    "container":{"background-color":"#C0D6E4"},
  }
)


# data Handelling and file reading

df = pd.read_csv('diabetes.csv')# this line here reads the csv file and stores in the df variable where the data is stored in data frame
# here data frame is an type of data structure that store data in 2d array much like excell 
# here The r before the string is used to specify a raw string

if selected == "Home":
 st.sidebar.markdown("> # Welcome to InsuCheck") #this creates an sidebar heading
 styles={
     "div.st-emotion-cache-1v0mbdj.e115fcil1":{"margin-left":"40px"}
     }


 img=Image.open("Mask.png")
 st.sidebar.image(img, width=180)
 

 com.iframe("https://lottie.host/embed/94f573fd-263b-45bd-8eb5-b15ee86ffd9d/AolhlFd6vj.json")
 # This adds an gif animated to the webapp
 original_title = '<p style="font-family:Courier; color:#819090 ; font-size: 40px; font-weight: bold; letter-spacing:  10px; ">Diabetes Checkup</p>'
 # Here this is for styling our heading 
 st.markdown(original_title, unsafe_allow_html=True) # This renders the string text writtent in the original_title 



 txt =st.text_input( # This part of the code is creating the functionality to add the name of the user 
       
        label="Enter your name:", 
        max_chars = 20 , # maximum number of the text 
        placeholder = "Your Name",
 )
 st.write("Hi,",txt) # it is used to get the output of the input you have 

 st.subheader('Training Data')
 st.write(df.describe())
 
 st.subheader('Visualisation') #This line creates a subheader for the visualization section.
 st.bar_chart(df) #This line creates a bar chart of the entire dataset using the st.bar_chart function.
	#  it is informative for the small data set but it might not be the most suitable visualization choice for larger datasets.


	# Dividing the data with independent variable and the result
 x = df.drop(['Outcome'], axis = 1) #This line creates a new DataFrame x by dropping the last column ("Outcome") from df.
	#This effectively separates the features (independent variables) from the target variable (dependent variable).

 y = df.iloc[:, -1] #The .iloc method in pandas is used for integer-location based indexing.
	#It allows you to select data by its integer location, i.e., by row and column numbers.


	# Data manipulation for making it usable for training and testing data into four different set 
 x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=0)
	#here we are using 20% of the data for the testing and rest 80% for the training of the data

 basic = '<p style="font-family:Courier; color:Black; font-size:19px; font-weight: none; letter-spacing: 0px; ">Empowering Health Through Predictive Insights</p>'
 
 
 
 def user_report():
    """_summary_This line defines a function named user_report(). 
    This function will be responsible for collecting user input 
    and creating a DataFrame containing the user's data.

    Returns:
        _type_: pandas DataFrame.
    """
    
    age = st.sidebar.slider('Age', 21,88,33)
    pregnancies = st.sidebar.slider('Pregnancies',0,17,3)
    glucose = st.sidebar.slider('Glucose', 0,200,120)
    bp = st.sidebar.slider('Blood Pressure', 0,122,70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0,100,20)
    insulin = st.sidebar.slider('Insulin',0,846,79)
    bmi = st.sidebar.slider('BMI', 0,67,20)
    dpf = st.sidebar.slider('Diabetes Pedigree Functuion',0.0,2.4,0.47)
    
    user_report = { # Here this is making an dictionaries of the user report from the data taken from the sliding bar 
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
        
        
        
        
        
        
        
    }
    
    report_data = pd.DataFrame(user_report, index=[0])
    #This line creates a single-row Pandas DataFrame named report_data from the user_report dictionary.
    #The index argument is set to [0] to create a single-row DataFrame suitable for model prediction.
    return report_data
   
   
 user_data = user_report() #here the user data is assigned the dataframe of the user and can be used to test 

 rf = RandomForestClassifier()
 # This line creates a Random Forest Classifier object named rf using the RandomForestClassifier class from sklearn.ensemble

 rf.fit(x_train, y_train)
 # This line trains the Random Forest Classifier model rf using the training data (x_train for features and y_train for target labels).

 st.subheader('Accuracy: ')
 st.write(str(accuracy_score(y_test,rf.predict(x_test))*100)+'%')
 # This line creates a subheader for displaying the model's accuracy.

 user_result = rf.predict(user_data)
 # here finally user data goes and the predictor ml model predicts the result which is either 0 or 1



         
if selected =="Report":
     
 #here this menu bar has all the functionality of the homepage or the main file here we have to wrote again 
 #so when we select the report this programs are running and show the report of the user 
    
 img=Image.open("Mask.png")
 st.sidebar.image(img, width = 200)
 predict = '<p style="font-family:Courier; color:Black; font-size: 25px; font-weight: bold; letter-spacing: 0px; ">Empowering Health Through Predictive Insights</p>'
 st.sidebar.markdown(predict, unsafe_allow_html=True)
 
 x = df.drop(['Outcome'], axis = 1)
 y = df.iloc[:, -1]

 x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state=0)


 def user_report():
    
    age = st.sidebar.slider('Age', 21,88,33)
    pregnancies = st.sidebar.slider('Pregnancies',0,17,3)
    glucose = st.sidebar.slider('Glucose', 0,200,120)
    bp = st.sidebar.slider('Blood Pressure', 0,122,70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0,100,20)
    insulin = st.sidebar.slider('Insulin',0,846,79)
    bmi = st.sidebar.slider('BMI', 0,67,20)
    dpf = st.sidebar.slider('Diabetes Pedigree Functuion',0.0,2.4,0.47)
    
    user_report = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
        
        
        
        
        
        
        
    }
    
    report_data = pd.DataFrame(user_report, index=[0])
    return report_data
 user_data = user_report()

 rf = RandomForestClassifier()
 # Here the user data is assigned the dataframe of the user and can be used to test
 rf.fit(x_train, y_train)
 # This line trains the Random Forest Classifier model rf using the training data
 # (x_train for features and y_train for target labels).


# Result prediction of the user data

 user_result = rf.predict(user_data)
 st.write('> # Your Report: ') # We are writing this line on the web app with sr.write function 
 
 output=' ' # Here this is like the flag variable which is set to empty and we will change it and display it as per the result
 
 if user_result[0]==0: # Here the prediction model will give either 1 or 0 depending upon the data provided
  
        output = 'You are healthy.' # If the models predicts output as 0 that means that you are not diabetic
        st.write(output, " \n Glad to assist you.") # This is the ending message of the report menu option 
        rain( # we have imported the rain function from the streamlit library we are using it to show the congratulatory message
           emoji="🎉", #here this emoji is showering with the rain effect
           
           font_size=40,
           falling_speed=3,
           animation_length="infinite", # The time till this effect will run is set to infininty
    )
 else:
         output = 'You are mostly diabitic.' # This is simply the message if your data says you are diabetic
         st.write(output)

if selected == "Contact Us":
  # here this is the menu for getting the user to contact us and we have ceated and interface for it.
  # here he will give the data for us to contact him/her
   txt =st.text_input(
       
        label="Enter your name:",
        max_chars = 20 ,
        placeholder = "Your Name",
       )
   txt1 =st.text_input(
       
        label="Enter your email id:",
        max_chars = 20 ,
        placeholder = "Your Email",
       )
   txt2 =st.text_area(
       
        label="Enter your message:",
        max_chars = 20 ,
        placeholder = "Your message",
       )
   with st.form("form"):
       
     s_state = st.form_submit_button("Submit")
     if s_state:
         if txt == "" and txt1 == "":
             st.warning("Please fill above fields")
         else:
             st.success("Submitted Successfully")

if selected == "Feedback":
 # """ Lastly we have integrated google formlink to get the user data and for any software the userfeedback is necessary.
 #It provide us with necessary criticism  which is helpfull in further developement of the product"""

  mark= ' <marquee  bgcolor="" behavior="scroll" scrollamount="6" ><FONT size=6 COLOR="black">Not to  be shy, spill the tea (feedback) on Diabetes Diagnosis.We will love to hear it by you!!!</FONT></marquee> '
  st.sidebar.markdown(mark, unsafe_allow_html=True)
  img=Image.open("Mask.png")
  st.sidebar.image(img, width = 200)
  predict = '<p style="font-family:Courier; color:Black; font-size: 25px; font-weight: bold; letter-spacing: 0px; ">Empowering Health Through Predictive Insights</p>'
  st.sidebar.markdown(predict, unsafe_allow_html=True)

  com.iframe("https://docs.google.com/forms/d/e/1FAIpQLSfh8U9SCfCic6GNVy1ILnteq0JaVjtGNga6cHRdKRXBlNhDdQ/viewform?embedded=true" ,width=670, height=999)
