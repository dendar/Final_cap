#!/usr/bin/env python
# coding: utf-8

# https://pythonforundergradengineers.com/streamlit-app-with-bokeh.html

#link
#https://www.analyticsvidhya.com/blog/2020/12/streamlit-web-api-for-nlp-tweet-sentiment-analysis/
#https://www.youtube.com/watch?v=SIu2VL-RAXc&list=PLJ39kWiJXSixyRMcn3lrbv8xI8ZZoYNZU&index=6
# https://www.youtube.com/watch?v=bEOiYF1a6Ak&list=PLJ39kWiJXSixyRMcn3lrbv8xI8ZZoYNZU&index=9



import streamlit as st
import altair as alt 
from joblib import load

#####
from sklearn.model_selection import train_test_split, GridSearchCV
# preprocesing 
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.utils import class_weight

from sklearn import metrics
from sklearn.pipeline import Pipeline
#from imblearn.over_sampling import SMOTE

from sklearn.dummy import DummyClassifier

from sklearn.metrics import classification_report, plot_roc_curve, plot_confusion_matrix, roc_curve
from yellowbrick.classifier import ROCAUC
######

from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import TfidfVectorizer

#load modal
model_age = load("lr_age_model")
model_status = load("nb_status_model")
vectorizer = CountVectorizer()


# load EDA pkgs
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
sns.set_style('whitegrid')
import numpy as np

#Wordcloud
from wordcloud import WordCloud
from PIL import Image

# frequent
import nltk
from nltk.corpus import stopwords
import string
from nltk import FreqDist

# display data
df = pd.read_csv("data/clean_data.csv.gz")
#df = pd.read_csv("eda_data.csv")
df = df.sample(frac=0.2)
df = df.dropna()

#numeric_col = df.select_dtypes(['int32', 'int64', 'float32', 'float64']).columns.tolist()



#method 1
#st.dataframe(df)
#import re
# function
@st.cache 
#age
def predict_text_age(text):
    results = model_age.predict([text])
    return results

def prob_text_age(text):
    results = model_age.predict_proba([text])
    return results
#marital status
def predict_text_status(text):
    results = model_status.predict([text])
    return results

def prob_text_status(text):
    results = model_status.predict_proba([text])
    return results

# wordcloud
def plot_wordcloud(corpus, max_words=150, max_font_size=35):
            wordcloud = WordCloud(collocations=True,
                                  background_color='black', 
                                  max_words=150,
                                  max_font_size=35, 
                                  )
            wordcloud.generate(str(corpus))
            fig, ax = plt.subplots(figsize=(10,10))
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(wordcloud, cmap=None)
            #plt.imshow(wordcloud, interpolation = "bilinear")
            st.pyplot(fig)
            
def frequent(text, number = 30, figsize=(10,7)):
    tokens = nltk.tokenize.word_tokenize(','.join(map(str, text)))
    freq = FreqDist(tokens)
    #display(freq.most_common(number))
    most_common = pd.DataFrame(freq.most_common(number),
                           columns=['word','count']).sort_values('count',
                                                                 ascending=True)
    #plot
    fig, ax = plt.subplots(figsize=figsize)
    most_common.set_index('word').tail(25).plot(kind='barh',ax=ax)
    ax.set(xlabel=None, ylabel=None, title="Most frequent words")
    ax.grid(axis="x")
    st.pyplot(fig)
    
    


def main():
    
    st.title("Text Analysis")
    menu = ["Home", "EDA", "Classifier"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Home
    if choice == "Home":
        #st.subheader("Home-Text")
        #st.success("Hello")
        
        # search to use later
        #search = st.text_input("Search")
        #with st.beta_expander("results"):
            #retrieved_df = df[df["clean"].str.contains(search)]
            #st.dataframe(retrieved_df[["Agegroup", "status", "clean"]])
                               
                               
        
        with st.form(key = "form"):
            raw_text = st.text_area("write...")
            column1, column2 = st.beta_columns(2)
            with column1:
                #st.success("Text")
                #st.write(raw_text)
                submit = st.form_submit_button(label = "Age")
            with column2:
                submit1 = st.form_submit_button(label = "Status")
            
            
        if submit:
            column1, column2 = st.beta_columns(2)
            
            prediction = predict_text_age(raw_text)
            prob = prob_text_age(raw_text)
            
            with column1:
                #st.success("Text")
                #st.write(raw_text)
               
                st.success("Predict")
                result = ['Under 35' if prediction == 0 else 'Over 35' for prediction in prediction]
                st.write(result[0])
                st.write("Confidence: {}".format(np.max(prob)))
                
            with column2:
                st.success("Prediction Prob")
                #st.write(prob)
                
                prob_df = pd.DataFrame(prob, columns = ["Under 35", "Over 35"])
                #st.write(prob.T)
                df_clean = prob_df.T.reset_index()
                df_clean.columns = ["Age", "Probability"]
                
                fig = alt.Chart(df_clean).mark_bar().encode(x = "Age", y = "Probability", color = "Age")
                st.altair_chart(fig, use_container_width = True)
                
              
                
              
            
        if submit1:
            
            #selectbox_1 = st.checkbox("Age")
            column1, column2 = st.beta_columns(2)
            # apply function
            prediction = predict_text_status(raw_text)
            prob = prob_text_status(raw_text)

            with column1:
                st.success("Text")
                st.write(raw_text)

                st.success("Predict")
                result = ['single' if prediction == 0 else 'married' for prediction in prediction]
                st.write(result[0])
                st.write("Confidence: {}".format(np.max(prob)))


            with column2:
                st.success("Prediction Prob")
                #st.write(prob)
                
                prob_df = pd.DataFrame(prob, columns = ["Single", "Married"])
                df_clean = prob_df.T.reset_index()
                df_clean.columns = ["Status", "Probability"]
                
                fig = alt.Chart(df_clean).mark_bar().encode(x = "Status", y = "Probability", color = "Status")
                st.altair_chart(fig, use_container_width = True)

        st.success("WordCloud And Frequent")
        
        
        
        
        

        #st.sidebar.subheader("Create plot")
        
        
        
        # add select widget 
        data = df[["Agegroup", "status"]]
        selectbox_1 = st.sidebar.selectbox(label = "Age & Marital Status", options = data.columns)
        
        #Age
        if selectbox_1 == "Agegroup":
        
                # wordcloud 
            column1, column2 = st.beta_columns(2) 
            with column1:

                with st.beta_expander("WordCloud"):
                    for i in df["Agegroup"].unique():
                        st.write("******** {} ********".format(i))
                        plot_wordcloud(corpus=df[df["Agegroup"]==i]["clean"],
                                       max_words=150, max_font_size=35)

            with column2:        
                with st.beta_expander("Frequent"):
                    for i in df["Agegroup"].unique():
                        st.write("********* {}:  ************".format(i))
                        frequent(df[df["Agegroup"]==i]["clean"],  20)
                     
        #status                
        elif selectbox_1 == "status":
            
            column1, column2 = st.beta_columns(2) 
            with column1:

                with st.beta_expander("WordCloud"):
                    for i in df["status"].unique():
                        st.write("******** {} ********".format(i))
                        plot_wordcloud(corpus=df[df["status"]==i]["clean"],
                                       max_words=150, max_font_size=35)

            with column2:        
                with st.beta_expander("Frequent"):
                    for i in df["status"].unique():
                        st.write("********* {}:  ************".format(i))
                        frequent(df[df["status"]==i]["clean"],  20)
        st.write("select agegroup or marital status to see \n worldCould and frequent word")

    
    
    # EDA
    
    elif choice == "EDA":
        st.subheader("Exploratory Data Analysis")
        
        
        
         #search to use later
        search = st.text_input("Search")
        with st.beta_expander("results"):
            retrieved_df = df[df["clean"].str.contains(search)]
            #search_df = dataframe(retrieved_df) #[["Agegroup", "status", "clean"]])
            
            #st.sidebar.subheader("plot for specific word you search")
        
            # add select widget
            selectbox_search = st.sidebar.selectbox(label = "plot for specific word you search", options = retrieved_df.columns)
            fig, ax = plt.subplots()
            g = sns.countplot(x = retrieved_df[selectbox_search], hue = "Agegroup", data= retrieved_df)
            g.set_xticklabels(ax.get_xticklabels(),rotation = 45, fontsize = 12,  ha="right")
            st.pyplot(fig)
        
       
        
        # create plot
        st.sidebar.subheader("Dataset Plot")
        
        # add select widget
        selectbox_1 = st.sidebar.selectbox(label = "Feature", options = df.columns)
        fig, ax = plt.subplots()
        g = sns.countplot(x = df[selectbox_1], hue = "Agegroup", data= df)
        g.set_xticklabels(ax.get_xticklabels(),rotation = 45, fontsize = 12,  ha="right")
        st.pyplot(fig)
       
        # create hist
        #fig, ax = plt.subplots()
        #st.sidebar.subheader("hist")
        #selectbox_2 = st.sidebar.selectbox(label = "Y axis", options = df.columns)
        #selectbox_3 = st.sidebar.selectbox(label = "X axis", options = df.columns)
        #hist_slider = st.sidebar.slider(label ="Number of bins", min_value = 2, max_value =100, value = 20)
        
        #sns.distplot(df[selectbox_3], hist_slider)
        #st.pyplot(fig)
        
    # About  
    else:
        st.subheader("Classifier")
        st.write("""
        # Explore different classifier
        """)
        stopwords_list = stopwords.words("english")
        stopwords_list += list(string.punctuation)
        
        # age 
        age_dict = {"18-35":0, "35+":1}
        df["Agegroup"] = df["Agegroup"].map(age_dict)
        df["Agegroup"].value_counts()
        
        y = df["Agegroup"]
        X = df["clean"]
        
        # train split

        X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=42)

        
        #vectorizer
        vectorizer =  TfidfVectorizer(stop_words = stopwords_list,
                                      encoding='utf-8',decode_error='ignore',
                                      max_features = 30000)
        
        X_train_tf = vectorizer.fit_transform(X_train) 
        X_test_tf = vectorizer.transform(X_test)
        
        # class weight
        class_weights = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)

        weights_dict = dict(zip(np.unique(y_train), class_weights))
        weights_dict

        sample_weights = class_weight.compute_sample_weight(weights_dict, y_train)
        
        
        # model evaluation
        def model_cl(model,X_train, X_test):   
            st.write("***********************************************")
            st.write(f"Training score:  {round(model.score(X_train, y_train), 2)}")
            st.write(f"Test score:  {round(model.score(X_test, y_test), 2)}")
                #Test score:  {round(model.score(X_test, y_test),2)}")
            st.write("\n")
            #st.write("********************Cl REPORT****************")

            #y_pred = model.predict(X_test)
            #y_prob = model.predict_proba(X_test)
            #st.write(metrics.classification_report(y_test, y_pred))

            st.write("\n")
            st.write("***********************************************")

            fig, ax = plt.subplots(ncols=2,figsize=(12,5))
            plot_confusion_matrix(model,X_test,
                                  y_test, cmap='Blues',
                                  xticks_rotation='vertical',
                                  normalize='true',
                                  display_labels=["18-35", "35+"],
                                        ax= ax[0])
            curve  = ROCAUC(model,encoder={0:"18-35", 
                                           1:"35+"})

            curve .fit(X_train, y_train)        
            curve .score(X_test, y_test)        
            curve .show()                
            st.pyplot(fig)


        
        
        
        
        
        # create sidebar for classifier
        classifier_name = st.sidebar.selectbox("Select Classifier", ("NB", "LR"))
        
        #classifier
        def get_classifier(classifier_name):
            if classifier_name=="NB":
                nb= MultinomialNB()
                nb.fit(X_train_tf, y_train, sample_weight=sample_weights)
                model_cl(nb, X_train_tf, X_test_tf)
            elif classifier_name=="LR":
                lr= LogisticRegression()
                lr.fit(X_train_tf, y_train, sample_weight=sample_weights)
                model_cl(lr, X_train_tf, X_test_tf)
                
        
        get_classifier(classifier_name)

        
        
        
        
        
        
        
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    

