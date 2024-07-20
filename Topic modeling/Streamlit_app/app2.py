import streamlit as st
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import urllib.parse
import requests

#initial state
if 'state_p1' not in st.session_state:
    st.session_state.state_p1 = 0
if 'state_p2' not in st.session_state:
    st.session_state.state_p2 = 0
if 'age' not in st.session_state:
    st.session_state.age = 0
if 'weight' not in st.session_state:
    st.session_state.weight = 0
if 'height' not in st.session_state:
    st.session_state.height = 0
if 'gender' not in st.session_state:
    st.session_state.gender = 0
if 'food_allergy' not in st.session_state:
    st.session_state.food_allergy = 0
if 'drug_allergy' not in st.session_state:
    st.session_state.drug_allergy = 0
if 'congentital_disease' not in st.session_state:
    st.session_state.congentital_disease = 0
if 'queries' not in st.session_state:
    st.session_state.queries = None
if 'sbert_searched_df' not in st.session_state:
    st.session_state.sbert_searched_df = None
if 'queries_p2' not in st.session_state:
    st.session_state.queries_p2 = None
if 'sbert_searched_df_p2' not in st.session_state:
    st.session_state.sbert_searched_df_p2 = None
for i in range(10):
    if 'score_'+str(i+1) not in st.session_state:
        st.session_state['score_'+str(i+1)] = 'NA'
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

def set_state_p1(state):
    st.session_state.state_p1 = state

def set_state_p2(state):
    st.session_state.state_p2 = state

def split_text(text):
    return text.split(',')

#import data
sbert_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

data = pd.read_csv('articles_data.csv')

with open('corpus_embeddings.pickle', 'rb') as file:
    corpus_embeddings = pickle.load(file)

#local function
def personal_check(age,weight,height,gender):

    #age check
    if age >= 60:
        age = 'ผู้สูงอายุ'
    else:
        age = 'ทำงาน'

    #gender check
    if gender == 'หญิง':
        gender = 'ผู้หญิง'
    else:
        gender = 'ผู้ชาย'

    #bmi check
    height_meters = height / 100  

    bmi = weight / (height_meters ** 2)

    if bmi >= 30:
        bmi = 'อ้วนมาก'
    elif bmi >= 23 and bmi <30:
        bmi = 'อ้วน'
    elif bmi >= 18.5 and bmi <23:
        bmi = ''
    else:
        bmi = 'ผอม'
    
    return age,gender,bmi

def sbert_search(queries,data,embeddiing,sbert_model=sbert_model):

    index_lst = []
    score_lst = []

    query_embedding = sbert_model.encode(queries, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddiing, top_k=10)
    hits = hits[0]
    for hit in hits:
        index_lst.append(hit['corpus_id'])
        score_lst.append(hit['score'])

    sbert_searched = data.iloc[index_lst]
    sbert_searched['score'] = score_lst

    return sbert_searched

def page1_recommendation():
    #header
    st.markdown("<h1 style='text-align: center; color: black;'>---ระบบแนะนำบทความสุขภาพ---</h1>", unsafe_allow_html=True)

    with st.form('user_info'):

        #personal information input
        age = st.slider("อายุ", 10, 100, 25)

        col1, col2 = st.columns(2)
        with col1:
            weight = st.number_input("น้ำหนัก (Kg.): ",30.0,120.0,step=1.0,value=50.0)
        with col2:
            height = st.number_input("ส่วนสูง (cm.): ",100.0,250.0,step=1.0,value=150.0)
        
        col3, col4, col5 = st.columns(3)
        with col3:
            gender = st.selectbox('เพศ',('ชาย', 'หญิง'))
        with col4:
            food_allergy = st.selectbox('แพ้อาหาร?',('ไม่แพ้', 'แพ้อาหาร'))
        with col5:
            drug_allergy = st.selectbox('แพ้ยา?',('ไม่แพ้', 'แพ้ยา'))
        congentital_disease = st.text_input('โรคประจำตัวของคุณ (ถ้าหากไม่มี ไม่ต้องกรอก หรือใส่ "ไม่มี")')

        st.form_submit_button(on_click=set_state_p1,args=(1,))

    if st.session_state.state_p1 == 1:

        #asign state
        st.session_state.age = age
        st.session_state.weight = weight
        st.session_state.height = height
        st.session_state.gender = gender
        st.session_state.food_allergy = food_allergy
        st.session_state.drug_allergy = drug_allergy
        st.session_state.congentital_disease = congentital_disease

        #algorithm
        age,gender,bmi = personal_check(age,weight,height,gender)

        if food_allergy == 'ไม่แพ้':
            food_allergy = ''
        if drug_allergy == 'ไม่แพ้':
            drug_allergy = ''
        if congentital_disease == 'ไม่มี':
            congentital_disease = ''

        if congentital_disease == '':
            queries = gender+age+bmi+food_allergy+drug_allergy
        else:
            queries = congentital_disease
        
        #Bertopic search
        sbert_searched = sbert_search(queries,data,corpus_embeddings)

        st.session_state.sbert_searched_df = sbert_searched
        st.session_state.queries = queries
        st.session_state.state_p1 = 2

    if st.session_state.state_p1 == 2:

        with st.form('recommendations'):
            st.markdown("<h1 style='text-align: center; color: black;'>📰บทความสำหรับคุณ😆</h1>", unsafe_allow_html=True)
            st.write("---------------------------------------------------------------------------------------")

            for i in range(len(st.session_state.sbert_searched_df)):
                st.header(str(i+1)+'. '+st.session_state.sbert_searched_df.iloc[i]['title'])
                st.markdown('**Keywords :** '+ st.session_state.sbert_searched_df.iloc[i]['vote_keywords'])
                st.markdown(f"[Page source (Click here.)]({st.session_state.sbert_searched_df.iloc[i].url})")

                try:
                    banner_url = urllib.parse.quote(st.session_state.sbert_searched_df.iloc[i]['banner'], safe=':/')
                    response = requests.get(banner_url,timeout=5)
                    st.image(response.content)
                except:
                    st.image('https://icon-library.com/images/no-photo-icon/no-photo-icon-1.jpg')
                finally:
                    st.write("---------------------------------------------------------------------------------------")
            
            st.form_submit_button('Submit',on_click=set_state_p1,args=(0,))

def page2_search_engine():
    st.title("Search engine")

    with st.form('queries'):
        queries = st.text_input('คำหรือหัวข้อที่ต้องการค้นหา')
        st.form_submit_button(on_click=set_state_p2,args=(1,))
    
    if st.session_state.state_p2 == 1:
        sbert_searched = sbert_search(queries,data,corpus_embeddings)

        st.session_state.sbert_searched_df_p2 = sbert_searched
        st.session_state.queries_p2 = queries
        st.session_state.state_p2 = 2

    if st.session_state.state_p2 == 2:
        with st.form('recommendations'):
            st.markdown("<h1 style='text-align: center; color: black;'>📰บทความสำหรับคุณ😆</h1>", unsafe_allow_html=True)
            st.write("---------------------------------------------------------------------------------------")

            for i in range(len(st.session_state.sbert_searched_df_p2)):
                st.header(str(i+1)+'. '+st.session_state.sbert_searched_df_p2.iloc[i]['title'])
                st.markdown('**Keywords :** '+ st.session_state.sbert_searched_df_p2.iloc[i]['vote_keywords'])
                st.markdown(f"[Page source (Click here.)]({st.session_state.sbert_searched_df_p2.iloc[i].url})")

                try:
                    banner_url = urllib.parse.quote(st.session_state.sbert_searched_df_p2.iloc[i]['banner'], safe=':/')
                    response = requests.get(banner_url,timeout=5)
                    st.image(response.content)
                except:
                    st.image('https://icon-library.com/images/no-photo-icon/no-photo-icon-1.jpg')
                finally:
                    st.write("---------------------------------------------------------------------------------------")
            
            st.form_submit_button('Submit',on_click=set_state_p2,args=(0,))

#main
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page:", ("Recommendation System", "Search Engine"))
    
    if page == "Recommendation System":
        st.session_state.current_page = 1
    else:
        st.session_state.current_page = 2

    if page == "Recommendation System" and st.session_state.current_page == 1:
        page1_recommendation()
    elif page == "Search Engine" and st.session_state.current_page == 2:
        page2_search_engine()
    
if __name__ == "__main__":
    main()