import streamlit as st
import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from pymongo import MongoClient
import urllib.parse
import requests
from bertopic import BERTopic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import deepcut
import unicodedata
from pythainlp.util import normalize
import torch
import csv

#initial state
if 'state' not in st.session_state:
    st.session_state.state = 0
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
if 'optional_keyword' not in st.session_state:
    st.session_state.optional_keyword = 0
if 'all_recommend' not in st.session_state:
    st.session_state.all_recommend = None
if 'true_check' not in st.session_state:
    st.session_state.true_check = None
if 'queries' not in st.session_state:
    st.session_state.queries = None
if 'string_contain' not in st.session_state:
    st.session_state.string_contain = False
if 'sbert_searched_df' not in st.session_state:
    st.session_state.sbert_searched_df = None
if 'string_contain_df' not in st.session_state:
    st.session_state.string_contain_df = None
for i in range(10):
    if 'score_'+str(i+1) not in st.session_state:
        st.session_state['score_'+str(i+1)] = 'NA'

def set_state(state):
    st.session_state.state = state

def split_text(text):
    return text.split(',')

#import data
sbert_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

with open('corpus_embeddings.pickle', 'rb') as file:
    corpus_embeddings = pickle.load(file)
corpus_embeddings = pd.DataFrame(corpus_embeddings)

topic_model = BERTopic.load("Jiranuwat/topic_model",embedding_model=sbert_model)
data = pd.read_csv('articles_data.csv')
data['child_topic'] = topic_model.topics_[:]

with open('sensitive_words.txt', 'r',encoding='utf-8') as file:
    sensitive_words = file.read()
sensitive_words = sensitive_words.lower().replace('\n','').split(' ')
sensitive_words = list(set(sensitive_words))

#local function
def save_session_state_data(session_state_data, filename):
            with open(filename, 'a', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=session_state_data.keys())
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerow(session_state_data)

def deepcut_tokenizer(text,sensitive_words=sensitive_words):
    cleanedText = "".join([i for i in text if i not in string.punctuation]).lower()
    cleanedText = normalize(unicodedata.normalize('NFKD', cleanedText).replace('\n','').replace('\r','').replace('\t','').replace('“','').replace('”','').replace('.','').replace('–','').replace('‘','').replace('’','').replace('ํา','ำ').replace('...','').replace(',','').replace( 'ี','ี'))
    #cleanedText = re.sub(r'\d+', '', cleanedText)
    cleanedText = deepcut.tokenize(cleanedText,custom_dict=sensitive_words)
    #stopwords = list(thai_stopwords())+'EMagazine GJ international bangkok hospital'.lower().split(' ')
    stopwords = 'EMagazine GJ international bangkok hospital'.lower().split(' ')
    cleanedText = [i for i in cleanedText if i not in stopwords]
    cleanedText = [i.replace(' ','') for i in cleanedText if len(i) != 1 and len(i) !=0]
    cleanedText = ','.join(cleanedText)
    return cleanedText

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
    hits = util.semantic_search(query_embedding, embeddiing, top_k=15)
    hits = hits[0]
    for hit in hits:
        index_lst.append(hit['corpus_id'])
        score_lst.append(hit['score'])

    sbert_searched = data.iloc[index_lst]
    sbert_searched['score'] = score_lst

    return sbert_searched

def sbert_tfidf_search(queries,head,topic_model=topic_model,data=data,corpus_embeddings=corpus_embeddings):
    
    similar_df = None
    text_to_predict_token = deepcut_tokenizer(queries)

    # Find topics
    try:
        similar_topics, similarity = topic_model.find_topics(text_to_predict_token, top_n=1)
    except:
        similar_topics, similarity = topic_model.find_topics(queries, top_n=1)

    # Example DataFrame
    similar_df = data[data['child_topic'] == similar_topics[0]]

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
    tfidf_matrix = vectorizer.fit_transform(similar_df['text_token'])

    # TF-IDF vector for input text
    text_tfidf = vectorizer.transform([text_to_predict_token])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(tfidf_matrix, text_tfidf)

    # Add similarity scores to DataFrame
    similar_df['score'] = similarity_scores

    similar_df = similar_df.sort_values('score', ascending=False).head(15)

    select_corpus = corpus_embeddings.iloc[similar_df.index.sort_values()]
    similar_embedding = torch.tensor(select_corpus.values)
    similar_searched = sbert_search(queries,similar_df,similar_embedding)
    sbert_searched = sbert_search(queries,data,torch.tensor(corpus_embeddings.values))
    combined_searched = pd.concat([similar_searched,sbert_searched])
    output = combined_searched.sort_values('score', ascending=False).head(head)

    return output

def string_contain_search(queries,sample,data=data):
    data['all_content'] = data['title']+data['content']
    return data[data['all_content'].str.contains(queries,na=False)].sample(sample)

#main
def main():
    #header
    st.markdown("<h1 style='text-align: center; color: black;'>---ระบบแนะนำบทความสุขภาพ---</h1>", unsafe_allow_html=True)
    st.subheader("ให้คะแนนบทความหน่อยนะครับ😄")

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
        optional_keyword = st.text_input('คำค้นหาเพิ่มเติม (ถ้ามี)')

        st.form_submit_button(on_click=set_state,args=(1,))

    if st.session_state.state == 1:

        #asign state
        st.session_state.age = age
        st.session_state.weight = weight
        st.session_state.height = height
        st.session_state.gender = gender
        st.session_state.food_allergy = food_allergy
        st.session_state.drug_allergy = drug_allergy
        st.session_state.congentital_disease = congentital_disease
        st.session_state.optional_keyword = optional_keyword

        #algorithm
        age,gender,bmi = personal_check(age,weight,height,gender)

        if food_allergy == 'ไม่แพ้':
            food_allergy = ''
        if drug_allergy == 'ไม่แพ้':
            drug_allergy = ''
        if congentital_disease == 'ไม่มี':
            congentital_disease = ''
        if congentital_disease != '' or optional_keyword != '':
            queries = optional_keyword+congentital_disease
        else:
            queries = gender+age+bmi+food_allergy+drug_allergy+congentital_disease+optional_keyword
        
        #Bertopic search
        try:
            sbert_searched = sbert_tfidf_search(queries,5)
            string_contain = string_contain_search(queries,5)
            all_recommend = pd.concat([sbert_searched,string_contain])
            all_recommend = all_recommend.drop_duplicates(subset=['url'])

            if len(all_recommend) != 10:
                for i in range(3):
                    if len(all_recommend) < 10:
                        all_recommend = None
                        sbert_searched = sbert_tfidf_search(queries,5+i+1)
                        sbert_searched = sbert_searched.head(5)
                        string_contain = string_contain_search(queries,5+i+1)
                        string_contain = string_contain.head(5)
                        all_recommend = pd.concat([sbert_searched,string_contain])
                        all_recommend = all_recommend.drop_duplicates(subset=['url'])

            st.session_state.sbert_searched_df = sbert_searched
            st.session_state.string_contain_df = string_contain
            st.session_state.string_contain = True

        except:
            sbert_searched = sbert_tfidf_search(queries,10)
            st.session_state.sbert_searched_df = sbert_searched
            all_recommend = sbert_searched

        st.session_state.all_recommend = all_recommend
        st.session_state.queries = queries
        st.session_state.state = 2

    if st.session_state.state == 2:
        placeholder = st.empty()

        #satisfaction
        with placeholder.form('Satisfaction Survey'):
            st.markdown("<h1 style='text-align: center; color: black;'>📰บทความสำหรับคุณ😆</h1>", unsafe_allow_html=True)
            st.header("ระดับความเกี่ยวข้อง")
            st.write("😞 หมายถึง ไม่เกี่ยวข้องเลย")
            st.write("🙁 หมายถึง เกี่ยวข้องเล็กน้อย")
            st.write("😐 หมายถึง เฉยๆ")
            st.write("🙂 หมายถึง ค่อนข้างเกี่ยวข้อง")
            st.write("😀 หมายถึง เกี่ยวข้องมากที่สุด")
            st.write("---------------------------------------------------------------------------------------")

            for i in range(len(st.session_state.all_recommend)):
                st.header(str(i+1)+'. '+st.session_state.all_recommend.iloc[i]['title'])
                st.markdown(f"[Page source (Click here.)]({st.session_state.all_recommend.iloc[i].url})")

                try:
                    banner_url = urllib.parse.quote(st.session_state.all_recommend.iloc[i]['banner'], safe=':/')
                    response = requests.get(banner_url)
                    st.image(response.content)
                except:
                    st.image('https://icon-library.com/images/no-photo-icon/no-photo-icon-1.jpg')

                #satisfaction survey
                st.subheader("Satisfaction Survey")
                st.write("บทความที่แนะนำเกี่ยวข้องกับคุณมากเพียงใด")
                st.radio('ระดับความพึงพอใจ',['NA','😞','🙁','😐','🙂','😀'],horizontal=True,key='score_'+str(i+1))
                st.write("---------------------------------------------------------------------------------------")

            if st.form_submit_button("ยืนยันการส่งคำตอบ"):
                # Check if all articles have satisfaction levels selected
                st.session_state.true_check = []
                for satis_val in [st.session_state[i] for i in ['score_' + str(i+1) for i in range(10)]]:
                    if satis_val != 'NA': 
                        st.session_state.true_check.append(True)
                    else:
                        st.session_state.true_check.append(False)

                if np.all(st.session_state.true_check):
                    st.session_state.state = 3
                    placeholder.empty()
                
                else:
                    idx = []
                    for i in range(len(st.session_state.true_check)):
                        if st.session_state.true_check[i] == False:
                            idx.append(i+1)
                    article_indexes = ', '.join(map(str, idx))
                    st.warning(f":red[กรุณาให้คะแนนบทความที่ {article_indexes} ด้วยครับ]")

    if st.session_state.state == 3:
        st.success('บันทึกคำตอบแล้ว')

        st.session_state.all_recommend = st.session_state.all_recommend.to_dict(orient='records')
        if st.session_state.sbert_searched_df is not None:
            st.session_state.sbert_searched_df = st.session_state.sbert_searched_df.to_dict(orient='records')
        if st.session_state.string_contain_df is not None:
            st.session_state.string_contain_df = st.session_state.string_contain_df.to_dict(orient='records')

        try:
            save_session_state_data(st.session_state.to_dict(), 'satisfaction.csv')

        except:
            #database insertion
            client = MongoClient('mongodb://192.168.1.103:27017/')
            database = client['test']
            collection = database['satisfy_articles']
            collection.insert_one(st.session_state.to_dict())

        finally:
            st.session_state.state = 0

if __name__ == "__main__":
    main()