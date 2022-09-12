#### 두개의 문장사이의 논리적 관계를 판단하는 연구 ####



# 수학문제 글 입력 (논리적이거나 아니거나)
# AI 는 사용자가 입력한 문장들을 개별 문장으로 나누고, 각 문장을 입력으로 새로운 문장을 자체 생성한다. 생성한 문장은 논리적인 문장일 것이라는 가설.(테스트 완료)
# 수학문제 문장들을 개별 문장으로 분해해서 리스트에 담는다. 
# 문서유사도를 비교하는 함수를 작동시킨다. (AI가 생성한 문장 1개  VS. 인간입력문장중 1개)
# 문서유사도 결과는 확률값으로 나온다. 예) 0.7 --> 70% 유사하다. 
# 정교하게 만들기 위해서 AI 10개를 돌리고, 합산 후 비교평균값을 산출하여 개별문장 점수를 얻어내고
# 모든 입력 문장에 대한 비교점수를 합산 후 평균을 내면, 결과점수는 바로 논리적인지 아닌지 알 수 있는 수치로 출력된다. 

# 문장 예제 입력(인간 작성 문장)
# input_sent = """You need flour to bake bread. You have a sack of flour in the garage. When you get there, you find on top of it a hat that you thought you had lost months ago. So you have to dry it out. To do that,"""
# input_sent = """You order a bowl of cold tomato soup in a restaurant. It looks delicious, but they forgot to bring you a spoon. You try to drink it by pouring it into your napkin."""
# input_sent = """ The restaurant has 175 normal chairs and 20 chairs for babies. How many tables does the restaurant have in total?"""
input_sent ="""You need flour to bake bread. You have a sack of flour in the garage. When you get there, you find that it got thoroughly soaked in a heavy rain last night. So you have to dry it out before you can use it."""

#input_sent = """You poured yourself a glass of cranberry, but then absentmindedly, you poured about a teaspoon of grape juice into it. It looks OK. You try sniffing it, but you have a bad cold, so you can’t smell anything. You are very thirsty. So you drink it."""

# input_sent = """Cayley earns five dollersan hour by delivering newspapers. She delivers newspapers three days each week, for four hours at a time. After delivering newspapers for eight weeks. How much does Cayley earn? """
# 다음 문장 입력(인간입력) 논리 - 이 문장을  AI와 비교할 것임, 이 문장은 논리적인 추론 결과로 작성된 문장임. 
# input_sent_next = """So you take the hat and put it on your head. You go back to the kitchen and find that you have forgotten to buy eggs."""

# input_sent_next = """Callie earn hundred eighty."""
# input_sent_next = "I couldn't eat with a napkin."

#input_sent_next = "You are now dead."
input_sent_next = "You can do this by spreading it out on a table and putting a fan on it."

# 다음 문장 입력(인간입력) 비논리 - 이 문장을  AI와 비교할 것임, 이 문장은 비논리적인 추론 결과로 작성된 문장임. 
# input_sent_next =""" you spread it out on a tarp in the sun."""
# 해석 : Flour that has gotten soaked has to be thrown out; drying it will not help.

# 문장을 sentence token으로 분리하여 리스트에 저장
from random import random
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
sentences  = sent_tokenize(input_sent)
#print(sentences)

# GPT 논리적 문장 생성
import requests
import json
import re
import pickle
import random
from happytransformer import HappyGeneration
from happytransformer import GENSettings

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-2.7B")

# Gramma-correction
from happytransformer import  HappyTextToText
grammaCorrection = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
from happytransformer import TTSettings

beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=100)

def grmmaCorrection(input_sentence):
    output_text= grammaCorrection.generate_text(input_sentence, args=beam_settings)
    return output_text


def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()


# 터부시되는 단어들 불러오기
with open('./data/BullySentence.pickle', 'rb') as f:
    words = pickle.load(f)


def CheckBullySent(inpSents):
    inputSent = inpSents.lower()
    inpSentToken = sent_tokenize(inputSent)
    catchSent = []
    for i in inpSentToken:
        i_ = word_tokenize(i)
        for k in word_tokenize(words):
            if k in i_:
                catchSent.append(i)
    #check
    if len(catchSent) == 0:
        print("No forbidden word was found. Go ahead.")
        result = True
    if len(catchSent) != 0:
        print("A forbidden word was found. Generate again.")
        result = False
    
    return result

def Clean_text(inputString):
    fixed_sentg_re = inputString.text
    fixed_sentg_re.strip("'")
    fixed_sentg_re.lstrip("TextToTextResult(text='")
    fixed_sentg_re.lstrip("TextToTextResult(text=\ ")
    fixed_sentg_re.lstrip(" \ ") 
    fixed_sentg_re.lstrip("**_")
    fixed_sentg_re.lstrip(" \' ") 
    fixed_sentg_re.lstrip("\')")
    fixed_sentg_re.lstrip("')")
    fixed_sentg_re = re.sub(r"\n", "", fixed_sentg_re)
    text_rmv = re.sub('[-=+,#/\?:^.@*\"※~ㆍ!』‘|\(\)\[\]`\'…》\”\“\’·]', ' ', fixed_sentg_re)
    text_rmv = ' '.join(text_rmv.split())
    return text_rmv


# Gramma-correction
from happytransformer import  HappyTextToText
grammaCorrection = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
from happytransformer import TTSettings

beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=100)

def GrammCorrection(input_sentence):
    output_text= grammaCorrection.generate_text(input_sentence, args=beam_settings)
    return output_text


## 6가지의 랜덤값을 도출하는 방식 -- 이것을 사용

from pprint import pprint

def GenText(input_txt, temp):
    resp = []
    resp_ = []
    resp__ = []
    
    #print("temp :", temp)

    top_k_sampling_settings = GENSettings(do_sample=True, 
                                        top_k=50, 
                                        temperature=temp,
                                        num_beams=1,
                                        min_length=10,  
                                        max_length=13,
                                        top_p=1.0,
                                        no_repeat_ngram_size=2,
                                        bad_words=words)
    ### Greedy...
    #top_k_sampling_settings = GENSettings(max_length=10, no_repeat_ngram_size=2, bad_words=words)
    # top_k_sampling_settings = GENSettings(do_sample=True, top_k=0, temperature=temp,  max_length=10, bad_words=words)
    # top_k_sampling_settings = GENSettings(do_sample=True, 
    #                                         top_k=200, 
    #                                         temperature=temp,  
    #                                         max_length=20, 
    #                                         no_repeat_ngram_size=2)

    result_top_k_sampling = happy_gen.generate_text(input_txt, args=top_k_sampling_settings)

    #전처리 필요함 GenerationResult(text=' 
    result_top_k_sampling_re = result_top_k_sampling.text
    result_top_k_sampling_re.strip("'")
    result_top_k_sampling_re.lstrip("GenerationResult(text=' ")
    result_top_k_sampling_re = re.sub(r"\n", "", result_top_k_sampling_re)


    # 생성된 글을 sentence tokenize 
    sent_token_re = sent_tokenize(result_top_k_sampling_re)

    # 마지막 문장 가져오기
    inp_sent = str(sent_token_re[-1])

    # 완결된 문장으로 수정하기
    fixed_sent = GrammCorrection(inp_sent)

    fixed_sent_ = Clean_text(fixed_sent)

    fixed_sent_re = str(fixed_sent_)

    # 마지막 요소 제거 후
    sent_token_re.pop()

    # 완결문장으로 수정하여 추가
    sent_token_re.append(fixed_sent_re)

    for i in sent_token_re:
        resp_.append(i)

    result = listToString(resp_)
    #print("result :" + result)
    #resp__.append(result)


    #폭력/성적인 단어 문장검사
    chk_re = CheckBullySent(result)
    if chk_re == True: # 발견되지 않았기 때문에 이대로 진행
        #print("Go ahead!")
        resp.append(result)
    elif chk_re == False: # 발견되었기 때문에 재생성(단일문장 재생성)
        regSent = ReGenText_Rendom(result) #### 단 또다시 발견되었을 경우... 1회만 재생성
        regSent_ = reCheck(regSent)
        resp.append(regSent_)

    #print("Generated sentence : ", resp)
        
    return resp

# 단일 문장생성결과를 수정하여 재생성하는 기능
def ReGenText_Rendom(input_txt):
    temp = random.uniform(0, 1)
    resp = []
    print(temp)
    top_k_sampling_settings = GENSettings(do_sample=True, 
                                            top_k=50, 
                                            temperature=temp,
                                            num_beams=5,
                                            min_length=10,  
                                            max_length=13,
                                            top_p=5,
                                            no_repeat_ngram_size=2,
                                            bad_words=words)
    ### Greedy...
    #top_k_sampling_settings = GENSettings(max_length=10, no_repeat_ngram_size=2)

    ### Standard ...
    ##top_k_sampling_settings = GENSettings(do_sample=True, top_k=0, temperature=0.7,  max_length=10)

    result_top_k_sampling = happy_gen.generate_text(input_txt, args=top_k_sampling_settings)
    
    #전처리 필요함 GenerationResult(text=' 
    result_top_k_sampling_re = result_top_k_sampling.text
    result_top_k_sampling_re.strip("'")
    result_top_k_sampling_re.lstrip("GenerationResult(text=' ")
    result_top_k_sampling_re = re.sub(r"\n", "", result_top_k_sampling_re)


    # 생성된 글을 sentence tokenize 
    sent_token_re = sent_tokenize(result_top_k_sampling_re)

    # 마지막 문장 가져오기
    inp_sent = str(sent_token_re[-1])

    # 완결된 문장으로 수정하기
    fixed_sent = grmmaCorrection(inp_sent)

    fixed_sent_ = Clean_text(fixed_sent)

    fixed_sent_re = str(fixed_sent_)

    # 마지막 요소 제거 후
    sent_token_re.pop()

    # 완결문장으로 수정하여 추가
    sent_token_re.append(fixed_sent_re)

    for i in sent_token_re:
        resp.append(i)

    result = listToString(resp)

    return result # 완성하였음. 테스트 필요함



def reCheck(inpSents):
    if CheckBullySent(inpSents) == False: # 발견되면 새로생성
        result = ReGenText_Rendom(inpSents)
    else: # 발견되지 않으면 생성한 문장을 그냥 사용하기
        result = inpSents
    return result

def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    return result.strip()



# # 문장 생성한 결과 
# gen_txt  = GenText(input_sent, A)
# print("===================")
# print("Answer : " , gen_txt)
# print("===================")

# # 인간입력문장 vs. AI 생성 문장 유사도 비교

# sen = [input_sent_next, gen_txt]


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
# #Encoding:
# sen_embeddings = model.encode(sen)

from sklearn.metrics.pairwise import cosine_similarity
#let's calculate cosine similarity for sentence 0:
# sim_re = cosine_similarity(
#     [sen_embeddings[0]],
#     sen_embeddings[1:]
# )


# # 1개의 AI로 비교한 문장 유사도 측정 결과 input_sent_A. vs. input_sent_B
# print("AI resoning result : ", sim_re[0][0])




### generation text - 6ea
# A = random.uniform(0, 0.20)
# B = random.uniform(0.20, 0.40)
# C = random.uniform(0.40, 0.60)
# D = random.uniform(0.60, 0.80)
# E = random.uniform(0.80, 0.99)
# F = random.uniform(0, 1.0)

A = random.uniform(0, 0.01)
B = random.uniform(0, 0.05)
C = random.uniform(0, 0.02)
D = random.uniform(0, 0.04)
E = random.uniform(0, 0.03)
F = random.uniform(0, 0.02)


list_tmps = [A,B,C,D,E,F]



k=0
result_genText = []
for i in list_tmps:
    print("temp_tmp :", i)
    # 6개의 AI로 비교한 문장 유사도 측정결과로 이것이 논리적 분석 값임
    gen_txt_2nd = GenText(input_sent, i)
    # 생성된 문장 확인
    print("생성된 문장 확인 :" , gen_txt_2nd)
    result_genText.append(gen_txt_2nd)
    k += 1



# 생성된 문장 개별 비교
sim_re_li = []
for itm in result_genText:
    itm_=listToString(itm)
    sent_ = [input_sent_next, itm_]
    sen_embeddings_ = model.encode(sent_)

    sim_re = cosine_similarity(
        [sen_embeddings_[0]],
        sen_embeddings_[1:]
    )
    sim_re_li.append(sim_re[0][0])

# 비교값 확인
print("유사문장 비교값 확인: " , sim_re_li)

# 보정을 위해서 최대, 최소값 삭제
min_value = min(sim_re_li)
max_value = max(sim_re_li)

sim_re_li.remove(min_value)
sim_re_li.remove(max_value)

print("최대 최소값 삭제 확인: " , sim_re_li)

# 개별 비교 분석 결과 평균값 산출하기
import numpy as np
a = np.array(sim_re_li)
AVG = np.mean(a)

#print(" AI resoning result score :", AVG)
#print(" AI resoning result score :" , AVG)

if AVG >= 0.5:
    result = "Logical"
else:
    result = "Illogical"

print("Checking the logicality of a sentence : ", result)
