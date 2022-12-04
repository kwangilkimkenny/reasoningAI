### 입력된 문장의 논리적 관계를 파악하는 기술 연구 ###

# 입력문장들 사이의 논리구성을 확인하는 기능으로, 전체 문장의 맥락을 확인하지 않고 단일문장간 논리구성만 파악하는 기능
# 향후에 입력된 이전 문장을 모두 검토해서 논리성을 파악하는 기능으로 업그레이드 해야 함

# 결과는 모든 문장의 논리구성을 파악함. 입력문장이 8개면 8개의 논리구성분석 결과값을 도출하게 되어있음



# Illogical
# FullSent = """Anna had been studying hard for the physics test for days; she had reviewed every page of the textbook and had done hundreds of practice problems. Lucy, on the other hand, had been busy having a great time with her friends all week. She glanced over the textbook the night before the exam; it looked more or less familiar. Not surprisingly, therefore, Lucy did not do as well on the test as Anna did.
# The next day, Lucy was in a good mood."""

### 분장 분석 시작
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
import re
import pickle
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
from sklearn.metrics.pairwise import cosine_similarity
import random
from happytransformer import HappyGeneration
from happytransformer import GENSettings

happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-2.7B")
# 터부시되는 단어들 불러오기
# with open('./data/BullySentence.pickle', 'rb') as f:
#     words = pickle.load(f)


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
                                        no_repeat_ngram_size=2
                                        )
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
    fixed_sent = grmmaCorrection(inp_sent)

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
        
    return result

# sigle 문장으로 single 문장생성
def genTxtFromInpSent(inp_single_sent):
    para_A = random.uniform(0, 0.01)
    gen_txt_2nd = GenText(inp_single_sent, para_A)
    gen_re=listToString(gen_txt_2nd)


def checkLogic(inp_value):
    if inp_value >= 0.2:
        each_sent_result = "contextual"
    else:
        each_sent_result = "non-contextual"

    #print(" AI resoning result score :", each_sent_result)
    return each_sent_result

def GPTLogicChekcer(input_sent):
    # 문장을 리스트로 분리
    sent_token =sent_tokenize(input_sent)
    print("입력문장의 수 :", len(sent_token))
    
    k=0
    itms = []
    sent_logic_line_by_line_list = []
    for sent in sent_token:
        gre_1 = genTxtFromInpSent(sent)
        itms.append(gre_1)
        gre_2 = genTxtFromInpSent(sent)
        itms.append(gre_2)
        gre_3 = genTxtFromInpSent(sent)
        itms.append(gre_3)
        
        sim_re_li =[]
        for itm_ in itms:
            sent_ = [sent_token[k], itm_]
            sen_embeddings_ = model.encode(sent_)

            sim_re = cosine_similarity(
                [sen_embeddings_[0]],
                sen_embeddings_[1:]
            )
            sim_re_li.append(sim_re[0][0])

            # print("유사문장 비교값: " , sim_re_li)

        try:
            a = np.array(sim_re_li)
            AVG = np.mean(a)

            sent_logic_line_by_line_list.append(AVG)
        except:
            pass

        k += 1
       
    print("sent_logic_line_by_line_list :", sent_logic_line_by_line_list)
    # 문장들을 리스트로, 각 문장vs3개의 문장생성 후 최대최소값 삭제하여 보정 후, 평균값을 도출하여,
    # 모든 문장과 다음 문장간의 contextual 관계를 파악하기위한 기준점수 산출하여 리스트로 저장하기
    # 저장한 리스트값을 실제 문장간의 contextual값 계산결과와 비교하여 유사하면(0.2이상)) contextual, 유사하지 않으면(0.2이하) non-contextual 판정
    
    result = []
    for ittm in sent_logic_line_by_line_list:
        
        chk_logic_re = checkLogic(ittm)
        result.append(chk_logic_re)
        
        
    return result   


# Logical
FullSent = """When I realized I cannot understand the world. I recently debated at the Orange County Speech League Tournament, within the Parliamentary Division. This specific branch of debate is an hour long, and consists of two parties debating either side of a current political issue. In one particular debate, I was assigned the topic: “Should Nation States eliminate nuclear arms?” It so happened that I was on the negative side and it was my job to convince the judges that countries should continue manufacturing nuclear weapons. During the debate, something strange happened: I realized that we are a special breed of species, that so much effort and resources are invested to ensure mutual destruction. And I felt that this debate in a small college classroom had elucidated something much more profound about the scale of human existence. In any case, I won 1st place at the tournament, but as the crowd cheered when my name was called to stand before an audience of hundreds of other debaters, and I flashed a victorious smile at the cameras, I couldn’t help but imagine that somewhere at that moment a nuclear bomb was being manufactured, adding to an ever-growing stockpile of doom. And that's when I realized that the world was something I will never understand."""

FullSent_list = [
"Charlene had a pack of thirty five pencil crayons. She gave six to her friend Theresa. She gave three to her friend Mandy. How many pencil crayons does Charlene have left?",

"A movie theatre has twenty five rows of seats with twenty seats in each row. How many seats are there in total?",

"Cayley earns five dollars an hour by delivering newspapers. She delivers newspapers three days each week, for four hours at a time. After delivering newspapers for eight weeks, how much money will Cayley earn?",

"The school has twenty thousands to buy new computer equipment. If each piece of equipment costs fifty, how many pieces can the school buy in total?",

"Rebecca left her dad’s store to go home at twenty to seven in the evening. Forty minutes later, she was home. What time was it when she arrived home?",

"The restaurant has one hundred seventy five normal chairs and twenty chairs for babies. How many tables does the restaurant have in total?",

"Adrianna has fifteen pieces of gum to share with her friends. When she went to the park, she shared ten pieces of strawberry gum. When she left the park, Adrianna shared another ten pieces of bubble gum. How many pieces of gum does Adrianna have now?",

"Ashley bought a big bag of candy containing a total of two hundred blue, red, and green candies. The bag had 102 blue candies and 100 red candies. How many green candies were there in total?",

"An Italian restaurant receives a odd number of veal cutlets. If it takes four cutlets to make a dish and no cutlets are left over, how many dishes can be made?",

"Retta put one hundred in a bank account that gains twenty percent interest annually. If she makes no withdrawals, how many years would it take until there is $80.00 in the account?"


]

result_li = []
for i in FullSent_list:
    res = GPTLogicChekcer(i)
    print(res)
    result_li.append(res)

print("-" * 20)
print(result_li)