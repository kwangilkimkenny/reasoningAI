### 입력된 문장의 논리적 관계를 파악하는 기술 연구 ###




# Logical
FullSent = """Never in his life has Bashan caught a hare, nor will he ever; the thing is as good as impossible. Many dogs, they say, are the death of a hare, a single dog cannot achieve it, even one much speedier and more enduring than Bashan. The hare can ``double'' and Bashan cannot --- and that is all there is to it. How Bashan runs! It is beautiful to see a creature expending the utmost of its powers. He runs better than the hare does, he has stronger muscles, the distance between them visibly diminishes before I lose sight of them. And I make haste too, leaving the path and cutting across the park towards the river-bank, reaching the gravelled street in time to see the chase come raging on— the hopeful, thrilling chase, with Bashan on the hare’s very heels; — “One more push, Bashan!” I think, and feel like shouting; “Well run, old chap, remember the double!” But there it is; Bashan does make one more push, and the misfortune is upon us; the hare gives a quick, easy, almost malicious twitch at right angles to the course, and Bashan , with a despairing howl, is left behind. He is not a dog to howl, but he howls now, and I am sorry for him."""

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
with open('./data/BullySentence.pickle', 'rb') as f:
    words = pickle.load(f)


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


def CheckLogic(input_sent):
    # 문장을 리스트로 분리
    sent_token =sent_tokenize(input_sent)
    print(sent_token)
    # 첫 문장과 두번째 문장의 논리성 비교
    sim_re_li = []
    A = len(sent_token)
    print("A:", A)
    for i in range(A):
        try:
            comp_sent = [sent_token[i], sent_token[i+1]]
            sen_embeddings_ = model.encode(comp_sent)

            sim_re = cosine_similarity(
                [sen_embeddings_[0]],
                sen_embeddings_[1:]
            )
            sim_re_li.append(sim_re[0][0])

            # 비교값 확인 - 여기서는 입력된 이전, 이후 문장만 비교함
            # gpt로 생성한 문장을 비교하지는 않음 
            print("문장 비교값 확인: " , sim_re_li)

        except:
            pass
    return sim_re_li




# 입력문장간 논리성 비교
# result = CheckLogic(FullSent)
# print(result)


def GPTLogicChekcer(input_sent):

    # 문장을 리스트로 분리
    sent_token =sent_tokenize(input_sent)
    print(sent_token)
    # 첫 문장과 두번째 문장의 논리성 비교 시작을 위해서
    # 첫 문장과 실제 입력된 다음문장 추출
    # 첫 문장을 기반으로 생성된 6개의 문장들과 문서유사도 개별비교하여 평균값 도출
    k=1
    list_AVG = []
    for sent in sent_token:
            # 6개의 문장 생성
        A = random.uniform(0, 0.01)
        B = random.uniform(0, 0.05)
        C = random.uniform(0, 0.02)
        D = random.uniform(0, 0.04)
        E = random.uniform(0, 0.03)
        F = random.uniform(0, 0.02)
        list_tmps = [A,B,C,D,E,F]

        
        result_genText = []
        for i in list_tmps:
            print("temp_tmp :", i)
            # 6개의 AI로 비교한 문장 유사도 측정결과로 이것이 논리적 분석 값임
            try:
                gen_txt_2nd = GenText(sent, i)
                # 생성된 문장 확인
                print("생성된 문장 확인 :" , gen_txt_2nd)
                result_genText.append(gen_txt_2nd)

                # 생성된 문장 개별 비교
                sim_re_li = []
                for itm in result_genText:
                    itm_=listToString(itm)
                    sent_ = [sent_token[k], itm_]
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
                list_AVG.append(AVG)

                # if AVG >= 0.5:
                #     result = "Logical"
                # else:
                #     result = "Illogical"

                # print("Checking the logicality of a sentence : ", result)
                k += 1
            except:
                pass

    return list_AVG


result_ = GPTLogicChekcer(FullSent)
print(result_)