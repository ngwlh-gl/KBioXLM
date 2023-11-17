import requests
import re,json
from urllib import parse
import html
import time
from tqdm import tqdm
import os
from get_en_knowledge_text import pre_caption
from multiprocessing.dummy import Pool
os.environ["http_proxy"]='127.0.0.1:7890'
os.environ["https_proxy"]='127.0.0.1:7890'
proxies = {
	"http":"http://127.0.0.1:7890",
	"https":"http://127.0.0.1:7890",
}
GOOGLE_TRANSLATE_URL = 'http://translate.google.com/m?q=%s&tl=%s&sl=%s'



def translate(wait_trans, to_language="EN", text_language="zh-CN"):
    result = []
    while True:
        time.sleep(1)
        try:
            text = parse.quote(wait_trans)
            url = GOOGLE_TRANSLATE_URL % (text,to_language,text_language)
            response = requests.get(url,proxies=proxies)
            data = response.text
            expr = r'(?s)class="(?:t0|result-container)">(.*?)<'
            result = re.findall(expr, data)
            break
        except TimeoutError:
            print("time out")
            continue  # 如果请求超时，继续循环发送请求
        except Exception as e:
            # 其他异常处理
            print(e, wait_trans)
            break
    if len(result):
        return html.unescape(result[0])

def init():
    ...

def translation():
    w_f=open('/data1/gl/project/program/chinese_data/code-switch-data/chinese_data_translation_20w_2.json','w',encoding='utf-8')
    with open('/data1/gl/project/program/chinese_data/cnki/filter_cnki_article.json','r',encoding='utf-8') as f:
        cons=f.readlines()
    bar=tqdm(cons)
    count=0
    texts=[]
    idx=0
    for line in bar:
        line=json.loads(line)
        text=line['text']
        text=pre_caption(text)
        # text_lst=list(text)
        count+=1
        if count>50000:
            if len(list(text))<200:
                if text:
                    texts.append(text)
                    if len(texts)==50000:
                        break
                

    with Pool(50, initializer=init, initargs=()) as pool:
        with tqdm(pool.imap(translate, texts), total=len(texts),desc = "building datasets...") as pbar:
            for text,res in zip(texts,pbar):
                if res!=None:
                    print(res)
                    json.dump({'en_text':text.strip(),'zh_text':res.strip()}, w_f, ensure_ascii=False)
                    w_f.write('\n')


if __name__=="__main__":
    translation()
   