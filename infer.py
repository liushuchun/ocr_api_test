# -*- coding: utf-8 -*-
# 调用ocr_cls原子API服务
# 默认线程10

ak = ""
sk = ""

import json
import argparse
import requests
from ava_auth import AuthFactory
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

def parse():
    args = argparse.ArgumentParser('ocr class detections infer')
    args.add_argument('--mode',type = str,required = True,choices = ['cls','det'],help='OCR检测 或 识别')
    args.add_argument('--gt',type = str,required = True, help = '标注过的json文件')
    args.add_argument('--log',type = str,required = True, help = '日志文件，每行一个结果（name\tclass）')
    args.add_argument('--ak',type = str,required = True,help = 'ak')
    args.add_argument('--sk',type = str,required = True,help = 'sk')
    return args.parse_args()

def token_gen(ak,sk):
    factory = AuthFactory(ak,sk)
    fauth = factory.get_qiniu_auth
    token = fauth()
    return token

def ocr_cls(img_url,ak,sk):
    res = {}
    request_url = 'http://argus.atlab.ai/v1/eval/ocr-classify'
    headers = {"Content-Type": "application/json"}
    body = json.dumps({"data": {"uri": img_url}})
    token = token_gen(ak,sk)
    try:
        r = requests.post(request_url, data=body,timeout=15, headers=headers, auth=token)
    except:
        print('http error.')
    else:
        if r.status_code == 200:
            r = r.json()
            if r['code'] == 0 and r['result']['confidences'] is not None:
                img_name = img_url.split('/')[-1]
                res['img_name'] = img_name
                label = r['result']['confidences'][0]['class']
                res['label'] = label
    return res

def ocr_det(img_url,ak,sk):
    res = {}
    request_url = 'http://argus.atlab.ai/v1/eval/ocr-detect'
    headers = {"Content-Type": "application/json"} 
    body = json.dumps({"data": {"uri": img_url}})
    token = token_gen(ak,sk)
    try:
        r = requests.post(request_url, data=body,timeout=15, headers=headers, auth=token)
    except:
        print('http error.')
    else:
        if r.status_code == 200:
            r = r.json()
            if r['code'] == 0:
                img_name = img_url.split('/')[-1]
                res['img_name'] = img_name
                label = r['result']['detections']
                res['label'] = label
    return res


def url_gen(gt_file):
    urls = []
    with open(gt_file,'r') as f:
        for line in f:
            line = json.loads(line.strip())
            url = line['url']
            urls.append(url)
    return urls

def ocr_infer(mode,img_urls,log,ak,sk,num_thread=10):
    """
    multithread 
    Args:
    -----
    img_urls : list of url
    log : ocr cls tsv log
    num_thread : num thread

    """
    with open(log,'w') as f_log:
        with ThreadPoolExecutor(max_workers=num_thread) as exe:
            if mode == 'cls':
                future_tasks = [exe.submit(ocr_cls, url,ak,sk) for url in img_urls]
            else:
                future_tasks = [exe.submit(ocr_det, url,ak,sk) for url in img_urls]
            all_url = len(future_tasks)
            count = 1
            for task in as_completed(future_tasks):
                if task.done():
                    print('ocr cls %d/%d'%(count,all_url))
                    count += 1
                    res = task.result()
                    if res == {}:
                        continue
                    f_log.write('%s\t%s'%(res['img_name'],res['label']))
                    f_log.write('\n')

if __name__ == '__main__':
    args = parse()
    urls = url_gen(args.gt)
    ocr_infer(args.mode,urls,args.log,args.ak,args.sk)
