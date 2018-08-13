import json
import argparse
import requests
from ava_auth import AuthFactory
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed


def parse():
    args = argparse.ArgumentParser('ocr class infer')
    args.add_argument('--gt', type=str, required=True, help='标注过的json文件')
    args.add_argument('--log', type=str, required=True, help='日志文件，每行一个结果（name\tclass）')
    args.add_argument('--ak', type=str, required=True, help='ak')
    args.add_argument('--sk', type=str, required=True, help='sk')
    return args.parse_args()


def token_gen(ak, sk):
    factory = AuthFactory(ak, sk)
    fauth = factory.get_qiniu_auth
    token = fauth()
    return token


def ocr_cls_excutor(img_url, texts, ak, sk):
    res = {}
    request_url = 'http://argus.atlab.ai/v1/eval/ocr-scene-recog'
    headers = {"Content-Type": "application/json"}
    boxes = []
    labels = []
    for text in texts:
        box = []
        for p in text["bboxes"]:
            box.append(int(p))

        boxes.append(box)
        labels.append(text["text"])
    
    body = json.dumps({"data": {"uri": img_url}, "params": {"bboxes": boxes}})
    token = token_gen(ak, sk)
    try:
        r = requests.post(request_url, data=body, timeout=15, headers=headers, auth=token)
    except:
        print('http error.')
    else:
        texts_new = []
        if r.status_code == 200:
            r = r.json()
            #print(r)
            if r['code'] == 0 and r['result']['texts'] is not None:
                texts = r['result']['texts']

                for i in range(0, len(texts)):
                    texts[i]["gt"] = labels[i]

                res["uri"] = img_url
                res["texts"] = texts

                #print("res:", res)
    return res


def data_gen(gt_file):
    urls = []
    bboxes_list = []
    with open(gt_file, 'r', encoding="utf8") as f:
        for line in f:
            line = json.loads(line.strip())
            url = line['url']
            bboxes = line['texts']
            urls.append(url)
            bboxes_list.append(bboxes)
    return urls, bboxes_list


def ocr_cls(img_urls, bboxes_list, log, ak, sk, num_thread=10):
    """
    multithread 
    Args:
    -----
    img_urls : list of url
    log : ocr cls tsv log
    num_thread : num thread

    """
    correct = 0
    sum_text = 0

    with open(log, 'w',encoding="utf8") as f_log:
        with ThreadPoolExecutor(max_workers=num_thread) as exe:

            future_tasks = [exe.submit(ocr_cls_excutor, url, texts, ak, sk) for url, texts in
                            zip(img_urls, bboxes_list)]

            all_url = len(future_tasks)

            count = 1
            for task in as_completed(future_tasks):
                if task.done():
                    print('ocr cls %d/%d' % (count, all_url))
                    count += 1
                    res = task.result()
                    if res == {}:
                        continue
                    for text in res["texts"]:
                        sum_text += 1
                        if text["gt"].replace(' ', '') == text["text"]:
                            correct += 1

                    f_log.write(json.dumps(res, ensure_ascii=False))
                    f_log.write('\n')

        precision = correct * 100 / sum_text
        print("precision:", precision, "%")
        f_log.write("precision:"+ str(precision)+ "%\n")


if __name__ == "__main__":
    args = parse()
    urls, bboxes_list = data_gen(args.gt)
    ocr_cls(urls, bboxes_list, args.log, args.ak, args.sk)
