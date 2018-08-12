# PULP TEST TOOL
* 调用线上剑皇服务,支持七牛/图普/阿里/百度
* 获取剑皇调用日志
* 计算剑皇指标

## requirements
* futures
* requests
* numpy

## Usage
```shell
python test.py --gt data.json --tool qpulp --log log.json --ak xxx --sk xxx --vis
```
## 参数
-  **gt**     ground truth 文件路径，labelx格式的json文件
-  **tool**     平台选择，目前支持七牛qpulp，图谱nrop，阿里ali，百度baidu
-  **log**      日志文件，若要保存日志文件，需指定日志文件名；若不指定，则不保存日志
-  **vis**      若指定，则绘制混淆矩阵
-  **as**       百度ak，仅 tool 指定为baidu时填写
-  **sk**       百度sk，仅 tool 指定为baidu时填写

