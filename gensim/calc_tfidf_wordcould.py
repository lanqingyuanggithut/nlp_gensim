# -*- coding: utf-8 -*-
import os
import logging
import time
import jieba
import re
from scipy.misc import imread  #这是一个处理图像的函数
from wordcloud import WordCloud,ImageColorGenerator
import matplotlib.pyplot as plt
from gensim import corpora,models
import threading
from queue import Queue

q = Queue()#队列

if not os.path.exists("logs\\"):
    os.makedirs("logs\\")
time_now = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_name = "logs\\" + time_now + '.log'

logger = logging.getLogger()#创建一个logger
logger.setLevel(logging.INFO)#log等级总开关
fh = logging.FileHandler(log_name, mode='w')#创建一个handler，用于把日志写入文件
fh.setLevel(logging.WARNING)#输出到file的log等级的开关,包括该级别
ch = logging.StreamHandler()#创建一个handler，用于把日志输出输出到控制台
ch.setLevel(logging.INFO)#输出到console的log等级的开关,包括该级别
formatter = logging.Formatter("%(asctime)s - %(filename)s\
[line:%(lineno)d] - %(levelname)s: %(message)s")#定义handler的输出格式
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

def get_files(path):
    """
    返回所有语料库的文档名字列表
    :param path: 语料库路径
    :return files_list: 返回所有语料库的文档名字列表
    """
    files_list = os.listdir(path)
    return files_list

def get_jieba_wordlist(path_filename):
    """
    输入当前文档，返回该文档的分词列表，分词列表去除所有特殊字符和英文数字
    :param path_filename:
    :return words: 返回该文档的分词结果列表    """

    with open("stopwords.txt","r",encoding="utf-8") as f:
        stoplist=f.read()
    with open(path_filename,"r",encoding="utf-8") as f:
        text=f.read()

    r1 = u'[ a-zA-Z0-9’!"#$%&\'()*+,-.·/:;<=>?@，。?★、…【】~《》？“”‘’！[\\]^_`{|}~\n]+'
    wordList = jieba.lcut(text)#jieba精确分词,返回列表
    words = []
    for word in wordList:#去除所有特殊字符、英文数字、停用词
        word = re.sub(r1, "", word)
        if word != "" and word not in stoplist.split('\n'):
            words.append(word)
    return words#返回分词结果列表

def multi_thread(path_filename):
    """
    多线程分词，将每篇文档的分词结果列表入队列
    :param path_filename: 文档路径及文档名
    :return: 无返回值
    """
    words = get_jieba_wordlist(path_filename)
    q.put(words)#分词结果列表入队列

def get_corpus(path):
    """
    获得路径下的所有文本的list，每个文本按空格分为list，形式为[[],[],[],·····]
    :param path: 语料路径（所有文档存放路径）
    :return corpus_list: 语料
    """
    corpus_list = []
    files_list = get_files(path)#获取文件目录，返回文件名list
    threadList = []  # 存储线程的列表集合
    for cur_filename in files_list:
        path_filename = os.path.join(path, cur_filename)
        t = threading.Thread(target=multi_thread, args=(path_filename,))#多线程分词
        threadList.append(t)
        t.start()
    for thread in threadList:#阻塞主线程，让子线程运行完毕
        thread.join()
    while not q.empty():
        corpus_list.append(q.get())
    return corpus_list

def train_tfidf_model(corpus_list):
    """
    训练tfidf模型，并保存词典、语料、tfidf模型
    :param corpus_list: 包含所有语料的list，一个文件为其中一个元素,形式为[[],[],[],·····]
    :return: 无返回值
    """
    dictionary = corpora.Dictionary(corpus_list)  #基于语料文本建立词典
    dictionary.save('libai_dictionary.dict')  # 保存词典
    corpus = [dictionary.doc2bow(text) for text in corpus_list]#由词袋向量组成的列表构成语料
    corpora.MmCorpus.serialize('libai_corpus.mm', corpus)  #将corpus持久化到磁盘中
    tfidf_model = models.TfidfModel(corpus=corpus, dictionary=dictionary)#得到训练的tfidf模型
    tfidf_model.save('libai_tfidf.model')#保存模型

def test_tfidf_model(test_doc):
    """
    测试训练好的tfidf模型
    :param test_doc: 测试文档分词列表
    :return test_doc_tfidf_dict: 返回测试文档的tfidf值
    """
    dictionary = corpora.Dictionary.load('libai_dictionary.dict')  #加载词典
    new_dict = {v: k for k, v in dictionary.token2id.items()}
    tfidf_model = models.TfidfModel.load('libai_tfidf.model')#加载模型
    test_word_bow = dictionary.doc2bow(test_doc)  #将词列表转换成稀疏词袋向量
    test_doc_tfidf_dict = {}
    try:
        doc__tfidf = tfidf_model[test_word_bow]
        sorted_words = sorted(doc__tfidf, key=lambda x: x[1], reverse=True) #根据tf-idf值降序排列
        logger.info("-------------------------------------------------------")
        for num, score in sorted_words:
            test_doc_tfidf_dict[new_dict[num]] = score
            logger.info("keyword: %s, tf-idf: %s" % (new_dict[num], score))
        logger.info("-------------------------------------------------------")
    except Exception as e:
        logger.error("异常")
    return test_doc_tfidf_dict

def generate_wordcloud_from_tfidf(tfidf_dict):
    """
    根据词的tfidf值来画词云
    :param tfidf_dict: 字典形式的tfidf值
    :return:无返回值
    """
    back_color = imread('jpg/李白.jpg')  # 解析该图片
    wc = WordCloud(background_color='white',  # 背景颜色
                   max_words=100,  # 最大词数
                   mask=back_color,  #以该参数值作图绘制词云，这个参数不为空时，width和height会被忽略
                   max_font_size=100,  # 显示字体的最大值
                   font_path="C:/Windows/Fonts/simhei.ttf",  #解决显示口字型乱码问题，字体路径C:/Windows/Fonts/
                   random_state=42,  # 为每个词返回一个PIL颜色
                   )
    wc.generate_from_frequencies(tfidf_dict)  #根据词频生产词云,传入的参数tfidf_dict是一个字典的形式
    image_colors = ImageColorGenerator(back_color)#基于彩色图像生成相应彩色
    plt.imshow(wc)# 显示图片
    plt.axis('off')# 关闭坐标轴
    plt.figure()# 绘制词云
    plt.imshow(wc.recolor(color_func=image_colors))
    plt.axis('off')
    jpgName="libai_wordcloud.png"
    wc.to_file(jpgName)#保存词云图片
    plt.show()#显示词云

if __name__=="__main__":
    path="libai_txt\\"
    corpus_list=get_corpus(path)
    train_tfidf_model(corpus_list)#训练模型
    test_doc=get_jieba_wordlist(u"李白诗的集合.txt")#测试文档结巴分词列表
    test_doc_tfidf_dict=test_tfidf_model(test_doc)#测试模型，返回测试文档的tfidf值
    generate_wordcloud_from_tfidf(test_doc_tfidf_dict)#根据tfidf生成词云