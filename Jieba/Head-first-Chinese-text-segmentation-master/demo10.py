#encoding=utf-8
import jieba

jieba.set_dictionary("data/dict.txt.big")

content = open('data/lyric2.txt', 'rb').read().decode()

print (content)

words = jieba.cut(content)
print(" / ".join(words))