# -*- coding: utf-8 -*-
"""CornellMovie_ChatBot_GPT2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pWCo4OONFaEeedbFUokGzGPjpEfqe2IQ

# **Cornell Movie Dialog Corpus Chatbot**
---

*   Kubilay BOZAK  - Furkan Başoğlu

https://kubilaybozak.medium.com/gpt2-kullanarak-bir-chatbot-geli%C5%9Ftirelim-ce76251f159e

# Load Dataset from Google Drive
"""

from google.colab import drive
drive.mount('/content/drive')

# Load the data

#from files
#movie_lines = open('/content/cornell_dataset/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
#movie_conv_lines = open('/content/cornell_dataset/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

# drive shareable link for dataset

#https://drive.google.com/drive/folders/16mBKhvI_t5ylA9w-t6rSmVfErPV7a22h?usp=sharing 

#from drive
movie_lines = open('/content/drive/MyDrive/cornell movie-dialogs corpus/movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
movie_conv_lines = open('/content/drive/MyDrive/cornell movie-dialogs corpus/movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

movie_lines

movie_conv_lines

"""# Data Preprocessing"""

# Create a dictionary to map each line's id with its text => (dictionary yap (id:line), +++$+++ kaldır)
id_line = {}
for line in movie_lines:

    new_line = line.split(' +++$+++ ')
    if len(new_line) == 5:
        id_line[new_line[0]] = new_line[4]

id_line

# Create a list of all of the conversations' lines' ids. (Tüm lineları liste yap)
conver = []
for line in movie_conv_lines[:-1]:
    #convert
    new_line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    conver.append(new_line.split(','))

conver2=[]
for i in conver:
  if(len(i)==4):
    conver2.append(i)

conver2

len(conver2)

conver3=[]
for i in range(0,1000):
  conver3.append(conver2[i])
len(conver3)
conver=conver3

# a example for id and conversation
all_elem=[]
for i in range(0,1000):
  temp=[]
  for k in conver[i]:
      temp.append(id_line[k])
  all_elem.append(temp)

all_elem

"""# Install Dependencies for GPT2


"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
!pip install gpt-2-simple
import gpt_2_simple as gpt2

gpt2.download_gpt2(model_name="355M")

"""#Train"""

data = []
qq=0
aa=0
for i in all_elem:
  for j in range(0,4):
    if(j%2==0):
      q ='[USER] : ' + str(i[j])
      qq=qq+1
      data.append(q)
    else:
      aa=aa+1
      a = '[BOT] : ' + str(i[j])
      data.append(a)
print(aa)
print(qq)

data

with open('chatbot.txt', 'w') as f:
     for line in data:
        try:
            f.write(line)
            f.write('\n')
        except:
            pass

file_name = "/content/chatbot.txt"

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name='355M',
              steps=1000,
              restore_from='fresh',  # change to 'latest' to resume training
              run_name='run1',
              print_every=5,  # how many steps between printing progress
              sample_every=100, # how many steps to print a demo sample
              save_every=100 # how many steps between saving checkpoint 
              )

"""#Test"""

from textblob import TextBlob
!pip install gtts
from gtts import gTTS #Import Google Text to Speech

def ChatApp(Text):
      gelen=Text 
      blob = TextBlob(gelen)
      Eng= blob.translate(from_lang='tr', to='en')
      Eng=str(Eng)
      #print(">> User:" +gelen + " " +Eng)
      #gelen = '[USER] : '+Eng+'\n'+'[BOT] :'
      x = gpt2.generate(sess,
                      length=20,
                      temperature = 0.6,
                      include_prefix=False,
                      prefix = Eng,
                      nsamples = 1,
                      return_as_list=True)[0]
      x=str(x)
      #print("GElen x değeri :" + x +" ++++++++ ")
      blob = TextBlob(x)
      Turkısh= blob.translate(from_lang='en', to='tr')
      Turkısh=str(Turkısh)
      Text=Turkısh+ " " + Text
      print('[BOT]' + Text)
      from gtts import gTTS #Import Google Text to Speech
      from IPython.display import Audio #Import Audio method from IPython's Display Class
      tts = gTTS(Turkısh,lang="tr") #Provide the string to convert to speech
      tts.save('1.wav') #save the string converted to speech as a .wav file
      sound_file = '1.wav'
      return Audio(sound_file, autoplay=True)

ChatApp("Merhaba")

ChatApp("Senin Adın ne ?")

ChatApp("Hayatın anlamı nedir?")

ChatApp("Elon Musk'u tanıyor musun?")

ChatApp("Atatürk'ü tanıyor musun?")

ChatApp("istanbul medipol üniversitesini biliyor musun?")

ChatApp("Seni seviyorum")

ChatApp("istanbul nerede?")

ChatApp("Sen kimsin?")

#Türkçe sorular ve Türkçe cevaplar.
for step in range(5):
       gelen = input("[USER] : ")
       blob = TextBlob(gelen)
       Eng= blob.translate(from_lang='tr', to='en')
       Eng=str(Eng)

       x = gpt2.generate(sess,
                length=20,
                temperature = 0.6,
                include_prefix=False,
                prefix=Eng,
                 return_as_list=True)[0]
       x=str(x)
       blob = TextBlob(x)
       Turkısh= blob.translate(from_lang='en', to='tr')
       Turkısh=str(Turkısh)
       Text=Turkısh+ " " + x
       print('[BOT]' + Text)

#Sadece ingilizce sorular ve cevaplar.

for step in range(5):
      ques = input("[USER] : ")

      x = gpt2.generate(sess,
                length=20,
                temperature = 0.6,
                include_prefix=False,
                prefix=ques,
                nsamples=1,
                )
