# GPT2 Kullanarak bir ChatBot geliştirelim.

![alt text](https://miro.medium.com/max/2800/0*SS2dhjs-1K2uUwwe.jpeg)

Herkese merhaba bu yazımda GPT2 kullanarak bir Chatbot uygulaması geliştirmekten bahsedeceğim . Uygulamadan kısaca bahsetmek gerekirse, genel olarak soru cevap şeklinde yada karşılıklı konuşma şeklinde toplanan verilerin bulunduğu bir veri setine ihtiyacımız var. Bu veri setini ele alarak kullanıcının girmiş olduğu/ yada söylemiş olduğun (input tipi size bağlı nasıl gelirtirmek isterseniz tip olarak Text yada Speech olabilir) veriye göre bir sonuç üretip bu sonucu geri döndürecek.
𝗩𝗲𝗿𝗶 𝘀𝗲𝘁𝗶

Öncelikle biz bu projemizde 𝘾𝙤𝙧𝙣𝙚𝙡𝙡 𝙈𝙤𝙫𝙞𝙚 - 𝘿𝙞𝙖𝙡𝙤𝙜𝙨 𝘾𝙤𝙧𝙥𝙪𝙨 isimli dataseti kullanacağız bu dataset İngilizce olup içerisinde onlarca filimin alt yazılarını içermekte.


Biz bu dataseti için sadece iki dosyayı ele alacağız. 
movie_conversations.txt ve movie_lines.txt isimlerden anlayabileceğimiz gibi 

movie_conversations: Konuşma İndexlerini içeriyor. 

movie_lines: Konuşma metinlerini içeriyor.


![alt text](https://miro.medium.com/max/2588/1*oimEe74o_u50Bavp9Y4nrQ.png)


Yukarıda paylaştığım ekran görüntüsüne bakarsanız Movie_Lines veri setimizin oldukça düzensiz olduğunu görebilirsiniz. İlk önce amacımız bu setimizi düzeltmek veri seti hakkında hangi index neyi belirtiyor gibi bilgilere girmek istemiyorum burada sadece bizi ilgilendiren kısım tüm satırların başında bulunan "id" kısmı ve son index içinde bulunan "konuşma metini".

![alt text](https://miro.medium.com/max/1086/1*yR_bim7nbV7d5RE4F4jNyA.png)

Bu ekran görüntüsünde ise bizim için önemli olan sadece son index.

['L866', 'L867', 'L868', 'L869']

Burada bu array içindeki elemanlar bizim yukarıda düzenlemek istediğimiz veri seti için çok önemli daha anşılır olması için ele aldığımız örnek için konuşmak gerekirse array içinde bulunan her eleman(L866, L867 gibi ) bizim için Movie_Lines veri setindeki bulunan şu konuşma metinlerini point ediyor .

'L869': 'Like my fear of wearing pastels?',

'L868': 'The "real you".',

'L867': 'What good stuff?', 

'L866': "I figured you'd get to the good stuff eventually.",

Veri setini istenen hale getirmek için yapmamız gerekenler ;

https://gist.github.com/kubilaybzk/2b74742f1c49d7aab22b7884e577df98#file-adimlar-py



Yaptığımız adımlar tek tek görseller ile göstermek gerekir ise ;
![Movie_Lines veri setinin ilk hali ](https://miro.medium.com/max/700/1*nmsVzaG3uMbHZzD3kVYahg.png)

Movie_Lines veri setinin ilk hali

![movie_conversations veri setinin ilk hali](https://miro.medium.com/max/700/1*18tILSD37gp8NWBkC1lpTQ.png)

movie_conversations veri setinin ilk hali

![Sadece Metinleri ve Metinlere Point eden İndex elemanlarını aldık .](https://miro.medium.com/max/700/1*jLx8_XtfGbTzdL8zu2dCjQ.png)

Sadece Metinleri ve Metinlere Point eden İndex elemanlarını aldık .

![alt text](https://miro.medium.com/max/700/1*y26xC0Gcg2WY4GJEN08Y0A.png)

Alıntıları gruplandırdık.

![Alıntıları gruplandırdık.](https://miro.medium.com/max/700/1*5w9a_RM3YK2iEH43QpmwiQ.png)

İstenen sonuç

![İstenen sonuç](https://miro.medium.com/max/700/1*5w9a_RM3YK2iEH43QpmwiQ.png)






# GPT2 GEREKSİNİMLERİNİN EDİNİLMESİ.
https://gist.github.com/kubilaybzk/96eeb44bfe5faeed97541ff87497e98e#file-import-py

Botumuzun çalışması için gerekli olan gereksinim olan GPT2 modelimizi yukarıda gösterdiğim kodları çalıştırarak Colab yada bilgisayarımız import ediyoruz. Daha sonra veri setimizi GPT2 modelimize import etdebilmemiz için bir ".txt" dosyasına çevirmemiz gerekiyor bunun için şu adımları uyguluyoruz.

https://gist.github.com/kubilaybzk/9e7224ebe977127a5fc192ab5a9b0b8f#file-set_date-py

Bu sayede verimizin son şekli görseldeki gibi oluyor .
![Son gruplandırma.](https://miro.medium.com/max/700/1*PfL2KgxjCFGruc4pWwyB7w.png)

𝗠𝗼𝗱𝗲𝗹𝗶𝗻 𝗲𝗴̆𝗶𝘁𝗶𝗹𝗺𝗲𝘀𝗶.
https://gist.github.com/kubilaybzk/109161418fecc993b45080e3e3e7a718#file-train-py

Yukarıda bulunan iki adet kod satırımızı çalıştırarak modelimizi eğitmeye başlıyoruz. Bu eğitim verinin büyüklüğüne bilgisayarınızın hızına göre değişiklik gösterebilir. Ek olarak modeli eğitirken GPU üzerinden çalıştırmayı unutmayın.
Modelin eğitimi bittikten sonra daha önceki yazılarımda anlatmış olduğum 
TextBlob ve gTTS kullanarak modelimizi Türkçe ve sesli hale getirelim .

https://gist.github.com/kubilaybzk/30a0183474363368e59c6be060c8831c#file-main-py

TextBlob:
https://kubilaybozak.medium.com/working-on-natural-language-processing-with-textblob-f4c80cd0c8ad

gTTS:
https://kubilaybozak.medium.com/convert-text-to-speech-with-colab-9ed89d310cba


Son olarak bir run alıp sonuca hep beraber bakalım. İşte Bu !

![Son](https://miro.medium.com/max/486/1*kP_866mHexGjFXXpM0Lfwg.png)



Colab üzerinden test etmek için . 
https://colab.research.google.com/drive/1pWCo4OONFaEeedbFUokGzGPjpEfqe2IQ?usp=sharing.
