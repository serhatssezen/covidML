# covidML
CovidML

Tasarım Aşamaları\n
COVİD-19 hastalarının çekilen akciğer X-RAY’lerine bağlı olarak hangi evrede olduğunu öğrenmeye çalışmak.
A ML Solution
Öncelikle ne kadar hastalarımızın COVİD-19 olduğunu bilerek X-RAY görüntülerini alarak hangi evrede olduğunu söylemiş olsakta, 
yapay zekamız COVİD-19 olmadığını düşündüğü hastalara COVİD-19 değil tahmini yapmakta. 
İkiden fazla seçenek olduğu için “classification” algoritmasının “multi-class” kullanılarak kategorik değerlendirmesi yapılmıştır.


Data Curation 
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia


Model Training
Modelimizi COVİD-19’ un karşılaştırması ile durumun ilerlemiş ve gerilemiş olarak 
eğtilmesini “Supervised Learning” yaparak modelimiz hastamızın durumunu ve hastalığın hangi bölgelere entegre ettiğini göstermiş olacak.

Evaluation Metrics
TP (Doğru - Pozitif): Covid covid olması.
FP (Yanlış - Pozitif): Covid olmayana covid demek.
TN (Doğru - Negatif): Covid olmayana Covid değil demek. 
FN (Yanlış - Negatif): Covid olana Covid değil demek
