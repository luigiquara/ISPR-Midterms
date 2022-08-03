import csv
from bs4 import BeautifulSoup
import requests
from datetime import datetime as time

start = time.now()
print('elapsed:' + str(time.now() - start))


#Snippet for scraping the latest headlines from Lercio.it
base_URL = 'https://www.lercio.it/category/brevi/page/'

for page_count in range(1,73):
    page = requests.get(base_URL + str(page_count))
    soup = BeautifulSoup(page.content, 'html.parser')

    for div in soup.find_all('div', {'class': 'jeg_sidebar'}):
        div.decompose()

    articles = soup.find_all('article')

    with open('headlines/titles_ultima.csv', 'a') as file:
        writer = csv.writer(file, delimiter=',')

        # get the title from the two highlighted news only once
        if page_count == 1:
            writer.writerow([articles[0].find('h2').a.text])
            writer.writerow([articles[1].find('h2').a.text])

        for element in articles[2:]:
            title = element.find('h3')
            headline = title.a.text
            writer.writerow([headline])





!python train.py --data_file=dataset.txt --hidden_size=hs
                 --num_layers=nl --model='lstm'
                 --dropout=d --batch_size=bs

#Generation of new headlines
!python sample.py --init_dir=output --start_text=st
                  --length=l


#Implementation used
https://github.com/crazydonkey200/tensorflow-char-rnn

#Parameters grid for model selection
{
    "hidden_size" : [128,256,512],
    "num_layers" : [2,3],
    "dropout" : [0.3,0.5],
    "batch_size" : [32,64,128]
    "num_unrollings" : [10,50] #don't know if to keep also this line, it will depend on the results
}