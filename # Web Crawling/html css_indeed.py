# scraping from https://indeed.com -> title, company, location and reviews of all pages
# start from this url https://kr.indeed.com/?from=gnav-homepage

import requests # internet에서 file을 가져오는 라이브러리
from bs4 import BeautifulSoup

# pratice 1 from here

url = 'https://kr.indeed.com/jobs?q=data+science&l=%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C'
link = requests.get(url)
soup = BeautifulSoup(link.text, 'html.parser')

job_elems = soup.select('.resultContent')  # class

for i in job_elems:
    title = i.find('h2')
    company = i.find('span', class_='companyName')
    location = i.find('div', class_='companyLocation')

    if None in (title, company, location):
        continue

    print(title.text.strip())
    print(company.text.strip())
    print(location.text.strip())

# practice 2 from here