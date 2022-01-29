import requests
from bs4 import BeautifulSoup

# get company info - Recruit subject, Name, Location

pages = [
'https://kr.indeed.com/data-scientist%EC%A7%81-%EC%B7%A8%EC%97%85-%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C-%EC%A7%80%EC%97%AD',
'https://kr.indeed.com/jobs?q=data+scientist&l=%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C&start=10',
'https://kr.indeed.com/jobs?q=data+scientist&l=%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C&start=20',
'https://kr.indeed.com/jobs?q=data+scientist&l=%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C&start=30']

for pIdx, page in enumerate(pages, 1):
  url = page
  link = requests.get(url)
  soup = BeautifulSoup(link.text, 'html.parser')
  elems = soup.select('.resultContent')
  print(f"Company Number {len(elems)} of Page {pIdx}")
  for elem in elems:
    Jobtitle = elem.find('h2' , class_= 'jobTitle').text.strip()
    CompanyName = elem.find('span' , class_= 'companyName').text.strip()
    CompanyLocation = elem.find('div' , class_= 'companyLocation').text.strip()

    if None in (Jobtitle, CompanyName, CompanyLocation):
        print()
        print("# -------------- FOund Missing data !! ------------- #")
        print()
        continue

    print(f"Page : {pIdx} ")
    print(f"Jobtitle: {Jobtitle} , CompanyName : {CompanyName}, CompanyLocation : {CompanyLocation}")
  print()

# get company reviews
url = 'https://kr.indeed.com/cmp/Lg-Electronics/reviews' # reviewers link
link = requests.get(url)
soup = BeautifulSoup(link.text, 'html.parser')
titles = soup.find_all('div' , 'css-r0sr81 e37uo190')
reviews = soup.find_all('span', 'css-1cxc9zk e1wnkr790')
reviews = reviews[1:]

for t, r in zip(titles, reviews):
    title = t.find('span', class_='css-82l4gy eu4oa1w0')
    review = r.find_all('span', 'css-82l4gy eu4oa1w0')

    if None in (title, review):
        print("# ----------------------------------- #")
        print("# ----- Missing data is found ------- # ")
        print("# ----------------------------------- #")
        continue

    print("Title : ", title.text.strip())
    print()
    print("Review :", end='\n\n')
    for k in review:
        print(k.getText())

    print("# -------------- #")
    print()
    print()
