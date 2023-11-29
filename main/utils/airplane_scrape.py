from bs4 import BeautifulSoup
from selenium import webdriver
import time
import pandas as pd
import glob


def read_csv_from_subfolder(path):
    if path:
        all_files = glob.glob(path)
        li = []
        for filename in all_files:
            df = pd.read_csv(filename)
            li.append(df)
        return pd.concat(li, axis=0, ignore_index=True)
    else:
        return None


flights_na = read_csv_from_subfolder("../data_raw/US_Domestic_Flights/*/*.csv")
atl_flights = flights_na[flights_na['DEST'] == 'ATL']

save = 100
cut = 3900
sleep = 3
tails = atl_flights['TAIL_NUM'].value_counts().keys().tolist()[cut:]

table = []
cnt = cut
for tail in tails:
    driver = webdriver.Chrome("chromedriver.exe")
    try:
        driver.get("https://www.airfleets.net/recherche/?key=" + tail)
        time.sleep(sleep)
        print("Progress: " + str(cnt + 1) + " / " + str(len(atl_flights['TAIL_NUM'].value_counts().keys().tolist())))
        content = driver.page_source
        soup = BeautifulSoup(content)
        for row in soup.find('p', {'class': 'soustitre'}).find_next_sibling().findAll('tr')[1:]:
            if row.text.__contains__(tail):
                table.append(row.text.split("\n")[1:-1])
            else:
                print("Tailnumber " + tail + " not found!")
    except AttributeError as ae:
        print("Attribute Exception: "+ ae + " \nSkipping entry " + tail + "!")
    except:
        print("General Exception! Skipping entry " + tail + "!")
    driver.close()
    if (cnt + 1) % save == 0:
        filename = '../data/scraped_aircraft/ac_' + str(int((cnt + 1) / save)) + '.pkl'
        print("Saving " + filename)
        pd.DataFrame(table, columns=['Aircraft', 'Regist.', 'MSN', 'Airline', 'Status']).to_pickle(filename)
        table = []
    cnt = cnt + 1
pd.DataFrame(table, columns=['Aircraft', 'Regist.', 'MSN', 'Airline', 'Status']).to_pickle('../data/scraped_flights/ac_final.pkl')
