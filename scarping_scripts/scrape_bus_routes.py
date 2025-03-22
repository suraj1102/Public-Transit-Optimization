import json
import time
import csv
import urllib.parse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


chrome_options = Options()
chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 10)

driver.get("https://ctucitybus.com/#/ViewLocations")

dropdown_arrow = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "k-icon.k-i-arrow-s")))
dropdown_arrow.click()

# Wait for the dropdown options to become visible
wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "k-animation-container")))

# Locate the dropdown options container
dropdown_container = driver.find_element(By.CLASS_NAME, "k-animation-container")
options = dropdown_container.find_elements(By.TAG_NAME, "li")

# Print out the text of each option
for option in options:
    print(f"Working on: {option.text}")
    option.click()


