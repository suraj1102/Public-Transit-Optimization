from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Initialize the WebDriver (e.g., using Chrome)
driver = webdriver.Chrome()

try:
    # Navigate to the webpage
    driver.get("https://ctuportal.amnex.com/#/ViewRoutes")

    # Wait for the page to load and the dropdown arrow to be clickable
    wait = WebDriverWait(driver, 10)
    dropdown_arrow = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "k-icon.k-i-arrow-s")))

    # Click the dropdown arrow to reveal the options
    dropdown_arrow.click()

    # Wait for the dropdown options to become visible
    wait.until(EC.visibility_of_element_located((By.CLASS_NAME, "k-animation-container")))

    # Locate the dropdown options container
    dropdown_container = driver.find_element(By.CLASS_NAME, "k-animation-container")

    # Retrieve all option elements within the container
    options = dropdown_container.find_elements(By.TAG_NAME, "li")

    # Print out the text of each option
    for option in options:
        print(option.text)

finally:
    # Close the WebDriver
    driver.quit()
