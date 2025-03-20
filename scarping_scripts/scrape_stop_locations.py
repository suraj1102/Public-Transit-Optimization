import json
import time
import csv
import urllib.parse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_required_info(logs):
    results = []
    for entry in logs:
        try:
            message = json.loads(entry['message'])['message']
            if message.get('method') == 'Network.responseReceived':
                response = message.get('params', {}).get('response', {})
                url = response.get('url', '')
                if 'GetNeaybyStops' in url:
                    # Fetch the request ID to get the response body
                    request_id = message.get('params', {}).get('requestId', '')
                    # Execute CDP command to get response body
                    response_body = driver.execute_cdp_cmd('Network.getResponseBody', {'requestId': request_id})
                    results.append({
                        'url': url,
                        'status': response.get('status', ''),
                        'headers': response.get('headers', {}),
                        'body': response_body
                    })
        except Exception as e:
            print(f"Error processing entry: {e}")
            continue
    return results



def click_dropdown():
    try:
        # 2. Click the "Edit Your Location" element.
        edit_button = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//*[contains(@title, 'Edit Your Location') and contains(@class, 'material-icons')]")
        ))
        edit_button.click()
        time.sleep(2)

        # 3. Click the dropdown arrow to open the dropdown.
        dropdown_arrow_xpath = "//span[contains(@class, 'k-select') and @unselectable='on']"
        wait.until(EC.element_to_be_clickable((By.XPATH, dropdown_arrow_xpath)))
        driver.find_element(By.XPATH, dropdown_arrow_xpath).click()
        time.sleep(2)

        # 4. Find dropdown entries. (Assuming entries are clickable elements inside the popup.)
        popup_xpath = "//kendo-popup[contains(@class, 'k-animation-container-shown')]"
        popup = wait.until(EC.visibility_of_element_located((By.XPATH, popup_xpath)))
        # Adjust the entry XPath based on actual structure. Here we assume each entry is a <li> element.
        entry_xpath = ".//li"
        entries = popup.find_elements(By.XPATH, entry_xpath)
        return entries
    except Exception:
        print("------------ ERROR BYEEE -----------")
        return None


def scrape():
    # Configure ChromeOptions to capture performance logs.
    chrome_options = Options()
    chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    driver = webdriver.Chrome(options=chrome_options)
    wait = WebDriverWait(driver, 10)

    try:
        driver.get("https://ctucitybus.com/#/ViewLocations")
        time.sleep(3)  # Wait for page to load completely
        
        num_entries = len(click_dropdown())
        print(f"Found {num_entries} dropdown entries.")
        main_entries = click_dropdown()

        if main_entries == None:
            quit()

        for i in range(179,180):
            entries = click_dropdown()
            
            try:
                current_entry = entries[i]
                entry_text = current_entry.text.strip()
            except:
                print(f"----------- ERROR AT i = {i} ----------")
                with open("scarping_scripts/scarped_data/raw_request.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([i, "Error"])
                continue

            print(f"Clicking entry {i}: {entry_text}")

            current_entry.click()
            time.sleep(1)

            logs = driver.get_log("performance")
            res = get_required_info(logs)
            
            with open("scarping_scripts/scarped_data/raw_request.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([entry_text, res])

    finally:
        driver.quit()


import ast
import pandas as pd

def extract_clean_data(request_entry):
    try:
        # Convert string representation of list to actual list
        request_list = ast.literal_eval(request_entry)
        
        # Extract the body field containing JSON data
        body_content = request_list[0]["body"]["body"]
        
        # Convert to a JSON object
        body_dict = json.loads(body_content)
        
        # Extract and parse the "data" field
        clean_data = json.loads(body_dict["data"])
        
        return clean_data
    
    except (SyntaxError, ValueError, KeyError, IndexError):
        return None



def is_valid_bus_stop(cleaned_data):
    if not cleaned_data:
        return None
    
    if float(cleaned_data[0]['distance']) <= 0.0:
        return True
    
    return None

def nearest_distance(cleaned_data):
    try:
        dist = float(cleaned_data[0]['distance'])
        return dist
    except (SyntaxError, ValueError, KeyError, IndexError):
        return None

def main():
    # scrape()
    # Load the CSV file
    file_path = "scarping_scripts/scarped_data/raw_request.csv"
    df = pd.read_csv(file_path)

    # Apply extraction function to each row
    df["Cleaned_Data"] = df["Request"].apply(extract_clean_data)
    # print(df.head())

    # Display cleaned data for first few rows
    # entry = df["Cleaned_Data"].iloc[1]
    # print(entry[0])
    # print(df["Cleaned_Data"].isnull().sum())

    df["Is_Valid_Stop"] = df["Cleaned_Data"].apply(is_valid_bus_stop)
    # print(df["Is_Valid_Stop"].isnull().sum())

    df["Nearest_Distance"] = df["Cleaned_Data"].apply(nearest_distance)
    print(df[df["Name", "Nearest_Distance"]])


if __name__ == "__main__":
    main()