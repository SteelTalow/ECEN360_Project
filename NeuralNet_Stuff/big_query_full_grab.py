#Import bigquery stuff
from sqlite3 import connect
from google.cloud import bigquery
from google.oauth2 import service_account

import pandas as pd

#define stuff for our project
BIGQUERY_KEY_FILE = r"nycrest-20ce24cbc0f6.json"  # Replace with your Service Account JSON key file that you just downloaded
PROJECT_ID = "nycrest"  # Replace with your GCP Project ID
DATASET_ID = "rest_data"

#Verify the connection
key_path = r"nycrest-20ce24cbc0f6.json"

try:
    # Authenticate using the Service Account
    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    print(f"🎉 Successfully authenticated! Project ID: {credentials.project_id}")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
    print("👉 Troubleshooting Tips:")
    print("- Ensure the key file path is correct.")
    print("- Verify the Service Account has the BigQuery Admin role.")
    print("- Confirm the BigQuery API is enabled for your project.")


#Now let's try to get the data
credentials = service_account.Credentials.from_service_account_file(BIGQUERY_KEY_FILE)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

project_id = "nycrest"

query = """
    SELECT *
    FROM `nycrest.rest_data.filtered_data`
    
"""

#print it out
data = client.query_and_wait(query).to_dataframe()

print(data)

#Push to csv
data.to_csv('example.csv', index = False)