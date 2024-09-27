import requests
from bs4 import BeautifulSoup
import csv

# URL of the page to scrape
url = 'https://cse.umn.edu/dsi/medicalhealth-technology'

# Send an HTTP request to the page and fetch the content
response = requests.get(url)

# Parse the content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Initialize lists to store the extracted names and emails
names = []
emails = []

# Find all the relevant sections of the page that contain the person information
people_sections = soup.find_all('div', class_='pl-col-two')

for section in people_sections:
    # Find the person's name
    name_tag = section.find('div', class_='pl-item people-title')
    if name_tag:
        name = name_tag.get_text(strip=True)
        names.append(name)
    
    # Find the person's email
    email_tag = section.find('div', class_='pl-item people-email')
    if email_tag:
        email = email_tag.get_text(strip=True)
        emails.append(email)

# Output file name
output_file = "(Initial) AID-H Working Group Mailing List Group2.csv"

# Write the names and emails to the CSV file
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Email'])  # Write the header
    writer.writerows(zip(names, emails))  # Write the data rows

print(f"Data successfully saved to {output_file}")
