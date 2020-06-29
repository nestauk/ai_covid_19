import urllib.request
import tarfile
from datetime import datetime,timedelta
import ai_covid_19
import os
project_dir = ai_covid_19.project_dir
target_path = f"{project_dir}/data/raw/cord_19"


#Create the target directory if it didn't exist
if os.path.exists(target_path)==False:
    os.mkdir(target_path)

#Get yesterday's date
yesterday = str((datetime.today() - timedelta(1)).date())

#File name
file = f"https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_{yesterday}.tar.gz"

#Note: this will overwrite the previous file
url = urllib.request.urlopen(file)
tarf_= tarfile.open(fileobj=url, mode="r|gz")
tarf_.extractall(target_path)