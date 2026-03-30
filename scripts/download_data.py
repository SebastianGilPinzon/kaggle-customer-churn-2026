"""Download competition + original data for GitHub Actions."""
import os, zipfile, io, requests
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFilesRequest
from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest

os.makedirs('data-kaggle', exist_ok=True)
os.makedirs('data-original', exist_ok=True)
client = KaggleClient()

# Competition data
req = ApiDownloadDataFilesRequest()
req.competition_name = 'playground-series-s6e3'
resp = client.competitions.competition_api_client.download_data_files(req)
r = requests.get(resp.url)
with zipfile.ZipFile(io.BytesIO(r.content)) as z:
    z.extractall('data-kaggle')
print('Competition data:', os.listdir('data-kaggle'))

# Original dataset
req2 = ApiDownloadDatasetRequest()
req2.owner_slug = 'blastchar'
req2.dataset_slug = 'telco-customer-churn'
resp2 = client.datasets.dataset_api_client.download_dataset(req2)
r2 = requests.get(resp2.url)
with zipfile.ZipFile(io.BytesIO(r2.content)) as z2:
    z2.extractall('data-original')
print('Original data:', os.listdir('data-original'))
