import urllib.request
url = "https://arxiv.org/pdf/2602.12375v1"
urllib.request.urlretrieve(url, "test_dl.pdf")
import os
print("Size:", os.path.getsize("test_dl.pdf"))
