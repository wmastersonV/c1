import pandas as pd
from pandas_profiling import ProfileReport

dat = pd.read_json("./transactions.txt", lines=True)
# pandas profiling
profile = ProfileReport(dat, title="Pandas Profiling Report")
profile.to_file("./transactions_output.html")