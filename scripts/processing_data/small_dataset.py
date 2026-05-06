import pandas as pd 



csvpath = "/home/davido/Projects/data/QMS9/csv/qm9s_csv/smile_uv.csv"
csv = pd.read_csv(csvpath) 

csv = csv.iloc[5000:]
print(len(csv)) 

csv.to_csv("/home/davido/Projects/ML/spectra_ml/data/testset_5000.csv")



