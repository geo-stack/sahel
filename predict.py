# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 16:16:37 2025

@author: User
"""

with open("./models/model.pkl", "rb") as file:
    model = pickle.load(file)

new_res = res.dropna()
X = new_res[
    [
        "alt_stream",
        "dist_stream",
        "alt_top",
        "dist_top",
        "ratio_alt",
        "ratio_dist",
        "ratio_stream",
        "ratio_top",
        "altitude",
        "accumulation",
    ]
].astype("float")
X["dist_stream_inverse"] = 1 / (X["dist_stream"] + 1)
X = X.drop(columns=["dist_stream"])

Y = model.predict(X)

new_res["Predicted_NS"] = Y

new_res.to_csv("./data/results/temp/" + f"map_{country}_{index}.csv")

if f"map_{country}.csv" in map_csv_files:
    continue
fichiers_csv = glob.glob(f"{folder}/*{country}*.csv")

dfs = [pd.read_csv(fichier, index_col=0) for fichier in fichiers_csv]
merged_df = pd.concat(dfs, ignore_index=True)
merged_df = merged_df[["LON", "LAT", "Predicted_NS"]]

merged_df.to_csv("./data/results/" + f"map_{country}.csv")
