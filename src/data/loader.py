import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_data(args:dict) -> pd.DataFrame:
    # path:str, target:str, sort_key:str, arm_key, select_cols:list, category_cols:list, drop_cols:list
    df = pd.read_csv(args["path"]).sort_values(args["sort_key"])
    fetures, encoder = transform(df, args)
    return fetures, df[args["target"]]*2-1, df[args["arm_key"]], encoder


def transform(data:pd.DataFrame, args:dict) -> pd.DataFrame:
    data = data[data[args["is_app"]]]
    data = data[data.ssp_id!=5335]
    data = data[args["select_cols"]]
    data[args["category_cols"]] = data[args["category_cols"]].astype(str)
    data = data.drop(args["drop_cols"] + [args["target"], args["arm_key"]], axis=1)

    if args["encoder"] is None:
        enc = OneHotEncoder(handle_unknown='ignore')
        encoder = enc.fit(data)
    else:
        encoder = args["encoder"]
    fetures = encoder.transform(data) # .toarray()
    return fetures, encoder