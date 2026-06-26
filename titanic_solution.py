import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from competition_ai.core.architecture import CompetitorArchitecture

def main():
    arch = CompetitorArchitecture()
    tabular = arch.skills['tabular']

    train = pd.read_csv('titanic_data/train.csv')
    test = pd.read_csv('titanic_data/test.csv')
    df = pd.concat([train, test], sort=False).reset_index(drop=True)

    # 1. Advanced Heuristics & Grouping using Architecture Skills
    df["Surname"] = df["Name"].apply(lambda x: x.split(",")[0].strip())
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    df["IsBoy"] = (df["Title"] == "Master").astype(int)
    df["IsWC"] = ((df["Sex"] == "female") | (df["IsBoy"] == 1)).astype(int)
    df["IsAdultMale"] = ((df["Sex"] == "male") & (df["IsBoy"] == 0)).astype(int)

    # Union-Find GID
    df["gid"] = tabular.union_find_groups(df, ["Surname", "Ticket"]).values
    df["gid_tic"] = tabular.union_find_groups(df, ["Ticket"]).values
    df["gid_sur"] = tabular.union_find_groups(df, ["Surname"]).values

    # Group survival rates
    df["wcg_rate"] = tabular.loo_group_rate(df, "gid", "Survived", "IsWC").values
    mt = tabular.loo_group_rate(df, "gid_tic", "Survived", "IsAdultMale")
    ms = tabular.loo_group_rate(df, "gid_sur", "Survived", "IsAdultMale")
    df["male_rate"] = mt.where(mt.notna(), ms).values

    # Family Survival
    df["Family_Survival"] = 0.5
    for grp, grp_df in df.groupby(["Surname", "Fare"]):
        if len(grp_df) > 1:
            for ind, row in grp_df.iterrows():
                smax = grp_df.drop(ind)["Survived"].max()
                smin = grp_df.drop(ind)["Survived"].min()
                if smax == 1.0: df.loc[ind, "Family_Survival"] = 1
                elif smin == 0.0: df.loc[ind, "Family_Survival"] = 0
    for grp, grp_df in df.groupby("Ticket"):
        if len(grp_df) > 1:
            for ind, row in grp_df.iterrows():
                if df.loc[ind, "Family_Survival"] == 0.5:
                    smax = grp_df.drop(ind)["Survived"].max()
                    smin = grp_df.drop(ind)["Survived"].min()
                    if smax == 1.0: df.loc[ind, "Family_Survival"] = 1
                    elif smin == 0.0: df.loc[ind, "Family_Survival"] = 0

    # 2. Traditional FE
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Fare_Bin"] = pd.qcut(df["Fare"], 5, labels=False)
    df["Age"] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    df["Age_Bin"] = pd.cut(df["Age"].astype(int), 5, labels=False)
    df["SexNum"] = df["Sex"].map({"male": 0, "female": 1})
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    df = tabular.label_encode(df, ['Sex', 'Embarked', 'Title', 'Deck'])

    # 3. Model Ensembling
    FEATS = ["Pclass", "SexNum", "Fare_Bin", "Age_Bin", "Family_Survival", "FamilySize", "wcg_rate", "male_rate"]
    df_ml = df.copy()
    df_ml["wcg_rate"] = df_ml["wcg_rate"].fillna(0.5)
    df_ml["male_rate"] = df_ml["male_rate"].fillna(0.5)

    is_train = df_ml["Survived"].notna()
    X_train = df_ml.loc[is_train, FEATS]
    y_train = df_ml.loc[is_train, "Survived"].astype(int)
    X_test = df_ml.loc[~is_train, FEATS]

    knn5 = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    svc1 = make_pipeline(StandardScaler(), SVC(C=1, gamma="auto"))
    rf = RandomForestClassifier(n_estimators=500, max_depth=6, random_state=42)

    knn5.fit(X_train, y_train)
    svc1.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    p_knn = knn5.predict(X_test)
    p_svc = svc1.predict(X_test)
    p_rf = rf.predict(X_test)

    # 4. Composite Decision Logic
    te = df[~is_train].reset_index(drop=True)
    sex, boy = te["SexNum"].values, te["IsBoy"].values
    pcl = te["Pclass"].values
    rate, mrate = te["wcg_rate"].values, te["male_rate"].values
    noinfo = np.isnan(rate)

    preds = np.where(sex == 1, 1, 0)
    preds[boy == 1] = 0
    preds[(boy == 1) & (rate == 1.0)] = 1

    known_w = (sex == 1) & ~noinfo & (rate > 0)
    preds[known_w] = p_rf[known_w]
    preds[(sex == 1) & (rate == 0.0)] = 0

    noi_w = (sex == 1) & noinfo
    preds[noi_w] = p_svc[noi_w]
    preds[noi_w & (pcl == 3) & (mrate == 0.0)] = 0
    preds[(sex == 0) & (boy == 0) & (pcl == 1) & (mrate == 1.0)] = 1

    submission = pd.DataFrame({"PassengerId": te["PassengerId"], "Survived": preds.astype(int)})
    submission.to_csv("submission.csv", index=False)
    print("[*] Local validation and submission generation complete.")

if __name__ == "__main__":
    main()
