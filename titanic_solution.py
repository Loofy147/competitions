import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from competition_ai.core.architecture import CompetitorArchitecture

def main():
    arch = CompetitorArchitecture()
    tabular = arch.skills['tabular']
    val_skill = arch.skills['val']

    # Load data
    train = pd.read_csv('titanic_data/train.csv')
    test = pd.read_csv('titanic_data/test.csv')
    all_data = pd.concat([train, test], axis=0).reset_index(drop=True)

    # Heuristics
    all_data["Surname"] = all_data["Name"].apply(lambda x: x.split(",")[0].strip())
    all_data["Family_Survival"] = 0.5
    for grp, grp_df in all_data.groupby(["Surname", "Fare"]):
        if len(grp_df) > 1:
            for ind, row in grp_df.iterrows():
                smax = grp_df.drop(ind)["Survived"].max()
                smin = grp_df.drop(ind)["Survived"].min()
                if smax == 1.0: all_data.loc[ind, "Family_Survival"] = 1
                elif smin == 0.0: all_data.loc[ind, "Family_Survival"] = 0
    for grp, grp_df in all_data.groupby("Ticket"):
        if len(grp_df) > 1:
            for ind, row in grp_df.iterrows():
                if all_data.loc[ind, "Family_Survival"] == 0.5:
                    smax = grp_df.drop(ind)["Survived"].max()
                    smin = grp_df.drop(ind)["Survived"].min()
                    if smax == 1.0: all_data.loc[ind, "Family_Survival"] = 1
                    elif smin == 0.0: all_data.loc[ind, "Family_Survival"] = 0

    all_data["Title"] = all_data["Name"].str.extract(r",\s*([^.]+)\.")[0].str.strip()
    all_data["IsBoy"] = (all_data["Title"] == "Master").astype(int)
    all_data["IsWC"] = ((all_data["Sex"] == "female") | (all_data["IsBoy"] == 1)).astype(int)

    all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    all_data['Title'] = all_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
    all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')
    all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
    all_data['Deck'] = all_data['Cabin'].str[0].fillna('U')
    all_data['TicketFreq'] = all_data.groupby('Ticket')['Ticket'].transform('count')
    all_data['Age'] = all_data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
    all_data['Sex_Pclass'] = all_data['Sex'] + "_" + all_data['Pclass'].astype(str)

    all_data = tabular.simple_impute(all_data)
    all_data = tabular.label_encode(all_data, ['Sex', 'Embarked', 'Title', 'Deck', 'Sex_Pclass'])

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                'Family_Survival', 'IsWC', 'FamilySize', 'Deck', 'TicketFreq', 'Title', 'Sex_Pclass']

    X = all_data[:len(train)][features].values
    y = train['Survived'].values
    X_test = all_data[len(train):][features].values

    # Best Model Configuration based on 0.80143 submission
    rf = RandomForestClassifier(n_estimators=1000, max_depth=6, random_state=42).fit(X, y)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42).fit(X, y)

    probs = (rf.predict_proba(X_test)[:, 1] + gb.predict_proba(X_test)[:, 1]) / 2
    preds = (probs > 0.5).astype(int)

    submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': preds})
    submission.to_csv('submission.csv', index=False)
    print("[*] Reverted to stable 0.80143 logic.")

if __name__ == "__main__":
    main()
