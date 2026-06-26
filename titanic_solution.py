import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from competition_ai.core.architecture import CompetitorArchitecture

def main():
    arch = CompetitorArchitecture()
    tabular = arch.skills['tabular']
    val_skill = arch.skills['val']
    monitoring = arch.skills['monitoring']

    # Load data
    train = pd.read_csv('titanic_data/train.csv')
    test = pd.read_csv('titanic_data/test.csv')

    # Pre-merge for global feature engineering
    all_data = pd.concat([train, test], axis=0).reset_index(drop=True)

    print(f"[*] Loaded data. Total records: {all_data.shape[0]}")

    # 1. Family_Survival Heuristic (from top kernels)
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

    # 2. Woman-Child-Group (WCG) logic
    all_data["Title"] = all_data["Name"].str.extract(r",\s*([^.]+)\.")[0].str.strip()
    all_data["IsBoy"] = (all_data["Title"] == "Master").astype(int)
    all_data["IsWomanChild"] = ((all_data["Sex"] == "female") | (all_data["IsBoy"] == 1)).astype(int)

    # 3. Traditional Feature Engineering
    def engineer_features(df):
        # Title groups
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

        # Family Size
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

        # Deck
        df['Deck'] = df['Cabin'].str[0].fillna('U')

        # Ticket Frequency
        df['TicketFreq'] = df.groupby('Ticket')['Ticket'].transform('count')

        # Age imputation by Title
        df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

        # Interactions
        df['Sex_Pclass'] = df['Sex'] + "_" + df['Pclass'].astype(str)

        return df

    all_data = engineer_features(all_data)

    # Impute and Encode
    all_data = tabular.simple_impute(all_data)
    all_data = tabular.label_encode(all_data, ['Sex', 'Embarked', 'Title', 'Deck', 'Sex_Pclass'])

    train_eng = all_data[:len(train)].copy()
    test_eng = all_data[len(train):].copy()

    # Features to use for ML model
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
                'Family_Survival', 'IsWomanChild', 'FamilySize', 'Deck', 'TicketFreq', 'Title', 'Sex_Pclass']

    X = train_eng[features].values
    y = train['Survived'].values
    X_test = test_eng[features].values

    print(f"[*] Preprocessing complete. X: {X.shape}")

    # Ensemble Models
    def rf_factory(): return RandomForestClassifier(n_estimators=1000, max_depth=6, random_state=42)
    def gb_factory(): return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)

    print("[*] Running CV for Ensemble...")
    oof_rf, score_rf = val_skill.run_stratified_cv(X, y, rf_factory)
    oof_gb, score_gb = val_skill.run_stratified_cv(X, y, gb_factory)

    monitoring.log_improvement(0.3931, (score_rf + score_gb)/2, "Mean CV Log Loss (WCG Integrated)")

    # Final Model Training
    rf = rf_factory().fit(X, y)
    gb = gb_factory().fit(X, y)

    probs = (rf.predict_proba(X_test)[:, 1] + gb.predict_proba(X_test)[:, 1]) / 2
    preds = (probs > 0.5).astype(int)

    # --- WCG Override (The "Magic" part) ---
    # According to top kernels, if a Woman/Child is part of a group where everyone died, they are likely to die.
    # If a Woman/Child is part of a group where someone survived, they are likely to survive.
    # Our 'Family_Survival' feature already captures this for the model,
    # but some kernels do hard overrides. We'll stick to the model-based approach
    # as it's more robust to CV, but we've added the critical 'Family_Survival' feature.

    # Prepare submission
    submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': preds})
    submission.to_csv('submission.csv', index=False)

    # Submit
    print(arch.submit_solution('titanic', 'submission.csv', 'Full Architecture + WCG Logic Ensemble'))

    # Check status
    print(monitoring.get_status('titanic')['raw_status'][:300])

if __name__ == "__main__":
    main()
