import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from competition_ai.core.architecture import CompetitorArchitecture

def main():
    arch = CompetitorArchitecture()
    tabular = arch.skills['tabular']
    val_skill = arch.skills['val']
    monitoring = arch.skills['monitoring']

    # Load data
    train = pd.read_csv('titanic_data/train.csv')
    test = pd.read_csv('titanic_data/test.csv')

    all_data = pd.concat([train, test], axis=0).reset_index(drop=True)

    print(f"[*] Loaded data. Total records: {all_data.shape[0]}")

    # Advanced Feature Engineering (Merging Architecture logic + TF-DF kernel logic)
    def engineer_features(df):
        # 1. Title (Architecture logic)
        df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

        # 2. Name Normalization & Tokenization concepts (TF-DF kernel)
        df['Name_Len'] = df['Name'].apply(len)

        # 3. Ticket Processing (TF-DF kernel)
        df['Ticket_number'] = df['Ticket'].apply(lambda x: x.split(" ")[-1])
        df['Ticket_item'] = df['Ticket'].apply(lambda x: "_".join(x.split(" ")[:-1]) if len(x.split(" ")) > 1 else "NONE")

        # 4. Family Size & IsAlone (Architecture logic)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = 0
        df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

        # 5. Deck (Architecture logic)
        df['Deck'] = df['Cabin'].str[0]
        df['Deck'] = df['Deck'].fillna('U')

        # 6. Ticket Frequency (Architecture logic)
        ticket_counts = df.groupby('Ticket')['Ticket'].transform('count')
        df['TicketFreq'] = ticket_counts

        # 7. Fare Bins (Architecture logic)
        df['FareBin'] = pd.qcut(df['Fare'].fillna(df['Fare'].median()), 4, labels=False)

        # 8. Age Bins (Architecture logic)
        df['Age'] = df.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))
        df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 100], labels=False)

        # 9. Sex & Pclass Interaction (Architecture logic)
        df['Sex_Pclass'] = df['Sex'] + "_" + df['Pclass'].astype(str)

        # Drop raw columns
        cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
        df = df.drop(cols_to_drop, axis=1)

        return df

    all_data_eng = engineer_features(all_data)

    # Impute and Encode
    all_data_eng = tabular.simple_impute(all_data_eng)
    all_data_eng = tabular.label_encode(all_data_eng, ['Sex', 'Embarked', 'Title', 'Deck', 'Sex_Pclass', 'Ticket_item', 'Ticket_number'])

    train_eng = all_data_eng[:len(train)].copy()
    test_eng = all_data_eng[len(train):].copy()

    X = train_eng.drop('Survived', axis=1).values
    y = train['Survived'].values
    X_test = test_eng.drop('Survived', axis=1).values

    print(f"[*] Preprocessing complete. X: {X.shape}, X_test: {X_test.shape}")

    # Model Ensemble
    def rf_factory(): return RandomForestClassifier(n_estimators=1000, max_depth=6, min_samples_leaf=2, random_state=42)
    def gb_factory(): return GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)

    print("[*] Running CV for Random Forest...")
    oof_rf, score_rf = val_skill.run_stratified_cv(X, y, rf_factory)

    print("[*] Running CV for Gradient Boosting...")
    oof_gb, score_gb = val_skill.run_stratified_cv(X, y, gb_factory)

    mean_cv_score = (score_rf + score_gb) / 2
    print(f"[*] RF Score: {score_rf:.4f}, GB Score: {score_gb:.4f}")

    # Monitoring: Log improvement
    monitoring.log_improvement(0.3921, mean_cv_score, "Mean CV Log Loss (Ensemble + TF-DF features)")

    # Train and Predict
    rf = rf_factory().fit(X, y)
    gb = gb_factory().fit(X, y)

    test_preds_rf = rf.predict_proba(X_test)[:, 1]
    test_preds_gb = gb.predict_proba(X_test)[:, 1]

    final_probs = (test_preds_rf + test_preds_gb) / 2
    final_preds = (final_probs > 0.5).astype(int)

    # Prepare submission
    submission = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': final_preds
    })
    submission_path = 'submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"[*] Submission file saved to {submission_path}")

    # Submit with monitoring
    print("[*] Submitting to Kaggle...")
    print(arch.submit_solution('titanic', submission_path, 'Architecture + TF-DF Kernel features ensemble with monitoring'))

    # Check status
    print("[*] Fetching competition status...")
    status = monitoring.get_status('titanic')
    print(status['raw_status'][:500])

if __name__ == "__main__":
    main()
