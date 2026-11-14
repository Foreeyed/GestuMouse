"""Обучение классификатора жестов и сохранение модели в model.pkl."""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

from gestures import GESTURES

DATASET_PATH = "gestures_dataset.csv"
MODEL_PATH = "model.pkl"


def main():
    df = pd.read_csv(DATASET_PATH)

    X = df.drop(columns=["class"])
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaler не нужен:
    # 1. Признаки уже нормализованы в пределах (-1, 1) относительно запястья.
    # 2. RandomForest устойчив к масштабу признаков.
    model = make_pipeline(
        RandomForestClassifier(random_state=42, class_weight="balanced")
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    labels = sorted(y.unique())
    target_names = [GESTURES.get(label, str(label)) for label in labels]
    print(classification_report(y_test, predictions, labels=labels, target_names=target_names))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf)
    print(f"Cross-validation accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    joblib.dump(model, MODEL_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")


if __name__ == "__main__":
    main()
