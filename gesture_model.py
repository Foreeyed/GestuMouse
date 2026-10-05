import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('gestures_dataset.csv')


X = df.drop(columns=['class'])
y = df['class']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = make_pipeline(
    #StandardScaler(), Scaler нужен если использовать алгоритмы неустойчивые к данным.
    # В моем случае есть 2 причины по которым не нужно использовать scaler
    # 1. Я уже масштабирую данные для каждого значения в handMarker.py в пределах (-1:1), а центром является точка на кисти с координатами (0, 0, 0)
    # 2. Алгоритм "RandomForestClassifier" является очень устойчивым к метрикам
    # Но при использовании других алгоритмов классификации желательно использовать scaler
    RandomForestClassifier(random_state=42, class_weight='balanced')
)


pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

result = cross_val_score(pipeline, X, y, cv=skf)

report = classification_report(y_test, predictions)

print(report)
print(result)