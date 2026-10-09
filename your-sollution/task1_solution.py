#!/usr/bin/env python3
"""Task 1: tabular supervised classification.

Скрипт следует 10 шагам из лекции 3:
    01. Скачать данные из репозитория
    02. Загрузить CSV в pandas
    03. Убрать выбросы (IQR или z-score)
    04. Заполнить пропуски средним
    05. Нормализовать через StandardScaler
    06. Разделить на train / val / test
    07. Обучить модель (1 слой, 32 нейрона, ReLU, 150 эпох)
    08. Сгенерировать предсказания для test_features.csv
    09. Сохранить в outputs/task1_predictions.csv
    10. Сдать код и описание

Читает:
    dataset/task1/train.csv
    dataset/task1/test_features.csv

Пишет:
    outputs/task1_predictions.csv  (колонки: patient_id, prediction)

Запуск:
    python your-sollution/task1_solution.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score

# Безопасный вывод в Windows-консоли (cp1251 не умеет emoji/юникод).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover
    pass

warnings.filterwarnings("ignore")

# --- Пути (относительно корня проекта) -------------------------------------
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "dataset" / "task1" / "train.csv"
TEST_PATH = ROOT / "dataset" / "task1" / "test_features.csv"
OUT_PATH = ROOT / "outputs" / "task1_predictions.csv"

# --- Константы --------------------------------------------------------------
TARGET = "prediction"
ID = "patient_id"

# Признаки-утечки: сильно связаны с таргетом, поэтому исключаются.
DROP_COLS = [ID, "Статус_глюкозы", "Доклинический_риск"]

RANDOM_STATE = 42


# === 01-02. Загрузка данных =================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Шаг 01-02: читает train и test с проверкой наличия файлов."""
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Не найден train: {TRAIN_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Не найден test: {TEST_PATH}")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def build_matrices(train: pd.DataFrame, test: pd.DataFrame):
    """Формирует X / y / X_test / test_ids (единая точка правды)."""
    missing = [c for c in DROP_COLS + [TARGET] if c not in train.columns]
    if missing:
        raise KeyError(f"В train отсутствуют колонки: {missing}")

    X = train.drop(columns=DROP_COLS + [TARGET])
    y = train[TARGET].astype(int)

    drop_in_test = [c for c in DROP_COLS if c in test.columns]
    X_test = test.drop(columns=drop_in_test)[X.columns]
    test_ids = test[ID].values

    assert list(X.columns) == list(X_test.columns), "колонки X и X_test расходятся"
    return X, y, X_test, test_ids


# === 03. Удаление выбросов (IQR) ============================================
def remove_outliers_iqr(
    X: pd.DataFrame, y: pd.Series, k: float = 1.5
) -> tuple[pd.DataFrame, pd.Series]:
    """Шаг 03: убирает строки с выбросами по правилу IQR.

    Границы [Q1 - k*IQR, Q3 + k*IQR] считаются по train-признакам.
    Выбросом считается значение, выходящее за границы хотя бы в одном
    непрерывном столбце (бинарные флаги не трогаем).
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    continuous = [c for c in numeric_cols if X[c].nunique(dropna=True) > 2]

    q1 = X[continuous].quantile(0.25)
    q3 = X[continuous].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    in_bounds = ((X[continuous] >= lower) & (X[continuous] <= upper)).all(axis=1)
    removed = int((~in_bounds).sum())
    print(f"[03] Удалено выбросов (IQR, k={k}): {removed} из {len(X)}")

    return (
        X.loc[in_bounds].reset_index(drop=True),
        y.loc[in_bounds].reset_index(drop=True),
    )


# === 04. Заполнение пропусков средним ======================================
def fit_imputer(X_train: pd.DataFrame) -> SimpleImputer:
    """Шаг 04: обучает импутер средним ТОЛЬКО на train (без утечки)."""
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train)
    return imputer


def apply_imputer(imputer: SimpleImputer, X: pd.DataFrame) -> pd.DataFrame:
    """Применяет обученный импутер, сохраняя колонки/индекс."""
    return pd.DataFrame(
        imputer.transform(X), columns=X.columns, index=X.index
    )


# === 05. Нормализация StandardScaler =======================================
def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Шаг 05: обучает StandardScaler ТОЛЬКО на train."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# === 07. Модель: 1 слой, 32 нейрона, ReLU, 150 эпох ========================
def build_model() -> MLPClassifier:
    """Шаг 07: нейросеть строго по заданию.

    - 1 скрытый слой из 32 нейронов: hidden_layer_sizes=(32,)
    - активация ReLU:                  activation='relu'
    - 150 эпох:                        max_iter=150
    """
    return MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        solver="adam",
        max_iter=150,
        random_state=RANDOM_STATE,
    )


def tune_threshold(y_true: pd.Series, proba: np.ndarray) -> float:
    """Подбирает порог по val, максимизируя F1.

    Классы сильно несбалансированы (81.9% / 18.1%), поэтому порог 0.5
    «схлопывает» все предсказания в мажоритарный класс. Подбор порога —
    корректный способ учесть дисбаланс без изменения архитектуры сети.
    """
    thresholds = np.arange(0.05, 0.95, 0.01)
    scores = [f1_score(y_true, (proba >= t).astype(int)) for t in thresholds]
    best_idx = int(np.argmax(scores))
    best_thr = float(thresholds[best_idx])
    print(
        f"[07] Порог по val: {best_thr:.2f} "
        f"(val F1={scores[best_idx]:.4f}, default 0.5 F1="
        f"{f1_score(y_true, (proba >= 0.5).astype(int)):.4f})"
    )
    return best_thr


def main() -> None:
    print(f"[01-02] Читаю: {TRAIN_PATH}")
    print(f"[01-02] Читаю: {TEST_PATH}")
    train, test = load_data()
    print(f"train: {train.shape} | test: {test.shape}")

    X, y, X_test, test_ids = build_matrices(train, test)
    print(f"X: {X.shape} | X_test: {X_test.shape}")
    print(f"Баланс классов: {y.value_counts(normalize=True).round(3).to_dict()}")

    # 03. Убираем выбросы только из train (test не фильтруем!).
    X, y = remove_outliers_iqr(X, y)

    # 06. Делим на train / val / test (60 / 20 / 20).
    # Внешний test_features.csv предсказывается в конце.
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_holdout, y_val, y_holdout = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=RANDOM_STATE
    )
    print(
        f"[06] Сплит: train={X_train.shape[0]} | "
        f"val={X_val.shape[0]} | test={X_holdout.shape[0]}"
    )

    # 04. Заполняем пропуски средним (fit только на train).
    imputer = fit_imputer(X_train)
    n_missing = int(X_train.isna().sum().sum())
    X_train = apply_imputer(imputer, X_train)
    X_val = apply_imputer(imputer, X_val)
    X_holdout = apply_imputer(imputer, X_holdout)
    X_test = apply_imputer(imputer, X_test)
    print(f"[04] Пропуски заполнены средним (train): {n_missing} значений")

    # 05. Нормализация (fit только на train).
    scaler = fit_scaler(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_holdout = scaler.transform(X_holdout)
    X_test = scaler.transform(X_test)
    print("[05] Данные нормализованы через StandardScaler")

    # 07. Обучение модели.
    model = build_model()
    model.fit(X_train, y_train)
    print(
        f"[07] MLP обучен: hidden_layer_sizes=(32,), activation='relu', "
        f"max_iter=150 (факт. эпох: {model.n_iter_})"
    )

    # Оценка вероятностей на val и отложенном test.
    proba_val = model.predict_proba(X_val)[:, 1]
    proba_hold = model.predict_proba(X_holdout)[:, 1]
    print(f"    val   ROC-AUC: {roc_auc_score(y_val, proba_val):.4f}")
    print(f"    test  ROC-AUC: {roc_auc_score(y_holdout, proba_hold):.4f}")

    # Подбор порога на val.
    best_thr = tune_threshold(y_val, proba_val)

    # Итоговые метрики на отложенном test с выбранным порогом.
    pred_hold = (proba_hold >= best_thr).astype(int)
    print(
        f"    test  F1@thr={best_thr:.2f}: "
        f"{f1_score(y_holdout, pred_hold):.4f}"
    )

    # 08. Предсказания для test_features.csv.
    proba_test = model.predict_proba(X_test)[:, 1]
    preds = (proba_test >= best_thr).astype(int)
    print(f"[08] Предсказания готовы: {len(preds)} строк")

    # 09. Сохранение результата.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({ID: test_ids, TARGET: preds})
    out.to_csv(OUT_PATH, index=False)
    print(f"[09] Сохранено: {OUT_PATH}")
    print(out[TARGET].value_counts().to_string())

    # Проверка выходного контракта.
    expected = pd.read_csv(TEST_PATH)
    assert len(out) == len(expected), "число строк не совпадает с test"
    assert list(out.columns) == [ID, TARGET], "неверные колонки"
    assert (out[ID].values == expected[ID].values).all(), "порядок ID нарушен"
    assert out[TARGET].notna().all(), "есть пропуски в prediction"
    assert set(out[TARGET].unique()) <= {0, 1}, "prediction вне {0, 1}"
    print("[OK] Контракт соблюдён: строк =", len(out))


if __name__ == "__main__":
    main()
