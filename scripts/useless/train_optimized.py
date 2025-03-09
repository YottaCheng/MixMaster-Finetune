import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder  # 新增导入

# 配置项
CONFIG = {
    "data_path": "../data/processed/train_data_en.csv",
    "model_save_dir": "../data/outputs/v2",
    "test_size": 0.2,
    "random_state": 42,
    "tfidf_params": {
        "ngram_range": (1, 3),
        "max_features": 2000,
        "sublinear_tf": True,
        "stop_words": "english"
    },
    "models": {
        "XGBoost": XGBClassifier(
            objective='multi:softmax',
            num_class=7,
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6
        ),
        "SVM": SVC(
            class_weight='balanced',
            kernel='rbf',
            probability=True
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight='balanced',
            max_depth=10
        )
    }
}


def main():
    # 创建输出目录
    os.makedirs(CONFIG["model_save_dir"], exist_ok=True)
    
    # 加载数据
    df = pd.read_csv(CONFIG["data_path"])
    print(f"✅ 数据加载完成 | 总样本数: {len(df)}")
    print("📊 标签分布:\n", df["primary_label_en"].value_counts())
    
    # 数据预处理
    X = df["text_en"].fillna("").str.lower()
    y_orig = df["primary_label_en"]
    le = LabelEncoder()
    y = le.fit_transform(y_orig)  # 将字符串标签转换为数字

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=CONFIG["test_size"],
        stratify=y,
        random_state=CONFIG["random_state"]
    )
    print(f"\n🔀 数据集划分 | 训练集: {len(X_train)} | 测试集: {len(X_test)}")

    # 特征工程
    tfidf = TfidfVectorizer(**CONFIG["tfidf_params"])
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # 交叉验证训练
    best_model = None
    best_f1 = 0
    for model_name, model in CONFIG["models"].items():
        print(f"\n🚀 正在训练模型: {model_name}")
        
        # 创建包含SMOTE的流水线
        pipeline = make_pipeline(
            SMOTE(random_state=CONFIG["random_state"]),
            model
        )
        
        # 交叉验证
        cv = StratifiedKFold(n_splits=5)
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_vec, y_train)):
            X_tr, X_val = X_train_vec[train_idx], X_train_vec[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            fold_f1 = f1_score(y_val, y_pred, average="macro")
            cv_scores.append(fold_f1)
            print(f"Fold {fold+1} | Macro-F1: {fold_f1:.3f}")

        mean_f1 = np.mean(cv_scores)
        print(f"📈 {model_name} 平均交叉验证Macro-F1: {mean_f1:.3f}")

        # 选择最佳模型
        if mean_f1 > best_f1:
            best_model = pipeline
            best_f1 = mean_f1
            joblib.dump(best_model, os.path.join(CONFIG["model_save_dir"], f"best_{model_name}.pkl"))
            joblib.dump(tfidf, os.path.join(CONFIG["model_save_dir"], "optimized_tfidf.pkl"))
            # 保存标签编码器
            joblib.dump(le, os.path.join(CONFIG["model_save_dir"], "label_encoder.pkl"))

    # 最终评估
    print("\n🔥 最佳模型最终测试集表现:")
    y_pred = best_model.predict(X_test_vec)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    main()

