# Bài tập 01 - Xây dựng Pipeline Machine Learning

## 📋 Thông tin chung

**Mục tiêu**: Xây dựng pipeline hoàn chỉnh để phân loại chất lượng kết nối RF (RF Link Quality) từ dữ liệu truyền thông không dây.

**Dataset**: `wireless_communication_dataset.csv`

**File thực hiện**: `nhom_01_BT01.ipynb`
---

## 🎯 Yêu cầu bài tập

1. ✅ Xây dựng Pipeline duy nhất xử lý toàn bộ dữ liệu
2. ✅ Sử dụng ColumnTransformer để xử lý riêng biệt:
   - Cột số (numeric): StandardScaler
   - Cột phân loại (categorical): OneHotEncoder
3. ✅ Mô hình phân loại: DecisionTreeClassifier hoặc KNeighborsClassifier
4. ✅ Chia dữ liệu 80% train, 20% test
5. ✅ Báo cáo Accuracy và F1-Score trên tập test

---

## 📊 Tổng quan về Dataset

### Thông tin dữ liệu
- **Số lượng mẫu**: 5000 hàng
- **Số lượng features**: 17 cột
- **Biến mục tiêu**: `RF Link Quality` (Poor, Moderate, Good, 0)

### Các cột trong dataset

#### 1. Cột số (Numeric Features) - 15 cột:
1. `User Speed (m/s)` - Tốc độ người dùng
2. `User Direction (degrees)` - Hướng di chuyển
3. `Handover Events` - Số lần chuyển giao
4. `Distance from Base Station (m)` - Khoảng cách từ trạm gốc
5. `Signal Strength (dBm)` - Cường độ tín hiệu
6. `SNR (dB)` - Tỷ lệ tín hiệu trên nhiễu
7. `BER` - Tỷ lệ lỗi bit
8. `PDR (%)` - Tỷ lệ gói tin được gửi
9. `Throughput (Mbps)` - Băng thông
10. `Latency (ms)` - Độ trễ
11. `Retransmission Count` - Số lần truyền lại
12. `Power Consumption (mW)` - Tiêu thụ năng lượng
13. `Battery Level (%)` - Mức pin
14. `Transmission Power (dBm)` - Công suất truyền
15. `Network Congestion` - Mức độ tắc nghẽn mạng (có thể là categorical)

#### 2. Cột phân loại (Categorical Features) - 2 cột:
1. `Modulation Scheme` - Sơ đồ điều chế (BPSK, QPSK, 16-QAM, 64-QAM)
2. `Network Congestion` - Tắc nghẽn mạng (Low, Medium, High)

#### 3. Biến mục tiêu (Target):
- `RF Link Quality` - Chất lượng kết nối RF (Poor, Moderate, Good, 0)

---

## 🔍 Quy trình thực hiện chi tiết

### **Bước 1: Import thư viện** 

#### Mục đích:
Chuẩn bị các công cụ cần thiết để xử lý dữ liệu, xây dựng pipeline và đánh giá mô hình.

#### Thư viện sử dụng:
```python
import pandas as pd                    # Xử lý dữ liệu dạng bảng
import numpy as np                     # Tính toán số học
import matplotlib.pyplot as plt        # Vẽ biểu đồ
import seaborn as sns                  # Trực quan hóa dữ liệu nâng cao

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
```

#### Kiến thức liên quan:
- **Pandas**: Thư viện xử lý dữ liệu dạng DataFrame
- **Scikit-learn**: Thư viện Machine Learning phổ biến nhất Python
- **Matplotlib/Seaborn**: Thư viện trực quan hóa dữ liệu

---

### **Bước 2: Tải và khám phá dữ liệu (Exploratory Data Analysis - EDA)**

#### 2.1 Đọc dữ liệu
```python
df = pd.read_csv('wireless_communication_dataset.csv')
```

#### 2.2 Kiểm tra thông tin cơ bản
- **Shape**: Kích thước dữ liệu (số hàng × số cột)
- **Info**: Kiểu dữ liệu của từng cột, số lượng non-null values
- **Describe**: Thống kê mô tả (mean, std, min, max, quartiles)
- **Head/Tail**: Xem một số hàng đầu/cuối

#### 2.3 Kiểm tra giá trị thiếu
```python
df.isnull().sum()
```
**Kết quả**: Không có giá trị thiếu trong dataset

#### 2.4 Phân tích biến mục tiêu
```python
df['RF Link Quality'].value_counts()
```
- Xác định số lượng mẫu trong từng lớp
- Kiểm tra sự cân bằng/mất cân bằng của dữ liệu
- Phát hiện giá trị '0' (unknown/undefined) cần xử lý

#### Kiến thức liên quan:
- **EDA**: Quá trình khám phá, hiểu rõ dữ liệu trước khi xây dựng mô hình
- **Missing values**: Giá trị thiếu có thể ảnh hưởng đến chất lượng mô hình
- **Class imbalance**: Sự mất cân bằng giữa các lớp có thể làm mô hình thiên vị

---

### **Bước 3: Phân tích và xác định loại cột**

#### 3.1 Tự động phân loại cột
```python
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
```

#### 3.2 Loại bỏ cột mục tiêu khỏi features
```python
if target_column in categorical_features:
    categorical_features.remove(target_column)
```

#### 3.3 Kết quả phân loại
- **15 cột số**: Các đặc trưng liên tục (continuous)
- **2 cột phân loại**: Modulation Scheme, Network Congestion
- **1 cột mục tiêu**: RF Link Quality

#### Kiến thức liên quan:
- **Feature types**:
  - **Numeric**: Dữ liệu số (continuous hoặc discrete)
  - **Categorical**: Dữ liệu phân loại (nominal hoặc ordinal)
- **Feature engineering**: Việc xử lý khác nhau cho từng loại features

---

### **Bước 4: Trực quan hóa dữ liệu**

#### 4.1 Biểu đồ phân phối biến mục tiêu
```python
df['RF Link Quality'].value_counts().plot(kind='bar')
```
**Mục đích**: Hiểu rõ phân phối các lớp trong dataset

#### 4.2 Ma trận tương quan (Correlation Matrix)
```python
correlation_matrix = df[numeric_features].corr()
sns.heatmap(correlation_matrix, annot=True)
```

**Mục đích**: 
- Phát hiện mối quan hệ tuyến tính giữa các biến
- Xác định multicollinearity (đa cộng tuyến)
- Loại bỏ features dư thừa nếu cần

#### Kiến thức liên quan:
- **Correlation**: Đo lường mối quan hệ tuyến tính (-1 đến +1)
- **Heatmap**: Biểu đồ nhiệt thể hiện correlation matrix
- **Feature selection**: Chọn features quan trọng, loại bỏ features không cần thiết

---

### **Bước 5: Chuẩn bị dữ liệu (Data Preparation)**

#### 5.1 Xử lý dữ liệu nhiễu
```python
df_cleaned = df[df['RF Link Quality'] != '0'].copy()
```
**Lý do**: Loại bỏ các mẫu có nhãn '0' (không xác định/không hợp lệ)

#### 5.2 Tách features và target
```python
X = df_cleaned.drop(columns=[target_column])
y = df_cleaned[target_column]
```

#### 5.3 Chia dữ liệu Train/Test (80/20)
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Tham số quan trọng**:
- `test_size=0.2`: 20% dữ liệu cho test set
- `random_state=42`: Đảm bảo kết quả có thể tái tạo
- `stratify=y`: Giữ nguyên tỷ lệ các lớp trong train và test set

#### Kiến thức liên quan:
- **Train/Test split**: Tránh overfitting, đánh giá khách quan
- **Stratified sampling**: Quan trọng với dữ liệu không cân bằng
- **Random state**: Đảm bảo reproducibility trong nghiên cứu

---

### **Bước 6: Xây dựng Pipeline với ColumnTransformer** ⭐

#### 6.1 Khái niệm Pipeline
**Pipeline** là một công cụ trong Scikit-learn cho phép:
- Kết hợp nhiều bước xử lý dữ liệu thành một chuỗi
- Đảm bảo các bước được thực hiện theo đúng thứ tự
- Tránh data leakage giữa train và test set
- Code gọn gàng, dễ bảo trì

#### 6.2 ColumnTransformer - Xử lý riêng biệt các loại cột

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)
```

**Giải thích**:
- **StandardScaler** cho cột số:
  - Chuẩn hóa dữ liệu về mean=0, std=1
  - Công thức: `z = (x - μ) / σ`
  - Quan trọng cho KNN và các thuật toán dựa trên khoảng cách
  
- **OneHotEncoder** cho cột phân loại:
  - Chuyển categorical thành binary vectors
  - Ví dụ: ['Low', 'Medium', 'High'] → [[1,0,0], [0,1,0], [0,0,1]]
  - `handle_unknown='ignore'`: Xử lý giá trị mới chưa gặp trong training

#### 6.3 Tạo Pipeline hoàn chỉnh

```python
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(...))
])
```

**Ưu điểm Pipeline**:
1. ✅ Tự động apply các bước preprocessing cho cả train và test
2. ✅ Tránh data leakage (fit_transform trên train, transform trên test)
3. ✅ Dễ dàng thay đổi và thử nghiệm các thuật toán khác nhau
4. ✅ Có thể lưu và tái sử dụng toàn bộ pipeline

#### Kiến thức liên quan:
- **Feature scaling**: Chuẩn hóa để các features có tầm ảnh hưởng tương đương
- **One-hot encoding**: Biến đổi categorical thành numeric
- **Data leakage**: Lỗi nghiêm trọng khi thông tin từ test "rò rỉ" vào train
- **Pipeline pattern**: Design pattern quan trọng trong ML

---

### **Bước 7: Huấn luyện và đánh giá mô hình**

#### 7.1 DecisionTreeClassifier

**Thuật toán**:
- Cây quyết định phân loại dựa trên việc chia dữ liệu theo các điều kiện
- Mỗi node là một điều kiện kiểm tra (if-else)
- Leaf nodes chứa kết quả phân loại

**Hyperparameters**:
```python
DecisionTreeClassifier(
    random_state=42,        # Reproducibility
    max_depth=10,           # Độ sâu tối đa của cây (tránh overfitting)
    min_samples_split=10    # Số mẫu tối thiểu để chia node
)
```

**Ưu điểm**:
- Dễ hiểu, dễ visualize
- Không cần feature scaling
- Xử lý được cả numeric và categorical
- Có thể học non-linear relationships

**Nhược điểm**:
- Dễ overfit
- Không ổn định (nhạy cảm với thay đổi dữ liệu)
- Không tốt với dữ liệu có nhiều chiều

#### 7.2 KNeighborsClassifier

**Thuật toán**:
- Phân loại dựa trên K láng giềng gần nhất
- Tính khoảng cách từ điểm test đến tất cả điểm train
- Lấy majority vote từ K láng giềng

**Hyperparameters**:
```python
KNeighborsClassifier(
    n_neighbors=5,      # Số láng giềng xét đến
    weights='distance'  # Láng giềng gần hơn có trọng số lớn hơn
)
```

**Ưu điểm**:
- Đơn giản, dễ implement
- Không có training phase (lazy learning)
- Hiệu quả với dữ liệu nhỏ

**Nhược điểm**:
- Chậm với dữ liệu lớn (phải tính khoảng cách đến tất cả điểm)
- Nhạy cảm với scaling và outliers
- Curse of dimensionality (không tốt với nhiều features)

#### 7.3 Huấn luyện
```python
pipeline.fit(X_train, y_train)
```
**Quá trình**:
1. Fit StandardScaler trên X_train (numeric)
2. Fit OneHotEncoder trên X_train (categorical)
3. Transform X_train bằng fitted transformers
4. Fit classifier trên transformed data

#### 7.4 Dự đoán
```python
y_pred = pipeline.predict(X_test)
```
**Quá trình**:
1. Transform X_test (không fit lại!)
2. Dự đoán bằng fitted classifier

---

### **Bước 8: Đánh giá hiệu suất mô hình**

#### 8.1 Các metrics sử dụng

**1. Accuracy (Độ chính xác)**
```python
accuracy = accuracy_score(y_test, y_pred)
```
- Công thức: `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- Tỷ lệ dự đoán đúng trên tổng số mẫu
- **Hạn chế**: Không phù hợp với dữ liệu mất cân bằng

**2. F1-Score (weighted)**
```python
f1 = f1_score(y_test, y_pred, average='weighted')
```
- Công thức: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
- `weighted`: Tính F1 cho từng class, sau đó lấy trung bình có trọng số
- Cân bằng giữa Precision và Recall
- **Tốt hơn Accuracy** với dữ liệu không cân bằng

**3. Classification Report**
```python
classification_report(y_test, y_pred)
```
- Báo cáo chi tiết cho từng class:
  - **Precision**: Độ chính xác của dự đoán positive
  - **Recall**: Tỷ lệ phát hiện được positive thực tế
  - **F1-Score**: Trung bình điều hòa của Precision và Recall
  - **Support**: Số mẫu thực tế của mỗi class

**4. Confusion Matrix (Ma trận nhầm lẫn)**
```python
confusion_matrix(y_test, y_pred)
```
- Ma trận hiển thị số lượng dự đoán đúng/sai cho mỗi class
- Trục dọc: True label
- Trục ngang: Predicted label
- Diagonal: Dự đoán đúng

#### 8.2 Trực quan hóa kết quả

**Heatmap của Confusion Matrix**:
```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
```
- Dễ nhìn thấy pattern dự đoán sai
- Xác định class nào bị nhầm lẫn nhiều

**Bar chart so sánh**:
- So sánh Accuracy giữa các mô hình
- So sánh F1-Score giữa các mô hình

---

### **Bước 9: So sánh và lựa chọn mô hình**

#### 9.1 Tiêu chí đánh giá
1. **Accuracy**: Độ chính xác tổng thể
2. **F1-Score**: Cân bằng Precision-Recall
3. **Training time**: Thời gian huấn luyện
4. **Prediction time**: Thời gian dự đoán
5. **Interpretability**: Khả năng giải thích

#### 9.2 Trade-offs
- **Decision Tree**: Nhanh, dễ hiểu, dễ overfit
- **KNN**: Đơn giản, chậm khi predict, nhạy scaling

#### 9.3 Lựa chọn mô hình tốt nhất
```python
best_model = 'Decision Tree' if accuracy_dt > accuracy_knn else 'K-Neighbors'
```

---

### **Bước 10: Lưu mô hình (Model Persistence)**

```python
import joblib
joblib.dump(best_pipeline, 'best_model.pkl')
```

**Lý do cần lưu mô hình**:
- Sử dụng lại mà không cần train lại
- Deploy vào production
- Chia sẻ với người khác

**Load mô hình**:
```python
loaded_model = joblib.load('best_model.pkl')
predictions = loaded_model.predict(new_data)
```

---

## 📚 Kiến thức quan trọng đã học

### 1. **Pipeline và ColumnTransformer**
- Xây dựng quy trình ML hoàn chỉnh
- Xử lý riêng biệt cho từng loại features
- Tránh data leakage

### 2. **Feature Preprocessing**
- **StandardScaler**: Chuẩn hóa cho numeric features
- **OneHotEncoder**: Mã hóa cho categorical features
- Tầm quan trọng của feature scaling

### 3. **Classification Algorithms**
- **Decision Tree**: Thuật toán dựa trên cây quyết định
- **K-Neighbors**: Thuật toán dựa trên khoảng cách
- Ưu nhược điểm của từng thuật toán

### 4. **Model Evaluation**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Classification Report
- Tầm quan trọng của việc chọn metric phù hợp

### 5. **Best Practices**
- Train/Test split với stratify
- Random state để reproducibility
- Pipeline để code gọn gàng
- EDA trước khi modeling

---

## 🔧 Cải thiện mô hình (Future Work)

### 1. **Hyperparameter Tuning**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__max_depth': [5, 10, 15, 20],
    'classifier__min_samples_split': [5, 10, 20]
}

grid_search = GridSearchCV(pipeline_dt, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### 2. **Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 3. **Feature Engineering**
- Tạo features mới từ features hiện có
- Feature selection (loại bỏ features không quan trọng)
- Polynomial features

### 4. **Ensemble Methods**
- Random Forest (cải tiến của Decision Tree)
- Gradient Boosting (XGBoost, LightGBM)
- Voting Classifier (kết hợp nhiều mô hình)

### 5. **Xử lý Class Imbalance**
- SMOTE (Synthetic Minority Over-sampling)
- Class weights
- Undersampling/Oversampling

---


## 💡 Tips và Lưu ý

### ✅ Do's (Nên làm)
1. **Luôn chia dữ liệu Train/Test** trước khi làm bất cứ điều gì
2. **Sử dụng Pipeline** để đảm bảo quy trình nhất quán
3. **Stratify** khi split với classification problems
4. **Set random_state** để reproducibility
5. **EDA kỹ lưỡng** trước khi modeling
6. **Evaluate trên nhiều metrics**, không chỉ Accuracy
7. **Visualize** confusion matrix để hiểu rõ lỗi

### ❌ Don'ts (Không nên làm)
1. **Không fit scaler trên cả dataset** (phải fit riêng trên train)
2. **Không bỏ qua EDA** và đi thẳng vào modeling
3. **Không chỉ dựa vào Accuracy** với imbalanced data
4. **Không overfit** bằng cách tune quá nhiều trên test set
5. **Không quên handle categorical features** đúng cách
6. **Không bỏ qua missing values** và outliers

---

## 🎓 Kết luận

Bài tập này đã giúp nắm vững:
- ✅ Quy trình hoàn chỉnh của một ML project
- ✅ Sử dụng Pipeline và ColumnTransformer hiệu quả
- ✅ Preprocessing khác nhau cho từng loại features
- ✅ So sánh và đánh giá nhiều mô hình
- ✅ Best practices trong Machine Learning

**Kỹ năng đạt được**:
1. Data preprocessing và feature engineering
2. Building ML pipelines
3. Model training và evaluation
4. Model comparison và selection
5. Code organization và documentation

---


**Ngày hoàn thành**: November 22, 2025

**Phiên bản**: 1.0

**Tác giả**: Nhóm 01
#   X - y - d - n g - P i p e l i n e - p h - n - l o - i - c h - t - l - n g - m - n g 
 
 