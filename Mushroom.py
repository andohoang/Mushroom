#!/usr/bin/env python
# coding: utf-8

# # Mycology
# 
# Nấm học (Mycology) là một nhánh của sinh học nói chung liên quan đến việc nghiên cứu nấm, đặc biệt là cấu tạo di truyền và sinh hóa của chúng cũng như việc sử dụng chúng đối với con người. Các nhà khoa học ở những nơi khác cũng đã ghi nhận nhiều công dụng chữa bệnh của nấm, nhưng không phải tất cả các loại nấm đều có lợi - một số loại nấm khá nguy hiểm.
# 
# Chúg ta sẽ sử dụng cây quyết định để xem xét Tập dữ liệu về nấm, được trích từ Hướng dẫn thực địa của Hiệp hội Audobon về Nấm ở Bắc Mỹ (1981). Tập dữ liệu cung cấp chi tiết các loại nấm được mô tả về nhiều đặc điểm vật lý, chẳng hạn như kích thước nắp và chiều dài cuống, cùng với phân loại độc hoặc ăn được.
# 
# 

# In[194]:


import pandas as pd


# 1. Chuẩn bị dữ liệu
# Truy cập [trang của tập dữ liệu](https://archive.ics.uci.edu/ml/datasets/Mushroom) để lấy dữ liệu và đọc hiểu dữ liệu

# In[195]:


X=pd.read_csv('agaricus-lepiota.data')
X


# In[196]:


# Hiện những dòng dữ liệu có giá trị Nan
X[pd.isnull(X).any(axis=1)]


# In[197]:


# Xem kích thước của tập dữ liệu
X.shape


# In[198]:


# Xóa những dòng có giá trị Nan
X= X.dropna()
print(X.shape)


# In[199]:


#Sao chép các nhãn ra khỏi khung dữ liệu vào biến y, sau đó xóa chúng khỏi X
y=X['p']
X=X.drop(columns='p')
y


# In[201]:


# Mã hóa các nhãn 
label ={'e':0,'p':1}
y = y.map(label)
y


# In[202]:


# Mã hóa dữ liệu X
for column in X.columns:
        value_map={}
        value_unique=X[column].unique()
        number_value=len(value_unique)
        for i in range(number_value):
            value_map[value_unique[i]]=i
        X[column]=X[column].map(value_map)

X


# In[203]:


# Chia dữ liệu của bạn thành các tập `test` và` train`. Kích thước `test` phải là 30% với `random_state` là 7.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=7)


# 2. Huấn luyện và Đánh giá Mô hình

# In[204]:


# Thiết lập mô hình
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(criterion="entropy", max_depth = 4)
model


# In[205]:


# Huấn luyện bộ phân loại với bộ dữ liệu huấn luyện X_train, y_train
model.fit(X_train,y_train)


# In[206]:


from sklearn.metrics import accuracy_score

# Dự đoán nhãn của tập test
y_pred=model.predict(X_test) 

# Đánh giá mô hình bằng độ chính xác
score=accuracy_score(y_test,y_pred)

print("High-Dimensionality Score: ", round((score*100), 3))


# In[ ]:


Accuracy score là một phép đo đánh giá hiệu suất của mô hình học máy.
Accuracy score là 98.318% đối với mô hình DecisionTreeClassifier cho thấy rằng mô hình đã phân loại đúng 
khoảng 98.318% nhãn các loại nấm là có độc(p) hay có thể ăn được(e).


# In[208]:


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test,y_pred)


# In[ ]:


* precision_recall_fscore_support dựa trên hai nhãn khác nhau (0 và 1) trên một tập dữ liệu kiểm tra. Có thể hiểu như sau:

    Precision (Độ chính xác) cho nhãn 0 là khoảng 96.85%, nghĩa là trong các trường hợp mô hình dự đoán nhãn 0 - có thể ăn được , khoảng 96.85% là đúng.

    Precision cho nhãn 1 là 100%, nghĩa là trong các trường hợp mô hình dự đoán nhãn 1 - có độc , tất cả đều đúng.

    Recall (Tỷ lệ tìm thấy tất cả các dự đoán tích cực) cho nhãn 0 là 100%, nghĩa là mô hình tìm thấy tất cả các mẫu thực sự thuộc nhãn 0 - có thể ăn được.

    Recall cho nhãn 1 là khoảng 96.52%, nghĩa là mô hình tìm thấy khoảng 96.52% các mẫu thực sự thuộc nhãn 1 - có độc.

    F1-score (F1-score) cho nhãn 0 là khoảng 98.39%

    F1-score cho nhãn 1 là khoảng 98.24%

    Support (Hỗ trợ) cho nhãn 0 là 1260 mẫu, và cho nhãn 1 là 1177 mẫu. Đây là số lượng mẫu thực sự thuộc mỗi nhãn trong tập dữ liệu kiểm tra.

Kết quả này cho thấy mô hình có hiệu suất tốt với các thông số Precision, Recall và F1-score cao cho cả hai nhãn.

