# 数据处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import plotly.figure_factory as ff
import plotly as py


#读入数据集
df=pd.read_csv("Telco-Customer-Churn.csv")
# print(df.head(10).to_string())

# 数据初步清洗
# 首先进行初步的数据清洗工作，包含错误值和异常值处理，并划分类别型和数值型字段类型，
# 其中清洗部分包含：OnlineSecurity、OnlineBackup、DeviceProtection、TechSupport、StreamingTV、StreamingMovies：
# 错误值处理 TotalCharges：异常值处理 tenure：自定义分箱
# 错误值处理
C_olumns=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for i in C_olumns:
    df[i]=df[i].replace({ 'No internet service': 'No'})
# 替换SeniorCitizen,Yes:1,No:0
df['SeniorCitizen']=df['SeniorCitizen'].replace({1:'Yes',0:'No'})
# 替换TotalCharges进而对空值进行删除
df['TotalCharges']=df['TotalCharges'].replace('',np.nan)
df=df.dropna(subset=['TotalCharges'])
# 重置索引
df.reset_index(drop=True,inplace=True)
# print(df.head(100).to_string())

# 将TotalCharges列中的字符串转换为浮点数
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# 转换tenure,编写函数
def transform_tenure(x):
    if x<=12:
        return('Tenure_1')
    elif x<=24:
        return('Tenure_2')
    elif x<=36:
        return('Tenure_3')
    elif x<=48:
        return('Tenure_4')
    elif x<=60:
        return('Tenure_5')
    else:
        return('Tenure_over_5')
df['tenure_group']=df.tenure.apply(transform_tenure)
# print(df.head(100).to_string())
# 数值型和类别型字段分类

# print("类别型字段：",Category_cols)
# print('**********************')
# print("数值型字段：",num_cols)

# 探索性分析
# 目标变量Churn分布
# 可视化
df['Churn'].value_counts
trace0 = go.Pie(labels=df[ 'Churn'].value_counts().index,
values=df[ 'Churn'].value_counts().values,
hole= 0.5,
rotation= 90,
marker=dict(colors=[ 'rgb(154,203,228)', 'rgb(191,76,81)'],
line=dict(color= 'white', width= 1.3))
)
data = [trace0]
layout = go.Layout(title= '目标变量Churn分布')
fig = go.Figure(data=data, layout=layout)
py.offline.plot(fig, filename= '整体流失情况分布.html',auto_open=False)


# 性别与是否流失的关系
# 男性和女性在客户流失比例上没有显著差异
a1 = df[(df['Churn']=='Yes')&(df['gender']=='Female')]['Churn'].count()
a_1 = df[(df['Churn']=='Yes')&(df['gender']=='Male')]['Churn'].count()
a2 = df[(df['Churn']=='No')&(df['gender']=='Female')]['Churn'].count()
a_2 = df[(df['Churn']=='No')&(df['gender']=='Male')]['Churn'].count()

a1_p = a1/(a_1+a1)
a_1_p = a_1/(a_1+a1)
a2_p = a2/(a2+a_2)
a_2_p = a_2/(a2+a_2)

plt.bar(['Female','Male'],height=[a1_p,a_1_p], width=0.3,color='yellow', data=None,label=u'yes')
plt.bar(['Female','Male'],height=[a2_p,a_2_p], width=0.3, bottom=[a1_p,a_1_p],color='red', data=None,label=u'no')

plt.legend(loc='best')
plt.show()

# 在网时长与是否流失的关系
# 用户的在网时长越长，表示用户的忠诚度越高，其流失的概率越低
a1 = df[(df['Churn']=='Yes')&(df['tenure_group']=='Tenure_1')]['Churn'].count()
a_1 = df[(df['Churn']=='No')&(df['tenure_group']=='Tenure_1')]['Churn'].count()
a2 = df[(df['Churn']=='Yes')&(df['tenure_group']=='Tenure_2')]['Churn'].count()
a_2 = df[(df['Churn']=='No')&(df['tenure_group']=='Tenure_2')]['Churn'].count()
a3 = df[(df['Churn']=='Yes')&(df['tenure_group']=='Tenure_3')]['Churn'].count()
a_3 = df[(df['Churn']=='No')&(df['tenure_group']=='Tenure_3')]['Churn'].count()
a4 = df[(df['Churn']=='Yes')&(df['tenure_group']=='Tenure_4')]['Churn'].count()
a_4 = df[(df['Churn']=='No')&(df['tenure_group']=='Tenure_4')]['Churn'].count()
a5 = df[(df['Churn']=='Yes')&(df['tenure_group']=='Tenure_5')]['Churn'].count()
a_5 = df[(df['Churn']=='No')&(df['tenure_group']=='Tenure_5')]['Churn'].count()
a6 = df[(df['Churn']=='Yes')&(df['tenure_group']=='Tenure_over_5')]['Churn'].count()
a_6 = df[(df['Churn']=='No')&(df['tenure_group']=='Tenure_over_5')]['Churn'].count()

a1_p = a1/(a1+a_1)
a_1_p = a_1/(a1+a_1)
a2_p = a2/(a2+a_2)
a_2_p = a_2/(a2+a_2)
a3_p = a3/(a3+a_3)
a_3_p = a_3/(a3+a_3)
a4_p = a4/(a4+a_4)
a_4_p = a_4/(a4+a_4)
a5_p = a5/(a5+a_5)
a_5_p = a_5/(a5+a_5)
a6_p = a6/(a6+a_6)
a_6_p = a_6/(a6+a_6)
#
plt.bar(['Tenure_1','Tenure_2','Tenure_3','Tenure_4','Tenure_5','Tenure_over_5'],height=[a1_p,a2_p,a3_p,a4_p,a5_p,a6_p], width=0.3,color='yellow', data=None,label=u'yes')
plt.bar(['Tenure_1','Tenure_2','Tenure_3','Tenure_4','Tenure_5','Tenure_over_5'],height=[a_1_p,a_2_p,a_3_p,a_4_p,a_5_p,a_6_p], width=0.3, bottom=[a1_p,a2_p,a3_p,a4_p,a5_p,a6_p],color='red', data=None,label=u'no')

plt.legend(loc='best')
plt.show()
# 对于二分类变量，编码为0和1;
# 对于多分类变量，进行one_hot编码；
# 对于数值型变量，部分模型如KNN、神经网络、Logistic需要进行标准化处理。
# 建模数据
df_model=df
Id_col=['customerID']
Target_cil=['Churn']
# 分类型
Category_cols=df.nunique()[df.nunique()<10].index.tolist()
# 数值型
num_cols=[i for i in df.columns if i not in Category_cols +Id_col]
# 二分类型
binary_cols=df_model.nunique()[df_model.nunique()==2].index.tolist()
# 多分类型
multi_cols=[i for i in Category_cols if i not in binary_cols]
# 二分类标签编码
le=LabelEncoder()
for i in binary_cols:
    df_model[i]=le.fit_transform(df_model[i])
#多分类哑变量变换
df_model=df_model.dropna()
df_model=pd.get_dummies(data=df_model,columns=multi_cols)
# print(df.head(100).to_string())
# 使用统计检定方式进行特征筛选。
X = df_model.copy().drop(['customerID','Churn'], axis=1)
y = df_model[Target_cil]
fs = SelectKBest(score_func=f_classif, k=20)
y = y.values.ravel()
X_train_fs = fs.fit_transform(X,y)
def SelectName(feature_data, model):
    scores = model.scores_
    indices = np.argsort(scores)[::-1]
    return list(feature_data.columns.values[indices[0:model.k]])
fea_name = [i for i in X.columns if i in SelectName(X,fs)]
X_train = pd.DataFrame(X_train_fs,columns = fea_name)
# print(X_train.to_string())
# 模型建立和评估
# 首先使用分层抽样的方式将数据划分训练集和测试集。
# 重新划分
# 分层抽样
X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2, random_state=0, stratify=y)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
#修正索引
for i in[X_train, X_test, y_train, y_test]:
    i.index= range(i.shape[0])
# 保存标准化训练和测试数据
st= StandardScaler()
num_scaled_train= pd.DataFrame(st.fit_transform(X_train[num_cols]), columns=num_cols)
num_scaled_test= pd.DataFrame(st.transform(X_test[num_cols]), columns=num_cols)
X_train_scaled= pd.concat([X_train.drop(num_cols, axis= 1), num_scaled_train], axis= 1)
X_test_scaled= pd.concat([X_test.drop(num_cols, axis= 1), num_scaled_test], axis= 1)
parameters = { 'splitter': ( 'best', 'random'),
'criterion': ( "gini", "entropy"),
"max_depth": [* range( 3, 20)],
}

clf = DecisionTreeClassifier(random_state= 25)

GS = GridSearchCV(clf, parameters, scoring= 'f1', cv= 10)
GS.fit(X_train, y_train)
# print(GS.best_params_)
# print(GS.best_score_)
clf = GS.best_estimator_
test_pred = clf.predict(X_test)
# print('测试集：n', classification_report(y_test, test_pred))
# 输出决策树属性重要性排序
imp = pd.DataFrame(zip(X_train.columns, clf.feature_importances_))
imp.columns = ['feature', 'importances']
imp = imp.sort_values('importances', ascending=False)
imp = imp[imp['importances'] != 0]
table = ff.create_table(np.round(imp, 4))
py.offline.iplot(table)

