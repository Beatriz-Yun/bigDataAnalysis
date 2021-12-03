# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

# 데이터 파일 읽기 예제
import pandas as pd
X_test = pd.read_csv("data/X_test.csv")
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

# 사용자 코딩
pd.options.display.max_columns = None

# print(X_test.shape)
# print(X_train.info())

# 데이터 전처리
x_test_cust_id = X_test['cust_id']
X_train = X_train.drop(columns=['cust_id'])
X_test = X_test.drop(columns=['cust_id'])
y_train = y_train.drop(columns=['cust_id'])

X_train['환불금액'] = X_train['환불금액'].fillna(0)
X_test['환불금액'] = X_test['환불금액'].fillna(0)

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X_train['주구매상품'] = encoder.fit_transform(X_train['주구매상품'])
X_test['주구매상품'] = encoder.fit_transform(X_test['주구매상품'])
X_train['주구매지점'] = encoder.fit_transform(X_train['주구매지점'])
X_test['주구매지점'] = encoder.fit_transform(X_test['주구매지점'])


condition = X_train['환불금액'] > 0
X_train.loc[condition, '환불금액_new'] = 1
X_train.loc[~condition, '환불금액_new'] = 0
X_train.drop(columns=['환불금액'], inplace=True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

X_train.drop(columns=['최대구매액'], inplace=True)
X_test.drop(columns=['최대구매액'], inplace=True)


# 학습시키기
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=7, criterion='entropy')

model.fit(X_train, y_train)

y_test_predicted = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)

result = pd.DataFrame(y_test_proba)[1]


# 모델 평가하기
y_train_predicted = model.predict(X_train)

from sklearn.metrics import roc_auc_score

# print(roc_auc_score(y_train, y_train_predicted))

final = pd.concat([x_test_cust_id, result], axis=1).rename(columns={1:'gender'})
final.to_csv('data/1234.csv', index=False)



# 답안 제출 참고
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
# pd.DataFrame({'cust_id': X_test.cust_id, 'gender': pred}).to_csv('003000000.csv', index=False)
