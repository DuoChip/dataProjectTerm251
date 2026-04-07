# ==========================================
# BƯỚC 1: Import thư viện
# ==========================================
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# BƯỚC 2: Đọc dữ liệu
# ==========================================
df = pd.read_csv("world_development_data_imputed.csv")

# Kiểm tra nhanh
print("Số dòng và cột:", df.shape)
print("Các cột có trong dữ liệu:")
print(df.columns.tolist())

# ==========================================
# BƯỚC 3: Tạo nhóm tuổi thọ (Low / Medium / High)
# ==========================================
df['LifeGroup'] = pd.qcut(df['LifeExpBirth'], q=3, labels=['Low', 'Medium', 'High'])

# ==========================================
# BƯỚC 4: Tạo nhóm thời gian (Period)
# ==========================================
def categorize_period(year):
    if year <= 2007:
        return "2000-2007"
    elif year <= 2014:
        return "2008-2014"
    else:
        return "2015+"

df['Period'] = df['Year'].apply(categorize_period)

# ==========================================
# BƯỚC 5: Tạo nhãn tổ hợp (Region + LifeGroup + Period)
# ==========================================
df['Strata'] = (
    df['Region'].astype(str) + "_" +
    df['LifeGroup'].astype(str) + "_" 
    # df['Period'].astype(str)
)

# ==========================================
# BƯỚC 6: Chia dữ liệu có phân tầng 3 điều kiện
# ==========================================
train_df, temp_df = train_test_split(
    df,
    test_size=0.4,               # 40% tạm chia cho valid + test
    random_state=42,
    stratify=df['Strata']        # Ràng buộc 3 điều kiện
)

valid_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,               # 20% test, 20% valid
    random_state=42,
    stratify=temp_df['Strata']
)

# ==========================================
# BƯỚC 7: Lưu ra file
# ==========================================
os.makedirs("./devided_sets", exist_ok=True)
train_df.to_csv("./devided_sets/train.csv", index=False)
valid_df.to_csv("./devided_sets/valid.csv", index=False)
test_df.to_csv("./devided_sets/test.csv", index=False)

print("\n✅ Đã chia dữ liệu xong theo 3 ràng buộc (Region + LifeGroup + Period)!")
print("Train:", train_df.shape)
print("Valid:", valid_df.shape)
print("Test:", test_df.shape)

# ==========================================
# BƯỚC 8: Kiểm tra phân bố tuổi thọ
# ==========================================
plt.figure(figsize=(10,6))
sns.histplot(train_df["LifeExpBirth"], bins=20, color="blue", alpha=0.5, label="Train")
sns.histplot(valid_df["LifeExpBirth"], bins=20, color="orange", alpha=0.5, label="Valid")
sns.histplot(test_df["LifeExpBirth"], bins=20, color="green", alpha=0.5, label="Test")
plt.title("Phân bố tuổi thọ trung bình ở 3 tập dữ liệu", fontsize=14)
plt.xlabel("Tuổi thọ trung bình")
plt.ylabel("Số lượng mẫu")
plt.legend()
plt.show()

# ==========================================
# BƯỚC 9: Kiểm tra phân bố theo vùng, nhóm tuổi thọ và thời gian
# ==========================================
print("\n--- Phân bố theo vùng (Region) ---")
print("Train:\n", train_df['Region'].value_counts(normalize=True))
print("\nValid:\n", valid_df['Region'].value_counts(normalize=True))
print("\nTest:\n", test_df['Region'].value_counts(normalize=True))


print("\n--- Phân bố nhóm tuổi thọ (LifeGroup) ---")
print("Train:\n", train_df['LifeGroup'].value_counts(normalize=True))
print("\nValid:\n", valid_df['LifeGroup'].value_counts(normalize=True))
print("\nTest:\n", test_df['LifeGroup'].value_counts(normalize=True))

print("\n--- Phân bố theo giai đoạn (Period) ---")
print("Train:\n", train_df['Period'].value_counts(normalize=True))
print("\nValid:\n", valid_df['Period'].value_counts(normalize=True))
print("\nTest:\n", test_df['Period'].value_counts(normalize=True))




# ==========================================
# BƯỚC 10: Biểu đồ so sánh phân bố vùng
# ==========================================
region_counts = pd.DataFrame({
    "Train": train_df['Region'].value_counts(normalize=True),
    "Valid": valid_df['Region'].value_counts(normalize=True),
    "Test": test_df['Region'].value_counts(normalize=True)
}).fillna(0)

region_counts.plot(kind='bar', figsize=(12,6))
plt.title("So sánh phân bố vùng giữa 3 tập dữ liệu", fontsize=14)
plt.ylabel("Tỷ lệ (%)")
plt.xlabel("Region")
plt.show()







# # ==========================================
# # BƯỚC 11: Kiểm tra xem mỗi tập có đủ vùng + giai đoạn + nhóm tuổi thọ hay không
# # ==========================================
# def check_coverage(df, name):
#     coverage = df.groupby(['Region', 'Period', 'LifeGroup']).size().reset_index()
#     print(f"\n{name} có {coverage.shape[0]} tổ hợp (Region, Period, LifeGroup)")
#     print("Một vài ví dụ:")
#     print(coverage.head(10))

# check_coverage(train_df, "Train")
# check_coverage(valid_df, "Valid")
# check_coverage(test_df, "Test")