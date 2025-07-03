import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt


# 1. Load the Excel file
df = pd.read_excel("Telco_customer_churn.xlsx")

# 2. Clean column names: strip whitespace
df.columns = df.columns.str.strip().str.lower().str.replace(" ","_")

# 3. Drop irrelevant columns
columns_to_drop = [
    'customerid', 'count', 'country', 'lat_long', 'latitude', 'longitude',
    'churn_value', 'churn_score', 'cltv'
]
df.drop(columns=columns_to_drop, inplace=True)

# 4. Convert 'Total Charges' to numeric (coerce errors to NaN)
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
df["monthly_charges"]=pd.to_numeric(df["monthly_charges"], errors = "coerce")

# 5. Drop rows with missing 'Total Charges'
df.dropna(subset=["total_charges", "monthly_charges"], inplace=True)

# 6. Remove any duplicate rows (if they exist)
df.drop_duplicates(inplace=True)

# 7. Strip whitespace from string cells
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# 8. Create 'Tenure Group' column (for analysis buckets)
def group_tenure(months):
    if months <= 12:
        return "0–1 year"
    elif months <= 24:
        return "1–2 years"
    elif months <= 48:
        return "2–4 years"
    else:
        return "4+ years"

df["tenure_group"] = df["tenure_months"].apply(group_tenure)

# 9. Rename columns (optional, cleaner names for analysis)
df.columns = df.columns.str.lower().str.replace(" ", "_")

churn_by_tenure = df.groupby("tenure_group")["churn_label"].value_counts(normalize=True).unstack() * 100
print(churn_by_tenure,)

churn_by_contract = df.groupby("contract")["churn_label"].value_counts(normalize = True).unstack() * 100
print(churn_by_contract)


# Churn by contract

sns.countplot(data = df, x ="contract",hue="churn_label",  palette = "Set2")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Number of Customers")
plt.xlabel("Contract Type")
plt.savefig("churn_by_contract.png", dpi=300, bbox_inches="tight")

#plt.show()

#Churn by tenure group

sns.countplot(data=df, x="tenure_group", hue="churn_label", palette="Set2")
plt.title("Churn by Tenure Group")
plt.ylabel("Number of Customers")
plt.xlabel("Tenure Group")
plt.savefig("churn_by_tenure.png", dpi=300, bbox_inches="tight")

plt.show()

#Churn by monthly charge group

# Create grouped charge ranges
bins = [0, 30, 60, 90, 120]
labels = ["$0–30", "$31–60", "$61–90", "$91–120"]
df["charge_group"] = pd.cut(df["monthly_charges"], bins=bins, labels=labels)


sns.countplot(data=df, x="charge_group", hue="churn_label", palette="Set2")
plt.title("Churn by Monthly Charge Group")
plt.xlabel("Monthly Charge Range")
plt.ylabel("Number of Customers")
plt.xticks(rotation=30) 

plt.tight_layout()
plt.savefig("churn_by_monthly_charge_group.png", dpi=300)
plt.show()





bins = [0, 1000, 2000, 4000, 6000, 8000, 10000]
labels = ["0-1k","1k-2k","2k-4k","4k-6k","6k-8k", "8k-10k"]
df["total_charge_group"] = pd.cut(df["total_charges"], bins=bins, labels = labels)


#Churn by total charge group

sns.countplot(data=df, x="total_charge_group", hue="churn_label", palette="Set2")
plt.title("Churn by Total Charges")
plt.xlabel("Total Charges Range")
plt.ylabel("Number of Customers")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("churn_by_total_charges.png", dpi=300)
plt.show()

# Churn by internet service

sns.countplot(data=df, x = "internet_service", hue = "churn_label", palette="Set2")
plt.title("Churn By Internet Service")
plt.xlabel("Internet Service Type")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("Churn_by_internet_service.png", dpi=300)
plt.show()

#Churn by Tech Support

sns.countplot(data=df, x="tech_support", hue="churn_label", palette="Set1")
plt.title("Churn by Tech Support")
plt.xlabel("Tech Support")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("churn_by_tech_support.png", dpi=300)
plt.show()

#Churn by Payment Method
sns.countplot(data=df,  x="payment_method",hue = "churn_label", palette="Set3")
plt.title("Churn by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of customers")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("churn_by_payment_method.png", dpi=300)
plt.show()

# 10. Save cleaned version to CSV
#df.to_csv("telco_churn_cleaned.csv", index=False)

#Analysis of churn by total charges reveals that customers who churn tend to have accrued low total charges — primarily under $1,000. This suggests early churn behavior, where customers are leaving within their first few months or billing cycles, possibly due to poor onboarding, unmet expectations, or lack of perceived value early on.”

# Confirm success
print("✅ Data cleaned and exported successfully.")







