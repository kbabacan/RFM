import  pandas as pd
import datetime as date
import numpy as np
import datetime as dt


df_=pd.read_csv("Data/flo_data_20k.csv")
pd.set_option('display.width', 500)
df=df_.copy()

#### ilk 10 gözlem
df.head(10)

#### değişken isimleri
df.columns

### betimsel istatistik
df.describe().T

###  boş değer
df.isnull().values.any()

###değişken tipleri
df.info()

### her müşterinin toplam alışveriş sayısı
df["Toplam_alisveris_sayisi"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df["Toplam_alisveris_tutarı"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.head()

### Değişkenleri incele tarih olanları date yap
df.info()
new_date_columns = ["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]

for column in new_date_columns:
    df[column]=pd.to_datetime(df[column])

date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns]= df[date_columns].apply(pd.to_datetime)


### Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız
df.groupby("order_channel").agg({"master_id" : "count",
                                "Toplam_alisveris_sayisi" : "sum",
                                "Toplam_alisveris_tutarı" :"sum"})


##en fazla kazanc getiren müşteri
df.sort_values(
    by="Toplam_alisveris_tutarı",
    ascending=False).head(10)


## en fazla sipariş veren
df.sort_values(
    by="Toplam_alisveris_sayisi",
    ascending=False).head(10)


df.groupby("master_id").agg({"Toplam_alisveris_tutarı": "sum"}).sort_values(by="Toplam_alisveris_tutarı",ascending=False).head(10)

### Veri önhazırlıksürecinifonksiyonlaştırınız
def dataprepare(veri):
    #### ilk 10 gözlem
    df.head(10)

    #### değişken isimleri
    df.columns

    ### betimsel istatistik

    ###  boş değer
    df.isnull().values.any()

    ###değişken tipleri
    df.info()

    ### her müşterinin toplam alışveriş sayısı
    df["Toplam_alisveris_sayisi"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
    df["Toplam_alisveris_tutarı"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    df.head()

    ### Değişkenleri incele tarih olanları date yap
    df.info()
    new_date_columns = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]

    ### Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız
    df.groupby("order_channel").agg({"master_id": "nunique",
                                     "Toplam_alisveris_sayisi": "sum",
                                     "Toplam_alisveris_tutarı": "sum"})

    ##en fazla kazanc getiren müşteri
    df.sort_values(
        by="Toplam_alisveris_tutarı",
        ascending=False).head(10)

    ## en fazla sipariş veren
    a=df.sort_values(
        by="Toplam_alisveris_sayisi",
        ascending=False).head(10)

    return a

prepareRFM=dataprepare(df)


### Recency, Frequency ve Monetary
## Recency == Analiz tarihi - son alışveriş tarihi
## Frequency  == müşterinin alışveriş sıklığı
## Monetary == müşterinin toplam alışveriş tutarı


### Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız
analiz_tarihi = df["last_order_date"].max() + pd.DateOffset(days=2)
today_date = date.datetime(2021,6,1)
type(today_date)

df.head()

rfm = df.groupby("master_id").agg({"last_order_date": lambda last_order_date : (today_date - last_order_date.max()).days,
                             "Toplam_alisveris_sayisi" : lambda Toplam_alisveris_sayısı : Toplam_alisveris_sayısı.sum() ,
                             "Toplam_alisveris_tutarı" : lambda Toplam_alisveris_tutarı: Toplam_alisveris_tutarı.sum() })

rfm.columns = ["Recency","Frequency","Monetary"]

rfm.describe().T


rfm["Recency_score"] = pd.qcut(rfm["Recency"],5,[5,4,3,2,1])

rfm["Frequency_score"] = pd.qcut(rfm["Frequency"].rank(method="first"),5,[1,2,3,4,5])

rfm["Monetary_score"] = pd.qcut(rfm["Monetary"],5,[1,2,3,4,5])

rfm["RFM_SCORE"] = (rfm["Recency_score"].astype(str) + rfm["Frequency_score"].astype(str))


rfm.head()

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}


rfm["Segment"] = rfm["RFM_SCORE"].replace(seg_map,regex=True)
rfm[["Recency","Frequency","Monetary","Segment"]].groupby("Segment").agg(["mean","count"])

new_df=pd.DataFrame()

new_df["new_cust_id"] = rfm[rfm["Segment"] == "champions"].index

rfm[rfm["Segment"] == "champions"].head()

df.head()

### KADIN VE SEGMENTI CHAMPIONS LOYAL OLAN MUSTERILERIN CIKTISI

merged_df = pd.merge(df, rfm, on="master_id") #iki ayrı veri tabanındaki veriyi birleştirdim.

targeted_customers = merged_df[(merged_df["Segment"].isin(["champions", "loyal_customers"]) &
                      merged_df["interested_in_categories_12"].str.contains("KADIN"))]


##Melisaya sorun :)
loyal_and_female_shop = rfm.loc[(rfm["Segment"] == 'loyal_customers') & df["interested_in_categories_12"].str.contains("KADIN"), "master_id"]

targeted_customers["master_id"].reset_index()

targeted_customers["master_id"].to_csv("targeted_customers.csv")

## ERKEK VE COCUK OLAN VE SEGMENTI NEW CUSTOMERS VE ABOUT TO SLEEP OLANLAR

rfm.head()
df.head()
rfm["Segment"].value_counts()

target_customers2 = merged_df[
    (merged_df["Segment"].isin(["new_customers", "about_to_sleep"])) &
    (merged_df["interested_in_categories_12"].str.contains("ERKEK|COCUK", case=False))
]

target_customers2
print(target_customers2["master_id"].reset_index()) #947 müşteri

target_customers2["master_id"].to_csv("target_customers2.csv")



