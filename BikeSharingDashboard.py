import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Bike Sharing Dashboard", page_icon="🚲", layout="wide")

@st.cache_data
def load_data():
    day_df = pd.read_csv("day.csv")
    hour_df = pd.read_csv("hour.csv")
    return day_df, hour_df

day_df, hour_df = load_data()

def main():
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Grafik Peminjaman Selama Setahun",
        "☁️ Pengaruh Cuaca",
        "🔮 Prediksi Peminjaman",
        "ℹ️ Kelompok"
    ])
    
    with tab1:
        grafik_mingguan()
    
    with tab2:
        pengaruh_cuaca()
    
    with tab3:
        prediksi_peminjaman()
    
    with tab4:
        about_me()

def grafik_mingguan():
    st.subheader("Grafik Peminjaman Sepeda Harian selama Setahun")
    day_df["weekday"] = pd.Categorical(day_df["weekday"], categories=[0,1,2,3,4,5,6], ordered=True)
    weekly_data = day_df.groupby("weekday")["cnt"].sum()
    
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(x=["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"], y=weekly_data.values, ax=ax)
    plt.ylabel("Jumlah Peminjaman")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    max_day = weekly_data.idxmax()
    days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    st.write(f"Hari dengan peminjaman terbanyak adalah **{days[max_day]}** dengan jumlah {weekly_data.max()} peminjaman.")

def pengaruh_cuaca():
    st.subheader("Pengaruh Cuaca terhadap Peminjaman Sepeda")
    weather_data = day_df.groupby("weathersit")["cnt"].mean()
    
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x=weather_data.index, y=weather_data.values, ax=ax)
    plt.xlabel("Kondisi Cuaca")
    plt.ylabel("Rata-rata Peminjaman Harian")
    plt.xticks(ticks=[0,1,2,3], labels=["Cerah", "Berawan", "Hujan Ringan", "Hujan Lebat"])
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    
    st.write("Dapat dilihat bahwa semakin buruk cuaca, semakin sedikit peminjaman sepeda yang terjadi.")

def prediksi_peminjaman():
    st.subheader("Prediksi Peminjaman Sepeda Berdasarkan Hari dan Cuaca")
    
    X = day_df[["weekday", "weathersit"]]
    y = day_df["cnt"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    day_options = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    weather_options = ["Cerah", "Berawan", "Hujan Ringan", "Hujan Lebat"]
    
    selected_day = st.selectbox("Pilih Hari", day_options)
    selected_weather = st.selectbox("Pilih Kondisi Cuaca", weather_options)
    
    pred = model.predict([[day_options.index(selected_day), weather_options.index(selected_weather)]])
    st.metric(label="Prediksi Jumlah Peminjaman", value=f"{int(pred[0])} Sepeda")

def about_me():
    st.write("### Kelompok 4")
    st.write("#### Anggota Kelompok:")
    st.write("- Afifah Nurfadhilah - 10123168")
    st.write("- Aulia Maysaroh Mardiansyah - 10123160")
    st.write("- Bintang Kurnia Saputra - 10123161")
    st.write("- Dimas Perkasa Agung Putra - 10123133")
    st.write("- Putu Adi Pratama - 10123137")
    st.write("- Riska Oktaviani - 10123142")


if __name__ == "__main__":
    main()
