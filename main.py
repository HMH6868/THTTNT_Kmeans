import tkinter as tk
from tkinter import filedialog, ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import seaborn as sns


class CustomerSegmentationApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Ứng Dụng Phân Cụm Dữ Liệu Khách Hàng")
        self.master.geometry("600x800")
        self.master.config(bg="#f4f4f9")
        
        # Dữ liệu
        self.df = None
        self.scaled_data = None
        self.clusters = None
        self.kmeans = None

        # Tiêu đề
        title_label = tk.Label(self.master, text="Phân Tích Dữ Liệu Khách Hàng", font=("Arial", 18, "bold"), bg="#f4f4f9", fg="blue")
        title_label.pack(pady=20)

        # Mô tả
        desc_label = tk.Label(self.master, text="Chọn file Excel chứa dữ liệu khách hàng để phân tích", font=("Arial", 12), bg="#f4f4f9")
        desc_label.pack(pady=10)

        # Các nút chức năng
        buttons = [
            ("Import Excel", self.import_excel),
            ("Biểu đồ 2D Scatter", self.plot_2d_scatter),
            ("Biểu đồ Heatmap", self.plot_heatmap),
            ("Silhouette Score", self.plot_silhouette),
            ("Biểu đồ t-SNE", self.plot_tsne),
            ("Biểu đồ PCA 3D", self.plot_pca_3d),
            ("Biểu đồ Phân Cụm 3D", self.plot_3d_kmeans),
            ("Biểu đồ Cột Giới Tính", self.plot_bar_gender),
            ("Biểu đồ Phân Phối", self.plot_distribution),
            ("Biểu đồ Pairplot", self.plot_pairplot),
            ("Biểu đồ Scatter Giới Tính", self.plot_scatter_by_gender),
            ("Biểu đồ Cụm K-Means", self.plot_cluster_income_vs_spending),
            ("Xuất dữ liệu", self.export_results),
            ("Nhập dữ liệu", self.predict_customer_type)
        ]

        for text, command in buttons:
            button = ttk.Button(self.master, text=text, command=command, width=30)
            button.pack(pady=5)

        # Style nút
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), background="#007bff", foreground="black", padding=10)
        style.map("TButton", background=[('active', '#0056b3')])

    def import_excel(self):
        file_path = filedialog.askopenfilename(title="Chọn file Excel", filetypes=[("Excel files", "*.xlsx")])
        if not file_path:
            return
        self.df = pd.read_excel(file_path)
        print("Dữ liệu từ file Excel:")
        print(self.df.head())

        # Lọc và chuẩn hóa dữ liệu
        try:
            df_filtered = self.df[['Tuổi', 'Thu nhập cá nhân (VND)', 'Điểm chi tiêu (1-100)']]
            scaler = StandardScaler()
            self.scaled_data = scaler.fit_transform(df_filtered)

            # Áp dụng KMeans với 3 cụm
            self.kmeans = KMeans(n_clusters=3, random_state=0)
            self.clusters = self.kmeans.fit_predict(self.scaled_data)
            self.df['Cluster'] = self.clusters

            # Gắn nhãn cụm
            cluster_labels = {0: "Thấp", 1: "Trung bình", 2: "Tiềm năng"}
            self.df['Phân loại'] = self.df['Cluster'].map(cluster_labels)

            print("Dữ liệu sau khi phân cụm:")
            print(self.df.head())
        except KeyError:
            print("Dữ liệu thiếu các cột cần thiết!")



    def plot_2d_scatter(self):
        if self.scaled_data is None or self.clusters is None:
            print("Chưa có dữ liệu phân cụm!")
            return
        plt.scatter(self.scaled_data[:, 0], self.scaled_data[:, 1], c=self.clusters, cmap='viridis')
        plt.xlabel('Tuổi')
        plt.ylabel('Thu nhập cá nhân (VND)')
        plt.title('Biểu đồ Phân Cụm 2D')
        plt.show()

    def plot_heatmap(self):
        if self.df is None:
            print("Chưa có dữ liệu!")
            return
        sns.heatmap(self.df[['Tuổi', 'Thu nhập cá nhân (VND)', 'Điểm chi tiêu (1-100)']].corr(), annot=True, cmap='coolwarm')
        plt.title('Ma trận Tương Quan giữa các Đặc Trưng')
        plt.show()

    def plot_silhouette(self):
        if self.scaled_data is None or self.clusters is None:
            print("Chưa có dữ liệu phân cụm!")
            return
        silhouette_avg = silhouette_score(self.scaled_data, self.clusters)
        print(f"Silhouette Score: {silhouette_avg}")

    def plot_tsne(self):
        if self.scaled_data is None:
            print("Chưa có dữ liệu phân cụm!")
            return
        tsne = TSNE(n_components=2)
        tsne_result = tsne.fit_transform(self.scaled_data)
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=self.clusters, cmap='viridis')
        plt.title('Biểu đồ t-SNE')
        plt.show()

    def plot_pca_3d(self):
        if self.scaled_data is None:
            print("Chưa có dữ liệu phân cụm!")
            return
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(self.scaled_data)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=self.clusters, cmap='viridis')
        ax.set_title('Biểu đồ PCA 3D')
        plt.show()

    def plot_3d_kmeans(self):
        if self.scaled_data is None or self.clusters is None or self.kmeans is None:
            print("Chưa có dữ liệu phân cụm!")
            return

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Danh sách màu sắc và nhãn cụm
        cluster_colors = {0: 'red', 1: 'blue', 2: 'green'}
        cluster_labels = {0: "Thấp", 1: "Trung bình", 2: "Tiềm năng"}

        # Vẽ từng cụm với nhãn tương ứng
        for cluster_id, color in cluster_colors.items():
            cluster_points = self.scaled_data[self.clusters == cluster_id]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], 
                    s=100, c=color, label=f'Cụm {cluster_labels[cluster_id]}')

        # Vẽ các tâm cụm
        cluster_centers = self.kmeans.cluster_centers_
        ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], cluster_centers[:, 2], 
                s=300, c='yellow', label='Tâm Cụm', marker='X')

        # Thêm nhãn và tiêu đề
        ax.set_xlabel('Tuổi')
        ax.set_ylabel('Thu nhập cá nhân (Chuẩn hóa)')
        ax.set_zlabel('Điểm chi tiêu (Chuẩn hóa)')
        ax.set_title('Phân Cụm 3D: Thấp, Trung bình, Tiềm năng')

        # Hiển thị chú thích
        ax.legend()

        # Hiển thị đồ thị
        plt.show()


    def plot_bar_gender(self):
        if self.df is None or 'Giới tính' not in self.df.columns:
            print("Chưa có dữ liệu!")
            return
        gender_counts = self.df['Giới tính'].value_counts()
        plt.bar(gender_counts.index, gender_counts.values, color=['blue', 'pink'], edgecolor='black')
        plt.title('Biểu đồ Cột: Số lượng theo Giới tính')
        plt.show()

    def plot_distribution(self):
        if self.df is None:
            print("Chưa có dữ liệu!")
            return
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        self.df[['Tuổi', 'Thu nhập cá nhân (VND)', 'Điểm chi tiêu (1-100)']].hist(ax=axes, bins=10, edgecolor='black')
        plt.tight_layout()
        plt.show()

    def plot_pairplot(self):
        if self.df is None:
            print("Chưa có dữ liệu!")
            return
        sns.pairplot(self.df, diag_kind="kde", hue='Cluster', vars=['Tuổi', 'Thu nhập cá nhân (VND)', 'Điểm chi tiêu (1-100)'])
        plt.show()


    def plot_scatter_by_gender(self):
        if self.df is None or 'Giới tính' not in self.df.columns:
            print("Chưa có dữ liệu!")
            return
        plt.figure(figsize=(10, 6))
        for gender in self.df['Giới tính'].unique():
            subset = self.df[self.df['Giới tính'] == gender]
            plt.scatter(subset['Tuổi'], subset['Thu nhập cá nhân (VND)'], label=gender)
        plt.legend()
        plt.title('Scatter Plot: Giới Tính')
        plt.show()

    def plot_cluster_income_vs_spending(self):
        if self.df is None or self.clusters is None:
            print("Chưa có dữ liệu phân cụm!")
            return
        plt.scatter(self.df['Thu nhập cá nhân (VND)'], self.df['Điểm chi tiêu (1-100)'], c=self.clusters, cmap='viridis')
        plt.title('Cluster Plot: Income vs Spending')
        plt.show()

    def export_results(self):
        if self.df is None or 'Cluster' not in self.df.columns:
            print("Chưa có dữ liệu để xuất!")
            return

        # Tạo nhãn cụm
        cluster_labels = {0: "Thấp", 1: "Trung bình", 2: "Tiềm năng"}
        self.df['Phân loại'] = self.df['Cluster'].map(cluster_labels)

        # Chọn các cột cần xuất
        columns_to_export = ["Mã khách hàng", "Giới tính", "Tuổi", "Thu nhập cá nhân (VND)", "Điểm chi tiêu (1-100)", "Phân loại"]
        data_to_export = self.df[columns_to_export]

        # Mở hộp thoại lưu file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Lưu file kết quả"
        )
        if not file_path:
            return

        # Xuất dữ liệu ra file Excel
        try:
            data_to_export.to_excel(file_path, index=False)
            print(f"Kết quả đã được lưu tại: {file_path}")
        except Exception as e:
            print(f"Đã xảy ra lỗi khi lưu file: {e}")

    def predict_customer_type(self):
        # Tạo cửa sổ mới để nhập dữ liệu
        predict_window = tk.Toplevel(self.master)
        predict_window.title("Dự Đoán Loại Khách Hàng")
        predict_window.geometry("400x400")
        predict_window.config(bg="#f4f4f9")

        # Tạo nhãn và trường nhập liệu
        labels = ["Giới tính (Nam/Nữ)", "Tuổi", "Thu nhập cá nhân (VND)", "Điểm chi tiêu (1-100)"]
        entries = {}

        for idx, label in enumerate(labels):
            lbl = tk.Label(predict_window, text=label, font=("Arial", 12), bg="#f4f4f9")
            lbl.pack(pady=5)
            entry = tk.Entry(predict_window, font=("Arial", 12))
            entry.pack(pady=5)
            entries[label] = entry

        # Hàm xử lý dự đoán
        def process_prediction():
            try:
                # Lấy dữ liệu nhập vào
                gender = entries["Giới tính (Nam/Nữ)"].get().strip().lower()
                age = int(entries["Tuổi"].get())
                income = float(entries["Thu nhập cá nhân (VND)"].get())
                spending_score = int(entries["Điểm chi tiêu (1-100)"].get())

                # Kiểm tra giá trị hợp lệ
                if gender not in ['nam', 'nữ']:
                    result_label.config(text="Giới tính phải là Nam hoặc Nữ!", fg="red")
                    return

                # Chuẩn hóa dữ liệu
                input_data = [[age, income, spending_score]]
                scaler = StandardScaler()
                scaled_input = scaler.fit_transform(self.df[['Tuổi', 'Thu nhập cá nhân (VND)', 'Điểm chi tiêu (1-100)']])
                input_data_scaled = scaler.transform(input_data)

                # Dự đoán cụm
                cluster = self.kmeans.predict(input_data_scaled)[0]

                # Map cụm sang loại khách hàng
                cluster_labels = {0: "Thấp", 1: "Trung bình", 2: "Tiềm năng"}
                customer_type = cluster_labels.get(cluster, "Không xác định")

                # Hiển thị kết quả
                result_label.config(text=f"Loại khách hàng: {customer_type}", fg="green")
            except Exception as e:
                result_label.config(text=f"Lỗi: {e}", fg="red")

        # Nút dự đoán
        predict_button = tk.Button(predict_window, text="Dự Đoán", font=("Arial", 12), command=process_prediction)
        predict_button.pack(pady=20)

        # Nhãn hiển thị kết quả
        result_label = tk.Label(predict_window, text="", font=("Arial", 12, "bold"), bg="#f4f4f9")
        result_label.pack(pady=10)


# Chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = CustomerSegmentationApp(root)
    root.mainloop()
