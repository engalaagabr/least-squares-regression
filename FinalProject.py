import tkinter as tk
from tkinter import messagebox
from tkinter import ttk as tkttk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class LeastSquaresApp:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 Least Squares Linear Regression")
        self.root.state('zoomed')  # Make fullscreen
        self.style = ttk.Style("cosmo")

        self.x_entries = []
        self.y_entries = []

        self.setup_ui()

    def setup_ui(self):
        # Header Buttons Frame
        header_frame = ttk.Frame(self.root)
        header_frame.pack(anchor='ne', padx=10, pady=10)

        ttk.Button(header_frame, text="❓ Help", command=self.show_help, bootstyle="info-outline", width=20).grid(row=0, column=0, padx=10, pady=5)
        ttk.Button(header_frame, text="ℹ️ About", command=self.show_about, bootstyle="warning-outline", width=20).grid(row=0, column=1, padx=10, pady=5)

        # Title
        ttk.Label(self.root, text="Enter Data Points", font=("Segoe UI", 22, "bold")).pack(pady=(30, 20))

        # Table Frame
        self.table_frame = ttk.Frame(self.root)
        self.table_frame.pack()

        headers = ["Index", "xi", "yi"]
        for col, header in enumerate(headers):
            ttk.Label(self.table_frame, text=header, font=("Segoe UI", 14, "bold")).grid(row=0, column=col, padx=15, pady=10)

        self.add_row()

        # Buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=30)

        ttk.Button(btn_frame, text="➕ Add Row", command=self.add_row, bootstyle="success-outline", width=20).grid(row=0, column=0, padx=15)
        ttk.Button(btn_frame, text="❌ Delete Last Row", command=self.delete_row, bootstyle="danger-outline", width=20).grid(row=0, column=1, padx=15)
        ttk.Button(btn_frame, text="✅ Calculate", command=self.calculate, bootstyle="primary-outline", width=20).grid(row=0, column=2, padx=15)

        # Result Label
        self.result_label = ttk.Label(self.root, text="", font=("Segoe UI", 14), bootstyle="dark")
        self.result_label.pack(pady=15)

        # Plot Area
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(pady=20)

    def add_row(self):
        index = len(self.x_entries) + 1
        ttk.Label(self.table_frame, text=str(index), font=("Segoe UI", 12)).grid(row=index, column=0, padx=10, pady=5)
        x_entry = ttk.Entry(self.table_frame, width=15, font=("Segoe UI", 12))
        y_entry = ttk.Entry(self.table_frame, width=15, font=("Segoe UI", 12))
        x_entry.grid(row=index, column=1, padx=10, pady=5)
        y_entry.grid(row=index, column=2, padx=10, pady=5)
        self.x_entries.append(x_entry)
        self.y_entries.append(y_entry)

    def delete_row(self):
        if self.x_entries:
            self.x_entries[-1].destroy()
            self.y_entries[-1].destroy()
            self.x_entries.pop()
            self.y_entries.pop()
            self.table_frame.grid_slaves(row=len(self.x_entries)+1, column=0)[0].destroy()

    def calculate(self):
        try:
            xi = [float(entry.get()) for entry in self.x_entries]
            yi = [float(entry.get()) for entry in self.y_entries]
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for all xi and yi.")
            return

        if len(xi) < 2:
            messagebox.showwarning("Insufficient Data", "Enter at least two data points.")
            return

        n = len(xi)
        sum_x = sum(xi)
        sum_y = sum(yi)
        sum_x2 = sum(x ** 2 for x in xi)
        sum_xy = sum(x * y for x, y in zip(xi, yi))

        a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        b = (sum_y - a * sum_x) / n

        rmse = np.sqrt(np.mean([(yi[i] - (a * xi[i] + b)) ** 2 for i in range(n)]))

        self.result_label.config(text=f"📈 Best Fit Line:  y = {a:.4f}x + {b:.4f}     |     RMSE = {rmse:.4f}")
        self.plot_regression(xi, yi, a, b)

    def plot_regression(self, xi, yi, a, b):
        self.ax.clear()
        x = np.array(xi)
        y = np.array(yi)
        y_pred = a * x + b

        self.ax.scatter(x, y, color="blue", label="Data Points")
        self.ax.plot(x, y_pred, color="red", label=f"y = {a:.2f}x + {b:.2f}")
        self.ax.set_title("Least Squares Regression Line")
        self.ax.set_xlabel("xi")
        self.ax.set_ylabel("yi")
        self.ax.legend()
        self.ax.grid(True)

        self.canvas.draw()

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - How to Use")
        help_window.geometry("600x400")
        ttk.Label(help_window, text="How to Use the App", font=("Segoe UI", 16, "bold")).pack(pady=10)
        text = (
            "1. Click 'Add Row' to input more data points.\n"
            "2. Enter numeric xi and yi values.\n"
            "3. Click 'Calculate' to get the best-fit line.\n"
            "4. View the results and graph below.\n"
            "5. Use 'Delete Last Row' to remove inputs."
        )
        text_box = tk.Text(help_window, wrap="word", font=("Segoe UI", 12))
        text_box.insert("1.0", text)
        text_box.config(state="disabled")
        text_box.pack(padx=20, pady=10, expand=True, fill="both")

    def show_about(self):
        about_window = tk.Toplevel(self.root)
        about_window.title("About - GlitchWave Team")
        about_window.geometry("700x500")
        ttk.Label(about_window, text="About us", font=("Segoe UI", 16, "bold")).pack(pady=10)
        about_text = (
            "📘 Team: GlitchWave\n"
            "👥 Members:\n"
            "  - Alaa Gabr\n"
            "  - Alaa Abo Elfadl\n"
            "  - Abdelrahman Sherif\n"
            "  - Omr Eladly\n"
            "  - Omr Abdelaziz\n\n"
            "⚙️ Project Goal:\n"
            "To solve real-world data fitting using the Least Squares method.\n\n"
            "⚙️ Challenges Faced:\n"
            "- Building dynamic and editable UI\n"
            "- Handling data validation\n"
            "- Integrating graphs smoothly\n\n"
            "💡 Tools Used:\n"
            "Tkinter, ttkbootstrap, Matplotlib, and NumPy"
        )
        text_box = tk.Text(about_window, wrap="word", font=("Segoe UI", 12))
        text_box.insert("1.0", about_text)
        text_box.config(state="disabled")
        text_box.pack(padx=20, pady=10, expand=True, fill="both")

# Run the app
if __name__ == "__main__":
    root = ttk.Window(themename="cosmo")
    app = LeastSquaresApp(root)
    root.mainloop()
