import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import subprocess
import os
import csv

class CBIRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CBIR System")
        self.root.geometry("1400x900")
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', font=('Helvetica', 12), padding=10)
        self.style.configure('TLabel', font=('Helvetica', 12))
        self.style.configure('Header.TLabel', font=('Helvetica', 16, 'bold'))

        # Variables
        self.target_image_path = None
        self.target_image_display = None
        self.results = []
        self.feature_var = tk.StringVar(value="BaselineMatching")
        self.distance_var = tk.StringVar(value="SSD")
        self.n_matches_var = tk.StringVar(value="5")

        # Layout
        self.create_widgets()

    def create_widgets(self):
        # Left Panel: Controls and Target Image
        left_panel = ttk.Frame(self.root, padding="20")
        left_panel.pack(side=tk.LEFT, fill=tk.Y)

        # Header
        ttk.Label(left_panel, text="Control Panel", style='Header.TLabel').pack(pady=(0, 20))

        # Image Picker
        ttk.Button(left_panel, text="Select Target Image", command=self.select_image).pack(fill=tk.X, pady=5)
        
        self.img_label = ttk.Label(left_panel, text="No Image Selected")
        self.img_label.pack(pady=10)

        # Settings Group
        settings_frame = ttk.LabelFrame(left_panel, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=20)

        # Feature Set Dropdown
        ttk.Label(settings_frame, text="Feature Set:").pack(anchor=tk.W)
        features = ["BaselineMatching", "Histogram", "MultiHistogram", "ColorTexture", "DeepEmbeddings", "CustomTask7", "GaborTexture"]
        self.feature_combo = ttk.Combobox(settings_frame, textvariable=self.feature_var, values=features, state="readonly")
        self.feature_combo.pack(fill=tk.X, pady=5)
        self.feature_combo.bind("<<ComboboxSelected>>", self.update_options)

        # Distance Metric Dropdown
        ttk.Label(settings_frame, text="Distance Metric:").pack(anchor=tk.W)
        metrics = ["SSD", "Cosine", "Intersection"] # Will update dynamically
        self.dist_combo = ttk.Combobox(settings_frame, textvariable=self.distance_var, values=metrics, state="readonly")
        self.dist_combo.pack(fill=tk.X, pady=5)

        # Database File Picker
        ttk.Label(settings_frame, text="Database CSV File:").pack(anchor=tk.W)
        db_frame = ttk.Frame(settings_frame)
        db_frame.pack(fill=tk.X, pady=5)
        
        self.db_entry = ttk.Entry(db_frame)
        self.db_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.db_entry.insert(0, "BaselineDatabase.csv") # Default
        
        ttk.Button(db_frame, text="Browse", width=8, command=self.select_db_file).pack(side=tk.RIGHT, padx=(5, 0))

        # N Matches
        ttk.Label(settings_frame, text="Top N Matches:").pack(anchor=tk.W)
        ttk.Entry(settings_frame, textvariable=self.n_matches_var).pack(fill=tk.X, pady=5)

        # Run Button
        run_btn = ttk.Button(left_panel, text="Run Search", command=self.run_search)
        run_btn.pack(fill=tk.X, pady=20)
        
        # Quit Button
        ttk.Button(left_panel, text="Quit", command=self.root.quit).pack(fill=tk.X, side=tk.BOTTOM)

        # Right Panel: Results
        right_panel = ttk.Frame(self.root, padding="20")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(right_panel, text="Retrieval Results", style='Header.TLabel').pack(pady=(0, 20), anchor=tk.W)

        # Scrollable Canvas for Results
        self.canvas = tk.Canvas(right_panel)
        self.scrollbar = ttk.Scrollbar(right_panel, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def select_image(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd()+"/olympus", title="Select Image",
                                              filetypes=(("JPEG files", "*.jpg"), ("All files", "*.*")))
        if filename:
            self.target_image_path = filename
            self.load_image_preview(filename)

    def load_image_preview(self, path):
        try:
            img = Image.open(path)
            img.thumbnail((250, 250))
            self.target_image_display = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.target_image_display, text="")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def select_db_file(self):
        filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select Database CSV",
                                              filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if filename:
            self.db_entry.delete(0, tk.END)
            self.db_entry.insert(0, filename)

    def update_options(self, event=None):
        feature = self.feature_var.get()
        
        # Update Distance Metric Defaults
        if feature == "Histogram" or feature == "MultiHistogram":
             self.dist_combo['values'] = ["Intersection"]
             self.distance_var.set("Intersection")
        elif feature == "ResNet18" or feature == "DeepEmbeddings" or feature == "CustomTask7":
             self.dist_combo['values'] = ["Cosine"]
             self.distance_var.set("Cosine")
        else: # Baseline, Texture, Gabor, ColorTexture
             self.dist_combo['values'] = ["SSD"]
             self.distance_var.set("SSD")

        # Update Database File Defaults
        db_map = {
            "BaselineMatching": "BaselineDatabase.csv",
            "Histogram": "HistogramDB.csv",
            "MultiHistogram": "MultiHistDB.csv",
            "ColorTexture": "ColorTextureDB.csv",
            "DeepEmbeddings": "ResNet18_Embeddings.csv",
            "GaborTexture": "GaborDB.csv",
            "CustomTask7": "ResNet18_Embeddings.csv" # CustomTask7 uses two, but first is ResNet
        }
        
        if feature in db_map:
            self.db_entry.delete(0, tk.END)
            self.db_entry.insert(0, db_map[feature])

    def run_search(self):
        if not self.target_image_path:
            messagebox.showwarning("Warning", "Please select a target image first.")
            return

        feature_set = self.feature_var.get()
        metric = self.distance_var.get()
        n_matches = self.n_matches_var.get()
        
        # Determine strict path format required by Matcher logic (olympus/...)
        # But we pass absolute path, Matcher handles standard extraction by path.
        # For CSV lookups (DeepEmbeddings), we need to be careful.
        
        # Construct Command
        # ./Matcher <Target> <Feature> <Metric> <N> <CSV1> [CSV2]
        
        cmd = ["./Matcher", self.target_image_path, feature_set, metric, n_matches]
        
        # Use manually entered DB file
        csv_file = self.db_entry.get()
        if not csv_file:
             messagebox.showwarning("Warning", "Please select a database CSV file.")
             return
             
        cmd.append(csv_file)

        # Handle CustomTask7 which needs a SECOND CSV
        if feature_set == "CustomTask7":
            cmd.append("HSVSpatialDB.csv") # Second CSV is hardcoded for now or we could add another input

            
        # Execute
        try:
            # We need to capture stdout
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                messagebox.showerror("Error", f"Matcher failed:\n{result.stderr}")
                return
            
            self.parse_and_display_results(result.stdout)
            
        except Exception as e:
            messagebox.showerror("Execution Error", str(e))

    def parse_and_display_results(self, output):
        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
            
        # Parse Output
        # Expected format: "Rank: Path (Distance: X)" 
        # But Matcher.cpp output format is: "0: olympus/pic.xxxx.jpg (Distance: 0.123)"
        
        lines = output.split('\n')
        row = 0
        col = 0
        
        for line in lines:
            if "Distance:" in line and ".jpg" in line:
                # Simple parsing logic
                # Example: "0: olympus/pic.0535.jpg (Distance: 2.23517e-08)"
                try:
                    parts = line.split(":") 
                    # parts[0] is rank
                    # parts[1] contains " olympus/pic... (Distance"
                    
                    rest = parts[1].strip()
                    path_part = rest.split(" (")[0]
                    dist_part = rest.split("Distance: ")[1].replace(")", "")
                    
                    self.add_result_card(path_part, dist_part, row, col)
                    
                    col += 1
                    if col > 3: # 4 columns
                        col = 0
                        row += 1
                        
                except Exception as ex:
                    print(f"Skipping line: {line} due to {ex}")

    def add_result_card(self, image_path, distance, row, col):
        frame = ttk.Frame(self.scrollable_frame, relief="ridge", borderwidth=2, padding=5)
        frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
        
        # Fix path
        if "olympus/" not in image_path and "/" not in image_path:
             image_path = "olympus/" + image_path
        
        try:
            pil_img = Image.open(image_path)
            pil_img.thumbnail((200, 200))
            photo = ImageTk.PhotoImage(pil_img)
            
            lbl_img = tk.Label(frame, image=photo)
            lbl_img.image = photo # Keep reference
            lbl_img.pack()
            
            ttk.Label(frame, text=f"Dist: {distance[:6]}").pack()
            ttk.Label(frame, text=os.path.basename(image_path)).pack()
            
        except Exception as e:
            ttk.Label(frame, text=f"Img Error: {image_path}").pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = CBIRApp(root)
    root.mainloop()
