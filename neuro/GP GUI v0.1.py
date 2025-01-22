import sys
import os
import signal
import nibabel as nib
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLineEdit, QFileDialog, QLabel,
    QMenuBar, QStatusBar, QProgressBar, QDockWidget, QSlider, QCheckBox, QGroupBox, QFormLayout, QGraphicsView
)
from tools import perform_glm_and_plot_z_map, extract_and_plot_timeseries, plot_residuals_histogram, plot_r_squared_map, perform_ftest_and_plot_f_map

# Redirect stderr to suppress macOS IMK logs
sys.stderr = open(os.devnull, 'w')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Window title and size
        self.setWindowTitle("Neuro Psychic Analysis")
        self.setGeometry(100, 100, 1500, 900)
        self.setWindowIcon(QIcon("Logo_Two.png"))

        # Central Widget Layout
        central_widget = QWidget()
        central_layout = QHBoxLayout()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        # Left Toolbox
        self.create_left_toolbox()

        # Main Content Area
        self.create_main_content_area()

        # Right Toolbox
        self.create_right_toolbox()

        # Top Toolbar and Menu
        self.create_top_toolbar()

        # Bottom Status Bar and Progress
        self.create_bottom_status_bar()

        # Initialize attributes
        self.fmri_glm = None
        self.z_map = None
        self.masker = None
        self.coords = None

    def create_left_toolbox(self):
        # Left Toolbox Layout
        self.left_toolbox = QDockWidget("Project Management", self)
        self.left_toolbox.setStyleSheet("background-color: #1E1E1E; padding: 20px; border-radius: 10px; color: #747474;")
        self.left_toolbox.setMinimumWidth(300)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        self.header_label = QLabel("Create New Project")
        self.header_label.setStyleSheet("font-size: 18px; color: #747474; font-weight: bold; margin-bottom: 20px;")
        left_layout.addWidget(self.header_label)

        # File Input
        browse_button = QPushButton("Browse")
        browse_button.setStyleSheet("""
            QPushButton {
                background-color: #bdc3c7;
                color: white;
                padding: 12px;
                border-radius: 8px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)
        browse_button.clicked.connect(self.browse_file)
        load_button = QPushButton("Load Data")
        load_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                padding: 12px;
                border-radius: 8px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        load_button.clicked.connect(self.load_data)

        # Adding the file input and buttons
        left_layout.addWidget(browse_button)
        left_layout.addWidget(load_button)

        # Creating a group for analysis settings
        self.create_analysis_settings_group(left_layout)

        # Setting the widget for the left toolbox
        self.left_toolbox.setWidget(left_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_toolbox)

    def create_analysis_settings_group(self, layout):
        # Analysis Settings Group
        settings_group = QGroupBox("Analysis Settings", self)
        settings_group.setStyleSheet("""
            background-color: #1E1E1E; 
            border-radius: 10px;
            padding: 10px;
        """)
        settings_layout = QFormLayout()

        # Adding a slider for adjusting analysis speed
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(50)
        self.speed_slider.setStyleSheet("""
            background-color: #1E1E1E;
            border-radius: 10px;
        """)
        settings_layout.addRow("Analysis Speed", self.speed_slider)

        # Adding checkboxes for analysis options
        self.option_check1 = QCheckBox("Option 1")
        self.option_check1.setStyleSheet("""
            color: white;
            font-size: 16px;
        """)
        settings_layout.addRow(self.option_check1)

        self.option_check2 = QCheckBox("Option 2")
        self.option_check2.setStyleSheet("""
            color: white;
            font-size: 16px;
        """)
        settings_layout.addRow(self.option_check2)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

    def create_right_toolbox(self):
        # Right Toolbox Layout
        self.right_toolbox = QDockWidget("Toolbox", self)
        self.right_toolbox.setStyleSheet(
            "background-color: #1E1E1E; padding: 20px; border-radius: 10px; color: #747474")
        self.right_toolbox.setMinimumWidth(300)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)

        self.right_header_label = QLabel("Analysis Tools")
        self.right_header_label.setStyleSheet(
            "font-size: 18px; color: #747474; font-weight: bold; margin-bottom: 18px;")
        right_layout.addWidget(self.right_header_label)

        # Add buttons for each function
        self.add_function_buttons(right_layout)

        self.right_toolbox.setWidget(right_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.right_toolbox)

    def add_function_buttons(self, layout):
        # Button for perform_glm_and_plot_z_map
        glm_z_map_button = QPushButton("Perform GLM and Plot Z-Map")
        glm_z_map_button.setStyleSheet(self.get_button_style())
        glm_z_map_button.clicked.connect(self.perform_glm_and_plot_z_map)
        layout.addWidget(glm_z_map_button)

        # Button for extract_and_plot_timeseries
        timeseries_button = QPushButton("Extract and Plot Timeseries")
        timeseries_button.setStyleSheet(self.get_button_style())
        timeseries_button.clicked.connect(self.extract_and_plot_timeseries)
        layout.addWidget(timeseries_button)

        # Button for plot_residuals_histogram
        residuals_button = QPushButton("Plot Residuals Histogram")
        residuals_button.setStyleSheet(self.get_button_style())
        residuals_button.clicked.connect(self.plot_residuals_histogram)
        layout.addWidget(residuals_button)

        # Button for plot_r_squared_map
        r_squared_button = QPushButton("Plot R-Squared Map")
        r_squared_button.setStyleSheet(self.get_button_style())
        r_squared_button.clicked.connect(self.plot_r_squared_map)
        layout.addWidget(r_squared_button)

        # Button for perform_ftest_and_plot_f_map
        ftest_button = QPushButton("Perform F-Test and Plot F-Map")
        ftest_button.setStyleSheet(self.get_button_style())
        ftest_button.clicked.connect(self.perform_ftest_and_plot_f_map)
        layout.addWidget(ftest_button)

    def get_button_style(self):
        return """
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 12px;
                border-radius: 8px;
                font-size: 16px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """

    # Define the functions that will be called when buttons are clicked
    def perform_glm_and_plot_z_map(self):
        self.fmri_glm, self.z_map = perform_glm_and_plot_z_map()

    def extract_and_plot_timeseries(self):
        if self.fmri_glm is None or self.z_map is None:
            print("Error: Please run 'Perform GLM and Plot Z-Map' first.")
            return
        self.masker, self.coords = extract_and_plot_timeseries(self.fmri_glm, self.z_map)

    def plot_residuals_histogram(self):
        if self.masker is None or self.fmri_glm is None or self.coords is None:
            print("Error: Please run 'Extract and Plot Timeseries' first.")
            return
        plot_residuals_histogram(self.masker, self.fmri_glm, self.coords)

    def plot_r_squared_map(self):
        if self.fmri_glm is None:
            print("Error: Please run 'Perform GLM and Plot Z-Map' first.")
            return
        plot_r_squared_map(self.fmri_glm)

    def perform_ftest_and_plot_f_map(self):
        if self.fmri_glm is None:
            print("Error: Please run 'Perform GLM and Plot Z-Map' first.")
            return
        perform_ftest_and_plot_f_map(self.fmri_glm)

    def create_main_content_area(self):
        # Main Content Area
        self.main_area = QWidget()
        self.main_area.setStyleSheet("background-color: #1E1E1E; padding: 20px; border-radius: 10px;")
        self.main_area.setMinimumWidth(1000)
        self.main_area.setMinimumHeight(700)

        main_layout = QVBoxLayout()
        self.main_area.setLayout(main_layout)

        # Placeholder for Graph or fMRI Scan
        self.create_graph_area(main_layout)

        self.setCentralWidget(self.main_area)

    def create_graph_area(self, layout):
        # Placeholder area for displaying graph or fMRI scan
        self.graph_area_label = QLabel("fMRI Analysis Results")
        self.graph_area_label.setStyleSheet("font-size: 20px; color: white; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(self.graph_area_label)

        # Graph/Results Placeholder (Use QGraphicsView for interactive graphs in future)
        self.graph_view = QGraphicsView()
        self.graph_view.setStyleSheet("""
            background-color: #1e1e1e; 
            border-radius: 10px; 
            margin-top: 20px;
        """)
        layout.addWidget(self.graph_view)

    def create_top_toolbar(self):
        # Top Menu Bar with File, Edit, Tools, etc.
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        edit_menu = menubar.addMenu("Edit")
        tools_menu = menubar.addMenu("Tools")
        help_menu = menubar.addMenu("Help")

        # File Menu Actions
        open_action = file_menu.addAction("Open...")
        open_action.triggered.connect(self.browse_file)
        save_action = file_menu.addAction("Save")
        save_action.triggered.connect(self.save_data)

        # Edit Menu Actions
        cut_action = edit_menu.addAction("Cut")
        copy_action = edit_menu.addAction("Copy")
        paste_action = edit_menu.addAction("Paste")

        # Help Menu Actions
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(self.show_about)

    def create_bottom_status_bar(self):
        # Status Bar for Updates and Info
        self.setStatusBar(QStatusBar(self))

        # Add Progress Bar to the Status Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.statusBar().addWidget(self.progress_bar)

        # Add Status Label to the Status Bar
        self.status_label = QLabel("Status: Ready")
        self.statusBar().addWidget(self.status_label)

    def get_button_style(self):
        return """
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 12px;
                border-radius: 8px;
                font-size: 16px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """

    def browse_file(self):
        # Open File Dialog to Browse Files
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select fMRI Data File",
            "",  # Start in the current directory
            "NIfTI Files (*.nii.gz);;1D Files (*.1D);;All Files (*)"
        )

        if file_name:
            print(f"Selected file: {file_name}")  # Debugging: Print the selected file path
            if file_name.endswith((".nii.gz", ".1D")):
                self.file_input.setText(file_name)
                self.status_label.setText(f"Selected file: {os.path.basename(file_name)}")
            else:
                self.status_label.setText("Error: Unsupported file format. Please select a .nii.gz or .1D file.")

    def load_data(self):
        file_path = self.file_input.text()
        if not file_path:
            self.status_label.setText("Error: No file selected.")
            return

        try:
            if file_path.endswith(".nii.gz"):
                # Load .nii.gz file using nibabel
                img = nib.load(file_path)
                data = img.get_fdata()
                print(f"Loaded .nii.gz file with shape: {data.shape}")
                self.status_label.setText("Status: .nii.gz file loaded successfully")
            elif file_path.endswith(".1D"):
                # Load .1D file using numpy
                data = np.loadtxt(file_path)
                print(f"Loaded .1D file with shape: {data.shape}")
                print("Sample data:")
                print(data[:5])  # Print the first 5 rows of the .1D file
                self.status_label.setText("Status: .1D file loaded successfully")
            else:
                self.status_label.setText("Error: Unsupported file format.")
                return

            self.progress_bar.setValue(50)  # Simulate progress
        except Exception as e:
            self.status_label.setText(f"Error: {e}")

    def save_data(self):
        # Placeholder for saving data
        print("Saving data...")
        self.status_label.setText("Status: Data Saved")
        self.progress_bar.setValue(100)  # Simulate progress
        QApplication.processEvents()

    def show_about(self):
        # About dialog or information
        print("fMRI Analysis Tool - Version 1.0")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Global Style (Refined Color Scheme)
    app.setStyleSheet("""
        QMainWindow { background-color: #000000; }
        QLabel { font-size: 18px; font-family: Arial, sans-serif; color: white; }
        QPushButton { border-radius: 8px; font-size: 16px; }
        QPushButton:hover { background-color: #1E1E1E; }
        QLineEdit { padding: 12px; font-size: 16px; border-radius: 8px; background-color: #1E1E1E; }
        QMenuBar { background-color: #000000; color: #747474; font-size: 16px; }
        QMenu { background-color: #34495e; }
        QStatusBar { background-color: #2c3e50; color: white; font-size: 16px; }
    """)
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())