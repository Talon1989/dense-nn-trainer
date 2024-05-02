from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QMainWindow, QProgressDialog, QFileDialog, QApplication


class MainWindow(QMainWindow):
    def __init__(self, title: str):
        super().__init__()
        self.setWindowTitle(title)
        self.text_box = None
        self.csv_file = None
        self.csv_file_path = None
        self.feature_dict = None
        self.target = None
        self.n_epochs = 0
        self.nn_shape = []
        self.model_location = None
        self.model = None
        self.pool = QThreadPool()
        self.progress_dialog = None

    def _generated_progress_dialog(self, n, title, text):
        self.progress_dialog = QProgressDialog()
        self.progress_dialog.setWindowTitle("Fitting")
        self.progress_dialog.setLabelText("fitting the model ...")
        self.progress_dialog.setMaximum(n)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()

    def _show_save_dialog(self):
        dialog = QFileDialog()
        dialog.setFileMode(
            QFileDialog.FileMode.Directory
        )  # Allow selecting directories
        dialog.setWindowTitle("Select a Save Location")
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)  # Show only directories
        if dialog.exec() == QFileDialog.DialogCode.Accepted:
            location = dialog.selectedFiles()[0]
            try:
                self.model_location = location[0 : location.index(".")] + ".pth"
                return True
            except ValueError:
                if "." in location:
                    self.model_location = location
                self.model_location = location + ".pth"
                return True
        else:
            return False
