import pandas as pd
import torch.jit

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QPushButton,
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QFileDialog,
    QMessageBox,
    QGridLayout,
    QCheckBox,
    QRadioButton
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader

from Window import MainWindow
from CustomUtilities import *


WELCOME_TEXT = (
    "This program let the user create a neural net dense model that"
    " learns from a dataset in .csv format.\nBoth regression and classification are possible.\n"
    "Pytorch, Pandas and Numpy are required"
)
FEATURES_TEXT = "Select features:"
TARGET_TEXT = "Select target:"
FORMAT_TEXT = (
    "In the text below specify the shape of the dense nn\n"
    "for example for a 16x16x32 write '16x16x32' without apostrophe\n"
    "only integers are accepted"
)


class CustomWindow(MainWindow):
    def __init__(self, title):
        super().__init__(title)
        self._file_selection()

    def _file_selection(self):

        layout = QVBoxLayout()

        welcome_label = QLabel(WELCOME_TEXT)
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        button_1 = QPushButton("Select .csv file")
        button_2 = QPushButton("Exit")

        button_1.clicked.connect(self._select_file)
        button_2.clicked.connect(QApplication.quit)

        layout.addWidget(welcome_label)  # span 1 row, 2 cols
        layout.addWidget(button_1)
        layout.addWidget(button_2)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def _select_file(self):
        dialog = QFileDialog()
        dialog.setNameFilter("CSV files (*.csv)")
        dialog.show()
        if dialog.exec():  # selection has occurred
            self.csv_file_path = dialog.selectedFiles()[0]
            self._type_selection()

    def _type_selection(self):

        def func(model_type: str):
            self.model_type = model_type
            self._features_and_target_selection()

        layout = QVBoxLayout()

        welcome_label = QLabel('Select type of learning')
        button_1 = QPushButton('Classification')
        button_2 = QPushButton('Regression')

        button_1.clicked.connect(lambda: func('classification'))
        button_2.clicked.connect(lambda: func('regression'))

        layout.addWidget(welcome_label, alignment=Qt.AlignmentFlag.AlignHCenter)
        # layout.addWidget(welcome_label)
        layout.addWidget(button_1)
        layout.addWidget(button_2)
        # layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def _features_and_target_selection(self):

        grid_layout = QGridLayout()

        feature_label = QLabel(FEATURES_TEXT)
        target_label = QLabel(TARGET_TEXT)
        confirm_button = QPushButton("Ok")
        epoch_label = QLabel("Select number of epochs for training")
        spin_box = QSpinBox(self)
        spin_box.setMaximum(5_000)
        self.csv_file = pd.read_csv(self.csv_file_path)
        self.feature_dict = {key: False for key in self.csv_file.columns}

        grid_layout.addWidget(feature_label, 0, 0)
        feats = []
        counter = 1
        for feature in self.feature_dict.keys():
            checkbox = QCheckBox(feature)
            grid_layout.addWidget(checkbox, counter, 0)
            feats.append(checkbox)
            counter += 1

        grid_layout.addWidget(target_label, 0, 1)
        targets = []
        counter = 1
        for feature in self.feature_dict.keys():
            radio_button = QRadioButton(feature)
            grid_layout.addWidget(radio_button, counter, 1)
            targets.append(radio_button)
            counter += 1

        feature_label.setAlignment(Qt.AlignmentFlag.AlignBottom)
        target_label.setAlignment(Qt.AlignmentFlag.AlignBottom)
        epoch_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(epoch_label, len(self.feature_dict.keys()) + 1, 0)
        grid_layout.addWidget(spin_box, len(self.feature_dict.keys()) + 1, 1)
        grid_layout.addWidget(confirm_button, len(self.feature_dict.keys()) + 2, 0)

        confirm_button.clicked.connect(
            lambda: self._confirm_features(feats, targets, spin_box)
        )

        widget = QWidget()
        widget.setLayout(grid_layout)
        self.setCentralWidget(widget)

    def _confirm_features(self, feats, targets, spin_box):
        self.feature_dict = {f: False for f in self.feature_dict.keys()}
        for f in feats:
            if f.isChecked():
                self.feature_dict[f.text()] = True
        for radio_b in targets:
            if radio_b.isChecked():
                self.target = radio_b.text()
        self.n_epochs = spin_box.value()
        if all(f is False for f in self.feature_dict.values()) or not self.target:
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Icon.Critical)
            error_dialog.setWindowTitle("Error")
            error_dialog.setText("Please select a target and at least one feature")
            error_dialog.show()
            error_dialog.exec()
            return
        self.nn_shape_editor()

    def nn_shape_editor(self):

        layout = QVBoxLayout()
        message = QLabel(FORMAT_TEXT)
        self.text_box = QLineEdit()
        confirm_button = QPushButton("Confirm")

        layout.addWidget(message)
        layout.addWidget(self.text_box)
        layout.addWidget(confirm_button)

        self.text_box.returnPressed.connect(
            lambda: self.worker_runner(self.text_box.text())
        )
        confirm_button.clicked.connect(lambda: self.worker_runner(self.text_box.text()))

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def worker_runner(self, text):
        try:
            self.nn_shape = list(map(int, text.split("x")))
            response = self._show_save_dialog()
            if not response:
                return

        except ValueError:
            self.text_box.setText("")
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Icon.Critical)
            error_dialog.setWindowTitle("Error")
            error_dialog.setText("Please enter shape format as specified")
            error_dialog.exec()
            return

        self.pool.clear()
        self._generated_progress_dialog(
            n=self.n_epochs, title="Fitting", text="fitting the model ..."
        )
        worker = Worker(self.fit_model)
        self.pool.start(worker)

    def fit_model(self):
        columns = []
        for feature, flag in self.feature_dict.items():
            if flag:
                columns.append(feature)

        x = self.csv_file.loc[:, columns].to_numpy()
        y = self.csv_file.loc[:, self.target].to_numpy()

        if self.model_type == 'classification':
            y = one_hot(LabelEncoder().fit_transform(y))
            model = ClassificationModel(len(columns), np.array(self.nn_shape), y.shape[1])
        elif self.model_type == 'regression':
            model = RegressionModel(len(columns), np.array(self.nn_shape))
        else:
            print('nothing')

        dataset = TensorDataset(
            torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
        )
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        criterion = nn.MSELoss()
        optim = torch.optim.Adam(lr=1 / 1_000, params=model.parameters())

        for ep in range(self.n_epochs):
            losses = []
            for x, y in dataloader:
                optim.zero_grad()
                model.train()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optim.step()
                losses.append(loss.detach().numpy())
                if self.progress_dialog.wasCanceled():
                    self.progress_dialog.destroy()
                    return
            self.progress_dialog.setValue(ep + 1)

            if (ep + 1) % 50 == 0 or ep < 10:
                print(f"Episode {ep + 1} | Loss: {np.mean(losses)}")
        self.progress_dialog.destroy()
        self.model = model
        try:
            torch.save(self.model, self.model_location)
            print("model saved.")
            info_dialog = QMessageBox()
            info_dialog.setIcon(QMessageBox.Icon.Information)
            info_dialog.setWindowTitle("Model saved")
            info_dialog.setText(
                f"Model saved locally with loss: {losses[-1]: .5f}"
            )
            info_dialog.show()
            info_dialog.exec()
            info_dialog.finished.connect(self.destroy)
        except Exception:
            print("unable to save model.")
