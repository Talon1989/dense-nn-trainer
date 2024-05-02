from CustomWindow import CustomWindow
from PySide6.QtWidgets import QApplication


if __name__ == "__main__":
    app = QApplication()
    window = CustomWindow("nn application")
    window.show()
    app.exec()
