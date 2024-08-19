# main_menu.py

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QComboBox, QSpinBox, QTextEdit, QMessageBox
from PyQt5.QtCore import Qt
from settings import GameSettings

settings = GameSettings()


class MainMenu(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Maze Game Menu")
        self.setGeometry(100, 100, 400, 300)

        # Create widgets
        self.label = QLabel('Maze Game Menu', self)
        self.label.setAlignment(Qt.AlignCenter)

        self.play_player_button = QPushButton('Play as Player', self)
        self.play_auto_button = QPushButton('Play Auto', self)
        self.settings_button = QPushButton('Settings', self)

        # Connect buttons
        self.play_player_button.clicked.connect(self.play_player)
        self.play_auto_button.clicked.connect(self.play_auto)
        self.settings_button.clicked.connect(self.open_settings)

        # Set layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.play_player_button)
        layout.addWidget(self.play_auto_button)
        layout.addWidget(self.settings_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def play_player(self):
        self.hide()  # Hide the main menu window
        from maze_solver import main_player
        main_player(self)  # Pass self as an argument

    def play_auto(self):
        self.hide()  # Hide the main menu window
        from maze_solver import main_auto
        main_auto(self)  # Pass self as an argument

    def open_settings(self):
        self.settings_window = SettingsWindow()
        self.settings_window.show()


class SettingsWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Settings")
        self.setGeometry(150, 150, 300, 300)

        # Create widgets
        self.gen_mode_label = QLabel('Generation Mode:', self)
        self.gen_mode_combo = QComboBox(self)
        self.gen_mode_combo.addItems(
            ['random', 'recursive_backtracking', 'string', 'prim', 'spiral', 'growing_tree', 'rooms_and_corridors', 'fractal'])
        self.gen_mode_combo.setCurrentText(settings.GENERATION_MODE)

        self.field_size_label = QLabel('Field Size:', self)
        self.field_size_x_spin = QSpinBox(self, minimum=10, maximum=200)
        self.field_size_y_spin = QSpinBox(self, minimum=10, maximum=200)
        self.field_size_x_spin.setRange(10, 200)
        self.field_size_y_spin.setRange(10, 200)
        self.field_size_x_spin.setValue(settings.FIELD_SIZES[0])
        self.field_size_y_spin.setValue(settings.FIELD_SIZES[1])

        self.map_label = QLabel(
            'Custom Map (use "+" for walls, "." for empty):', self)
        self.map_text_edit = QTextEdit(self)
        self.map_text_edit.setPlainText(settings.OWN_MAP)  # Set the current map content

        self.save_button = QPushButton('Save', self)
        self.save_button.clicked.connect(self.save_settings)

        # Set layout
        layout = QVBoxLayout()
        layout.addWidget(self.gen_mode_label)
        layout.addWidget(self.gen_mode_combo)
        layout.addWidget(self.field_size_label)
        layout.addWidget(self.field_size_x_spin)
        layout.addWidget(self.field_size_y_spin)
        layout.addWidget(self.map_label)
        layout.addWidget(self.map_text_edit)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def save_settings(self):
        from maze_solver import find_exit, create_maze_from_string, find_random_start_old
        settings.GENERATION_MODE = self.gen_mode_combo.currentText()
        settings.FIELD_SIZES = (self.field_size_x_spin.value(),
                       self.field_size_y_spin.value())
        settings.OWN_MAP = self.map_text_edit.toPlainText()
        max_field_size = max(settings.FIELD_SIZES)
        settings.CELL_SIZE = max(10, 1000 // max_field_size)

        try:
            if settings.GENERATION_MODE == "string" and settings.OWN_MAP:
                maze = create_maze_from_string(settings.OWN_MAP)
                if maze:
                    start_row, start_col = find_random_start_old(maze)
                    path = find_exit(maze, start_row, start_col)
                    if path is None:
                        settings.OWN_MAP = ""
                        self.map_text_edit.setText(
                            "Wrong map, there's no way out.")
                        QMessageBox.critical(
                            self, "Error", f"Wrong map, there's no way out.")
                        return
                else:
                    settings.OWN_MAP = ""
                    self.map_text_edit.setText(
                        "Invalid map format. Use '+' for walls, '.' for empty.")
                    QMessageBox.critical(
                        self, "Error", f"Wrong map, entered the wrong characters.")
                    return
            elif settings.GENERATION_MODE == "spiral":
                if settings.FIELD_SIZES[0] != settings.FIELD_SIZES[1]:
                    self.field_size_x_spin.setValue(50)
                    self.field_size_y_spin.setValue(50)
                    QMessageBox.critical(
                        self, "Error", f"This generation is only possible for square fields.")
                    return
            print(settings)
            # print(
            #     f"Settings saved: Generation Mode: {settings.GENERATION_MODE}, Field Size: {settings.FIELD_SIZES}, Custom Map Length: {len(settings.OWN_MAP)}")
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")


def main():
    app = QApplication(sys.argv)
    main_menu = MainMenu()
    main_menu.show()
    sys.exit(app.exec_())
