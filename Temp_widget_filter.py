from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLineEdit,
                             QApplication, QFrame, QDialog, QDialogButtonBox, QScrollArea)
from repay_code_table import CustomSortFilterProxyModel
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtCore import Qt
from collections import defaultdict


class FilterControl:
    """
    Controller для фильтра
    """
    CONDITIONS = ['Равно', 'Не равно',
                  'Больше', 'Больше или равно',
                  'Меньше', 'Меньше или равно',
                  'Начинается с', 'Не начинается с',
                  'Заканчивается на', 'Не заканчивается на',
                  'Не содержит', 'Содержит']

    COLUMNS = {"repay_code_table": {"digit": ['Колонка_1', 'Колонка_2'],
                                    "text": ['Колонка_3', 'Колонка_4']}}
    CONDITIONS_TYPES = {"digit": ['Больше', 'Больше или равно',
                                  'Меньше', 'Меньше или равно',
                                  ],
                        "text": ['Начинается с', 'Не начинается с',
                                 'Заканчивается на', 'Не заканчивается на',
                                 'Не содержит', 'Содержит',
                                 'Равно', 'Не равно']}

    def __int__(self):
        self._user_choice = []

    @property
    def user_choice(self):
        return self._user_choice

    @user_choice.setter
    def user_choice(self, value):
        self._user_choice = self.CheckChosenFilters(value)

    def update_condition(self, new_user_column, condition_select):
        new_col_type = self.find_key_by_value(new_user_column, self.COLUMNS)
        if new_col_type:
            condition_select.clear()
            condition_select.addItems(self.CONDITIONS_TYPES[new_col_type])
        else:
            condition_select.clear()
            condition_select.addItems(self.CONDITIONS)

    @staticmethod
    def find_key_by_value(value, dictionary):
        return next((sub_key for key, sub_dict in dictionary.items()
                     for sub_key, value_list in sub_dict.items()
                     if value in value_list), None)

    def CheckChosenFilters(self, chosen_filters: list):
        filters_by_column = self.GroupChosenFiltersByColumn(chosen_filters)
        print(filters_by_column)
        # Получение всех конфликтов
        conflicts = self.check_conflicts(filters_by_column)
        if conflicts:
            for column, (filter1, filter2) in conflicts:
                print(f"Противоречие в колонке {column}: {filter1} и {filter2}")
        else:
            return chosen_filters

    @staticmethod
    def GroupChosenFiltersByColumn(chosen_filters):
        # Группировка фильтров по колонкам
        filters_by_column = defaultdict(list)
        for column, filter_type, value in chosen_filters:
            filters_by_column[column].append((filter_type, value))

        return filters_by_column

    def check_conflicts(self, filters_by_column):
        """
        Проверяет каждую группу фильтров на предмет противоречий.
        """
        all_conflicts = []

        for column, conditions in filters_by_column.items():
            if any(filter_type in self.CONDITIONS_TYPES["digit"] for filter_type, _ in conditions):
                numeric_conflicts = self.is_conflict_numeric(conditions)
                for conflict in numeric_conflicts:
                    all_conflicts.append((column, conflict))

            if any(filter_type in self.CONDITIONS_TYPES["text"] for filter_type, _ in conditions):
                textual_conflicts = self.is_conflict_textual(conditions)
                for conflict in textual_conflicts:
                    all_conflicts.append((column, conflict))

        return all_conflicts

    def is_conflict_numeric(self, conditions):
        """
        Определяет противоречия в числовых условиях и возвращает пары противоречивых фильтров.
        """
        print("numeric", conditions)
        conflicts = []
        #if len(conditions) > 1:
        for i in range(len(conditions)):
            for j in range(i + 1, len(conditions)):
                if not self.is_consistent(conditions[i], conditions[j]):
                    conflicts.append((conditions[i], conditions[j]))
        return conflicts

    @staticmethod
    def is_consistent(filter_a, filter_b):
        """
        Проверяет логическую согласованность двух фильтров.
        Возвращает True, если фильтры согласованы, и False в противном случае.
        """
        filter_name_a, value_a = filter_a
        filter_name_b, value_b = filter_b
        value_a = float(value_a)
        value_b = float(value_b)

        if filter_name_a in ['Больше', 'Больше или равно'] and filter_name_b in ['Меньше', 'Меньше или равно']:
            if filter_name_a == 'Больше' and filter_name_b == 'Меньше':
                return value_a < value_b
            if filter_name_a == 'Больше или равно' and filter_name_b == 'Меньше':
                return value_a < value_b
            if filter_name_a == 'Больше' and filter_name_b == 'Меньше или равно':
                return value_a <= value_b
            if filter_name_a == 'Больше или равно' and filter_name_b == 'Меньше или равно':
                return value_a <= value_b

        if filter_name_a in ['Меньше', 'Меньше или равно'] and filter_name_b in ['Больше', 'Больше или равно']:
            if filter_name_a == 'Меньше' and filter_name_b == 'Больше':
                return value_a > value_b
            if filter_name_a == 'Меньше или равно' and filter_name_b == 'Больше':
                return value_a >= value_b
            if filter_name_a == 'Меньше' and filter_name_b == 'Больше или равно':
                return value_a >= value_b
            if filter_name_a == 'Меньше или равно' and filter_name_b == 'Больше или равно':
                return value_a >= value_b

        # В случае одинаковых фильтров, проверяем согласованность значений
        if filter_name_a == filter_name_b:
            if filter_name_a in ['Больше', 'Больше или равно']:
                return value_a <= value_b
            if filter_name_a in ['Меньше', 'Меньше или равно']:
                return value_a >= value_b

        return True  # Если ни одно из условий выше не сработало, фильтры согласованы

    @staticmethod
    def is_conflict_textual(conditions):
        print("text", conditions)
        """
        Определяет противоречия в текстовых условиях и возвращает пары противоречивых фильтров.
        """
        # Здесь должна быть ваша логика определения противоречий
        conflicts = []
        # Аналогичный подход, как в is_conflict_numeric
        return conflicts


class FilterWidget(QDialog, FilterControl):
    """
    Окно с виджетом фильтра
    """
    def __init__(self, proxy_model, parent):
        super().__init__()
        self.proxy_model = proxy_model  # Ссылка на вашу модель фильтрации
        self.parent = parent  # тот кто запустил окно

        self.layout = QVBoxLayout()

        self.control_buttons_layout = QHBoxLayout()
        # Кнопка для добавления нового фильтра
        self.add_filter_button = QPushButton("Добавить фильтр")
        self.add_filter_button.clicked.connect(self.add_filter_row)
        self.control_buttons_layout.addWidget(self.add_filter_button)

        self.remove_filter_button = QPushButton("Удалить фильтр", self)
        self.remove_filter_button.clicked.connect(self.remove_selected)
        self.control_buttons_layout.addWidget(self.remove_filter_button)

        self.layout.addLayout(self.control_buttons_layout)

        # Scroll Area для строк фильтров
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area_widget_contents = QWidget()
        self.scroll_area.setWidget(self.scroll_area_widget_contents)
        self.filters_layout = QVBoxLayout(self.scroll_area_widget_contents)
        self.layout.addWidget(self.scroll_area)

        # Диалоговые кнопки OK и Cancel
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept_filters)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

        self.setLayout(self.layout)

        self.setMinimumSize(750, 300)
        self.setWindowTitle("Фильтр")

        self.add_filter_row()

    def add_filter_row(self):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setPalette(QPalette(QColor(255, 255, 255)))
        frame.setAutoFillBackground(True)
        frame.mousePressEvent = lambda event, f=frame: self.select_frame(f, event)

        # Создаем новую строку фильтра
        filter_row_layout = QHBoxLayout(frame)

        # Выбор столбца для фильтрации
        column_select = QComboBox()
        column_select.addItems([value for v in self.COLUMNS[self.parent].values() for value in v])
        filter_row_layout.addWidget(column_select)

        # Выбор условия
        condition_select = QComboBox()
        condition_select.addItems(self.CONDITIONS)
        filter_row_layout.addWidget(condition_select)

        column_select.currentTextChanged.connect(lambda text, cs=condition_select: self.update_condition(text, cs))

        # Поле для ввода значения фильтра
        value_input = QLineEdit()
        filter_row_layout.addWidget(value_input)

        # Добавляем строку в layout

        self.filters_layout.addWidget(frame)

    def remove_selected(self):
        for i in reversed(range(self.filters_layout.count())):
            widget = self.filters_layout.itemAt(i).widget()
            if widget.palette().color(QPalette.Background) == QColor(30, 144, 255):
                self.filters_layout.removeWidget(widget)
                widget.deleteLater()

    def select_frame(self, frame, event):
        if event.button() == Qt.LeftButton:
            palette = frame.palette()
            if palette.color(QPalette.Background) == QColor(255, 255, 255):
                palette.setColor(QPalette.Background, QColor(30, 144, 255))
            else:
                palette.setColor(QPalette.Background, QColor(255, 255, 255))
            frame.setPalette(palette)

    def remove_filter_row(self, layout):
        # Удаляем виджеты из layout и затем сам layout
        for i in reversed(range(layout.count())):
            widget_to_remove = layout.itemAt(i).widget()
            layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)
        layout.setParent(None)

    def apply_filters(self):
        # Собираем все условия фильтра и применяем их к proxy_model
        for i in range(self.filters_layout.count()):
            filter_layout = self.filters_layout.itemAt(i)

            column_index = filter_layout.itemAt(0).widget().currentIndex()
            condition = filter_layout.itemAt(1).widget().currentText()
            value = filter_layout.itemAt(2).widget().text()

            # Можно добавить дополнительную обработку, если значение не валидно
            # Здесь должен быть вызов метода proxy_model для установки фильтра
            self.proxy_model.setNumericFilter(column_index, condition, float(value))

        self.proxy_model.invalidateFilter()

    def get_filters(self):
        filters = []
        for i in range(self.filters_layout.count()):
            filter_frame = self.filters_layout.itemAt(i).widget()
            if not isinstance(filter_frame, QFrame):
                continue

            filter_layout = filter_frame.layout()

            # Извлекаем информацию о фильтре
            column_name = filter_layout.itemAt(0).widget().currentText()
            condition = filter_layout.itemAt(1).widget().currentText()

            # # Для условия "диапазон" может потребоваться особая обработка
            # if condition == "диапазон":
            #     left_value = filter_layout.itemAt(2).widget().text()
            #     right_value = filter_layout.itemAt(3).widget().text()
            #     value = f"{left_value} - {right_value}"
            # else:
            value = filter_layout.itemAt(2).widget().text()

            filters.append((column_name, condition, value))

        return filters

    def accept_filters(self):
        self.user_choice = self.get_filters()
        return self.user_choice


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    proxy_model = CustomSortFilterProxyModel(perc_cols=[], int_cols=[])

    filter_widget = FilterWidget(proxy_model, "repay_code_table")

    filter_widget.show()
    if filter_widget.exec_() == QDialog.Accepted:
        options = filter_widget.get_filters()
        print(options)  # Или передать этот словарь дальше для обработки
    else:
        print("return")

    filter_widget.close()
