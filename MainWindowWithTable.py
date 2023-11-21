from libs import *
from ui_widget_SurveyReport_wind import Ui_widget_SurveyReport_location_wind, CustomTableModel
from repay_code_table import CustomSortFilterProxyModel, SQLite_model, MyDelegate
import logging
from SurveyReport_service import SurveyReportService


class Back_SurveyReport_function_wind(QtWidgets.QWidget):
    """
    Окно с таблицей для редактирования данных
    """
    def __init__(self, report_type, data, db, session_value, data_repository):
        """
        :param report_type: тип отчёта (от него зависит UI)
        :param data: название функции
        :param db: соединение с БД
        :param session_value: id сессии
        :param data_repository: хранилище для запросов к БД
        """
        super().__init__()
        self.report_type = report_type
        self.session_value = session_value

        if isinstance(data, dict):
            self.is_agg = True
            self.function_value = data['function_value']
            self.function_list = data['list']
        else:
            self.is_agg = False
            self.function_value = data
            self.function_list = data
        print(self.function_value, self.function_list)
        self.db = db

        self.ui = Ui_widget_SurveyReport_location_wind(self.function_value, self.report_type)
        self.ui.setupUi(self)

        self.report_service = SurveyReportService(data_repository)

        if self.report_type == "headcount":
            self.editable_columns = [3, 4]
            self.column_indexes = {'id': 0, 'Клиент': 1, 'Функция\n II уровня': 2,
                                   'Кол-во от\n общей численности': 3, 'Нагрузка, чел': 4, 'session_id': 5}
            self.table = "headcount_view"
            self.attribute = "division"
            self.calculated_cols = []
            self.attribute_col = 2
            self.perc_cols = [3]
            self.int_cols = [4]
            self.str_cols = []
            self.non_empty_cols = []
            self.calculated_cols = []
            self.light_grey_cols = []
            self.heavy_grey_cols = []
            self.str_fixed_values = {}
            self.column_names = {"id": "id", "Клиент": "client_id",
                                 "Функция\n II уровня": "division",
                                 "Кол-во от\n общей численности": "function_perc", "Нагрузка, чел": "function_load",
                                 "session_id": 'session_id'}
            self.original_table = "headcount_rawdata"
            self.delete_command = f"""INSERT INTO {self.original_table} 
                                                    (id, client_id, division,                                   
                                                    function_perc, function_load, session_id) 
                                                    VALUES (?, ?, ?, ?, ?, ?)"""
            self.filter_numeric_columns = ['Кол-во от\n общей численности', 'Нагрузка, чел']

            if self.is_agg:
                self.filtered_columns = ['Клиент', 'Функция\n II уровня', 'Кол-во от\n общей численности',
                                         'Нагрузка, чел']
                self.hidden_columns = [0, 5]
            else:
                self.filtered_columns = ['Клиент', 'Кол-во от\n общей численности', 'Нагрузка, чел']
                self.hidden_columns = [0, 2, 5]

        elif self.report_type == "absolute":
            self.hidden_columns = [0, 2, 6]
            self.filtered_columns = ['Клиент', 'Стоимость\n функции', 'Фонд заработной\n платы', 'Премиальный\n фонд']
            self.editable_columns = [3, 4, 5]
            self.column_indexes = {'id': 0, 'Клиент': 1, 'Функция\n I уровня': 2,
                                   'Стоимость\n функции': 3, 'Фонд заработной\n платы': 4, 'Премиальный\n фонд': 5,
                                   'session_id': 6}
            self.table = "function_cost_absolute_view"
            self.attribute = "function"
            self.calculated_cols = []
            self.attribute_col = 2
            self.perc_cols = []
            self.int_cols = [3, 4, 5]
            self.str_cols = []
            self.non_empty_cols = []
            self.calculated_cols = []
            self.light_grey_cols = []
            self.heavy_grey_cols = []
            self.str_fixed_values = {}
            self.column_names = {"id": "id", "Клиент": "client_id", "Функция\n I уровня": "function",
                                 "Стоимость\n функции": "total_cost", "Фонд заработной\n платы": "AGP",
                                 "Премиальный\n фонд": "AB", "session_id": 'session_id'}
            self.original_table = "function_cost_absolute_rawdata"
            self.delete_command = f"""INSERT INTO {self.original_table} 
                                               (id, client_id, function,                                   
                                               total_cost, AGP, AB, session_id) 
                                               VALUES (?, ?, ?, ?, ?, ?, ?)"""
            self.filter_numeric_columns = ['Стоимость\n функции', 'Фонд заработной\n платы', 'Премиальный\n фонд']

        else:
            self.hidden_columns = [0, 2, 12]
            self.filtered_columns = ['Клиент', "Стоимость функции\n от выручки", "Стоимость функции\n от прибыли",
                                     "Стоимость функции\n от расходов", "Фонд заработной платы\n от выручки",
                                     "Фонд заработной платы\n от прибыли", "Фонд заработной платы\n от расходов",
                                     "Премиальный фонд\n от выручки", "Премиальный фонд\n от прибыли",
                                     "Премиальный фонд\n от расходов"]

            self.editable_columns = [3, 4, 5, 6, 7, 8, 9, 10, 11]
            self.column_indexes = {"id": "0", "Клиент": 1, "Функция\n I уровня": 2,
                                   "Стоимость функции\n от выручки": 3,
                                   "Стоимость функции\n от прибыли": 4,
                                   "Стоимость функции\n от расходов": 5,
                                   "Фонд заработной платы\n от выручки": 6,
                                   "Фонд заработной платы\n от прибыли": 7,
                                   "Фонд заработной платы\n от расходов": 8,
                                   "Премиальный фонд\n от выручки": 9,
                                   "Премиальный фонд\n от прибыли": 10,
                                   "Премиальный фонд\n от расходов": 11,
                                   "session_id": 12}
            self.table = "function_cost_relative_view"
            self.attribute = "function"
            self.calculated_cols = []

            self.attribute_col = 2
            self.perc_cols = [3, 4, 5, 6, 7, 8, 9, 10, 11]
            self.int_cols = []
            self.str_cols = []
            self.non_empty_cols = []
            self.calculated_cols = []
            self.light_grey_cols = []
            self.heavy_grey_cols = []
            self.str_fixed_values = {}
            self.column_names = {"id": "id", "Клиент": "client_id", "Функция\n I уровня": "function",
                                 "Стоимость функции\n от выручки": "total_cost_revenue",
                                 "Стоимость функции\n от прибыли": "total_cost_profit",
                                 "Стоимость функции\n от расходов": "total_cost_operating_costs",
                                 "Фонд заработной платы\n от выручки": "AGP_revenue",
                                 "Фонд заработной платы\n от прибыли": "AGP_profit",
                                 "Фонд заработной платы\n от расходов": "AGP_operating_costs",
                                 "Премиальный фонд\n от выручки": "AB_revenue",
                                 "Премиальный фонд\n от прибыли": "AB_profit",
                                 "Премиальный фонд\n от расходов": "AB_operating_costs",
                                 "session_id": 'session_id'}
            self.original_table = "function_cost_relative_rawdata"

            self.delete_command = f"""INSERT INTO {self.original_table} 
                                                           (id, client_id, function,                                   
                                                           total_cost_revenue, total_cost_profit, total_cost_operating_costs, 
                                                           AGP_revenue, AGP_profit, AGP_operating_costs, 
                                                           AB_revenue, AB_profit, AB_operating_costs, 
                                                           session_id) 
                                                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            self.filter_numeric_columns = ["Стоимость функции\n от выручки", "Стоимость функции\n от прибыли",
                                           "Стоимость функции\n от расходов", "Фонд заработной платы\n от выручки",
                                           "Фонд заработной платы\n от прибыли", "Фонд заработной платы\n от расходов",
                                           "Премиальный фонд\n от выручки", "Премиальный фонд\n от прибыли",
                                           "Премиальный фонд\n от расходов"]

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        screen_resolution = QApplication.instance().desktop().screenGeometry()
        sizes = [screen_resolution.width(), screen_resolution.height()]
        self.model = SQLite_model(self.editable_columns, self.column_indexes, self.db, sizes, self.perc_cols,
                                  self.int_cols, self.calculated_cols, self.report_type)
        self.model.setTable(self.table)
        self.model.dataChanged.connect(self.load_data_stats)

        if self.is_agg:
            function_list_string = ', '.join(f"'{item}'" for item in self.function_list)
            model_filter = f""" session_id = {self.session_value} AND {self.attribute} IN ({function_list_string})"""
        else:
            model_filter = f""" session_id = {self.session_value} AND {self.attribute} = '{self.function_value}' """

        self.model.setFilter(model_filter)
        self.model.select()

        self.proxy_model = CustomSortFilterProxyModel(self.perc_cols, self.int_cols)
        self.proxy_model.setSourceModel(self.model)

        self.ui.table_view.setModel(self.proxy_model)

        self.delegate = MyDelegate(self.proxy_model, self.ui.table_view, self.db,
                                   self.editable_columns, self.perc_cols, self.int_cols, self.calculated_cols,
                                   self.str_cols, self.non_empty_cols, self.light_grey_cols, self.heavy_grey_cols,
                                   self.str_fixed_values, self.column_names, self.original_table, self.attribute,
                                   self.attribute_col, self.delete_command, self.logger, self)

        self.ui.table_view.setItemDelegate(self.delegate)
        self.ui.table_view.horizontalHeader().sectionDoubleClicked.connect(self.toggle_sorting)

        for col in self.hidden_columns:
            self.ui.table_view.hideColumn(col)

        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.delegate.undo)
        self.redo_shortcut = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.redo_shortcut.activated.connect(self.delegate.redo)
        self.delete_shortcut = QShortcut(QKeySequence("Delete"), self)
        self.delete_shortcut.activated.connect(self.delete_selected_rows)
        self.backspace_shortcut = QShortcut(QKeySequence("Backspace"), self)
        self.backspace_shortcut.activated.connect(self.delete_selected_rows)
        self.clear_filter_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        self.clear_filter_shortcut.activated.connect(self.clear_filter)

        self.ui.comboBox_column_filter.addItems(self.filtered_columns)
        self.ui.comboBox_condition_filter.addItems(['по тексту', 'больше чем', 'меньше чем'])

        self.ui.comboBox_condition_filter.currentTextChanged.connect(self.update_filter_binding)

        #   self.ui.pushbutton_clear.clicked.connect(self.clear_filter)
        self.ui.lineEdit.returnPressed.connect(self.apply_filter)
        self.last_condition = None
        self.filter_enabled = True
        self.sorting_enabled = False

        self.filter_hidden = False
        self.ui.label_filter.clicked.connect(self.control_filter)

        self.load_data_stats()

    def control_filter(self):
        indicator = "▼" if self.filter_hidden else "▶"
        if self.filter_hidden:
            self.ui.label_filter.setText(indicator + "Фильтр")
            self.ui.label_filter.setGraphicsEffect(self.ui.shadow_effect)
            self.ui.lineEdit.show()
            self.ui.comboBox_column_filter.show()
            self.ui.comboBox_condition_filter.show()
            # self.ui.pushbutton_clear.show()
            self.ui.checkbutton_all_cols.show()
            self.filter_hidden = False
        else:
            self.ui.label_filter.setText(indicator + "Фильтр")
            self.ui.lineEdit.hide()
            self.ui.comboBox_column_filter.hide()
            self.ui.comboBox_condition_filter.hide()
            # self.ui.pushbutton_clear.hide()
            self.ui.checkbutton_all_cols.hide()
            self.filter_hidden = True

    def control_checkbutton(self):
        if self.ui.checkbutton_all_cols.isChecked():
            self.ui.comboBox_column_filter.setEnabled(False)
            self.ui.comboBox_condition_filter.setEnabled(False)
        else:
            self.ui.comboBox_condition_filter.setEnabled(True)
            self.ui.comboBox_column_filter.setEnabled(True)

    def toggle_sorting(self):
        if self.ui.table_view.isSortingEnabled():
            self.ui.table_view.setSortingEnabled(False)
        else:
            self.ui.table_view.setSortingEnabled(True)

    def toggle_filter(self):
        if self.filter_enabled:
            self.clear_filter()
            self.filter_enabled = False
        else:
            self.filter_enabled = True

    def update_filter_binding(self, text):
        if text == "по тексту":
            self.ui.comboBox_column_filter.clear()
            self.ui.comboBox_column_filter.addItems(list(self.filtered_columns))
        else:
            if self.last_condition not in ['больше чем', 'меньше чем']:
                # привязка сигнала clicked pushbutton_filter к apply_filter
                self.ui.comboBox_column_filter.clear()
                self.ui.comboBox_column_filter.addItems(self.filter_numeric_columns)
        self.last_condition = text
        self.apply_filter()

    def clear_filter(self):
        self.ui.comboBox_condition_filter.setEnabled(True)
        self.ui.comboBox_column_filter.setEnabled(True)
        self.ui.checkbutton_all_cols.setChecked(False)
        self.ui.lineEdit.clear()

        # Очистите фильтры прокси-модели
        self.proxy_model.setFilterColumn(-1)  # Сброс фильтрации колонки
        self.proxy_model.setFilterRegExp("")  # Очистка регулярного выражения для фильтрации
        self.proxy_model.invalidateFilter()  # Пересчет фильтра

    def apply_filter(self):
        column = self.ui.comboBox_column_filter.currentText()
        condition = self.ui.comboBox_condition_filter.currentText()
        text = self.ui.lineEdit.text()
        column_index = self.column_indexes[column]

        if self.ui.checkbutton_all_cols.isChecked():
            self.proxy_model.setFilterKeyColumn(-1)
            self.proxy_model.setFilterRegExp(text)
        else:
            self.proxy_model.setFilterKeyColumn(column_index)

            # логика фильтрации
            if condition == 'по тексту':
                self.proxy_model.setFilterRegExp(text)
            elif condition in ['больше чем', 'меньше чем']:
                if text.replace("%", "").replace(" ", "").isdigit():
                    threshold = int(text.replace("%", "").replace(" ", ""))
                    self.proxy_model.setNumericFilter(column_index, condition, threshold)

    def delete_selected_rows(self):
        rows = sorted(set(index.row() for index in self.ui.table_view.selectedIndexes()))
        self.delegate.delete_rows(rows)

    def load_data_stats(self):
        # отправляем в GUI
        self.stats_model = CustomTableModel(self.report_service.get_single_code_report(self.report_type,
                                                                                       self.function_list,
                                                                                       self.session_value))

        self.ui.table_stats.setModel(self.stats_model)
        self.ui.table_stats.setEditTriggers(QtWidgets.QTableView.NoEditTriggers)  # Make table uneditable
        for col in self.hidden_columns:
            self.ui.table_view.hideColumn(col)
