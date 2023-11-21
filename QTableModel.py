from libs import *


class CustomSortFilterProxyModel(QSortFilterProxyModel):
    """
    Прокси-модель для фильтрации и сортировки
    """
    def __init__(self, perc_cols, int_cols, parent=None):
        super(CustomSortFilterProxyModel, self).__init__(parent)
        self.int_cols = perc_cols + int_cols
        self.filter_column = -1
        self.filter_condition = None
        self.filter_threshold = None

    def lessThan(self, left, right):
        left_data = self.sourceModel().data(left)
        right_data = self.sourceModel().data(right)

        # Преобразуем данные из колонки коэффициента в числа
        if left.column() in self.int_cols and right.column() in self.int_cols:
            left_data = self.process_data(left_data)
            right_data = self.process_data(right_data)

        # Проверяем, что данные представляют собой числа
        if isinstance(left_data, int) and isinstance(right_data, int):
            return left_data < right_data

        # Возвращаем стандартное поведение для сортировки в других случаях
        return super().lessThan(left, right)

    def process_data(self, data):
        if data is None or data == "":
            return 0
        else:
            # удаляем знак '%' и преобразуем строку в число
            if '%' in data:
                return float(data.replace("%", "").replace(" ", ""))
            else:
                return int(data.replace(" ", ""))

    def setNumericFilter(self, column, condition, threshold):
        self.filter_column = column
        self.filter_condition = condition
        self.filter_threshold = threshold
        self.invalidateFilter()  # этот метод пересчитывает фильтр

    def filterAcceptsRow(self, source_row, source_parent):
        if self.filterRegExp().pattern():
            return super().filterAcceptsRow(source_row, source_parent)

        if self.filter_column == -1:
            return True

        index = self.sourceModel().index(source_row, self.filter_column, source_parent)
        data = self.process_data(self.sourceModel().data(index))

        if self.filter_condition == 'больше чем':
            return data > self.filter_threshold
        elif self.filter_condition == 'меньше чем':
            return data < self.filter_threshold
        return True

    def clearNumericFilter(self):
        self.setNumericFilter(-1, None, None)

    def setFilterColumn(self, column):
        self.filter_column = column
        self.invalidateFilter()


class CustomTableView(QTableView):
    """
    Унаследование представления для переопределения метода чтобы удалять строки по кнопке Delete на клавиатуре
    """
    def __init__(self, parent=None):
        super(CustomTableView, self).__init__(parent)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            selected_rows = sorted(set(index.row() for index in self.selectedIndexes()), reverse=True)
            self.itemDelegate().delete_rows(selected_rows)
        super(CustomTableView, self).keyPressEvent(event)


class SQLite_model(QSqlTableModel):
    """
    Модель
    """
    def __init__(self, editable_columns, custom_headers, connection, sizes, perc_cols, int_cols, calculated_cols,
                 report_type, parent=None):
        """
        :param editable_columns: редактируемые колонки
        :param custom_headers: UI заголовки и их БД названия атрибутов
        :param connection: соединение с БД
        :param sizes: размеры экрана
        :param perc_cols: колонки с процентами
        :param int_cols: колонки с числами
        :param calculated_cols: рассчитываемые колонки для выделения цветом
        :param report_type: тип отчёта
        :param parent: QSqlTableModel
        """
        super(SQLite_model, self).__init__(parent, connection)
        self.editable_columns = editable_columns

        self.custom_headers = list(custom_headers.keys())
        self.perc_cols = perc_cols
        self.int_cols = int_cols
        self.calculated_cols = calculated_cols
        self.report_type = report_type # для флагов

        self.sizes = sizes
        if self.sizes[0] + self.sizes[1] > 5000:
            self.font_size = 6
        else:
            self.font_size = 7

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.custom_headers[section]
        if role == Qt.FontRole and orientation == Qt.Horizontal:
            font = QFont()
            font.setPointSize(self.font_size)  # Set the font size you want
            if section in self.calculated_cols:  # Columns to make bold
                font.setBold(True)
                font.setPointSize(self.font_size + 1)
            return font
        return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.DisplayRole):
        value = QSqlTableModel.data(self, index, role)
        if role == Qt.DisplayRole:
            if value is not None:
                if index.column() in self.perc_cols:
                    return value if value is None or value == "" else f'{float(value) * 100:.3f}%'
                if index.column() in self.int_cols:
                    return value if value is None or value == "" else "{:,}".format(int(value)).replace(",", " ")
            else:
                return QSqlTableModel.data(self, index, role)

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter  # Set the alignment to center
        return super(SQLite_model, self).data(index, role)

    def flags(self, index):
        if self.report_type == 'headcount':
            return super(SQLite_model, self).flags(index) & ~Qt.ItemIsEditable
        original_flags = super(SQLite_model, self).flags(index)
        if index.column() not in self.editable_columns:
            return original_flags & ~Qt.ItemIsEditable
        return original_flags


class MyDelegate(QStyledItemDelegate):
    """
    Делегат для проверки вводимых данных и настройки UI
    """
    def __init__(self, proxy_model, table_view, db_connection, editable_columns, perc_cols, int_cols, calculated_cols,
                 str_cols, non_empty_cols, light_grey_cols, heavy_grey_cols, str_fixed_values, column_names,
                 original_table, attribute, attribute_col, delete_command, logger, parent_window_tool, parent=None):
        """
        :param proxy_model: прокси-модель
        :param table_view: представление
        :param db_connection: содинение
        :param editable_columns: список редактируемых колонок
        :param perc_cols: колонки с процентами
        :param int_cols: колонки с числами
        :param calculated_cols: рассчитываемые колонки для выделения цветом
        :param str_cols: колонки с текстом
        :param non_empty_cols: обязательные к заполнению
        :param light_grey_cols: для закраса светло-серым
        :param heavy_grey_cols: для закраса темно-серым
        :param str_fixed_values: колонки с фиксированными значениями (нельзя вводить любые значения)
        :param column_names: имена колонок
        :param original_table: название таблицы (в представление выводится view)
        :param attribute: первичный ключ для таблицы
        :param attribute_col: имя атрибута ПК
        :param delete_command:
        :param logger: для печати запросов в консоль
        :param parent_window_tool: родитель для вызова зависимых от редактирования методов в окне
        :param parent: QStyledItemDelegate
        """
        super(MyDelegate, self).__init__(parent)
        self.proxy_model = proxy_model
        self.table_view = table_view
        self.db_connection = db_connection
        self.editable_columns = editable_columns
        self.command_stack = []
        self.redo_stack = []
        self.logger = logger
        self.parent_window_tool = parent_window_tool
        #  self.new_row_index = parent_window_tool.new_row_index
        self.perc_cols = perc_cols
        self.int_cols = int_cols
        self.str_cols = str_cols
        self.non_empty_cols = non_empty_cols
        self.calculated_cols = calculated_cols
        self.light_grey_cols = light_grey_cols
        self.heavy_grey_cols = heavy_grey_cols
        self.str_fixed_values = str_fixed_values
        self.column_names = column_names
        self.original_table = original_table
        self.attribute = attribute
        self.attribute_col = attribute_col
        self.delete_command = delete_command

    def setModelData(self, editor, model, index):
        #  if self.new_row_index is None:

        source_index = self.proxy_model.mapToSource(index)

        if source_index.column() not in self.editable_columns:
            return

        primary_key_index = self.proxy_model.sourceModel().index(source_index.row(), 0)
        primary_key_value = self.proxy_model.sourceModel().data(primary_key_index, Qt.DisplayRole)
        column_name_view = self.proxy_model.sourceModel().headerData(source_index.column(), Qt.Horizontal,
                                                                     Qt.DisplayRole)

        column_name = self.column_names[column_name_view]
        old_value = self.proxy_model.sourceModel().data(source_index)
        new_value = editor.text()
        print('старт делегата')
        print("ПК - ", primary_key_value, "номер ", source_index.column(), "колонка - ", column_name, "старое - ",
              old_value, "новое - ", new_value)

        if new_value == "" or new_value is None:
            if source_index.column() in self.non_empty_cols:
                print("не может быть пустым")
                print("return")
                return
            else:
                print("может быть пустым")
                new_value = None
        else:
            if source_index.column() in self.perc_cols:
                col_type, col_name = 'perc', None
            elif source_index.column() in self.int_cols:
                col_type, col_name = 'int', None
            elif source_index.column() == self.attribute_col:
                col_type, col_name = self.attribute, None
            elif source_index.column() in self.str_cols:
                col_type, col_name = 'str', column_name
            else:
                col_type, col_name = 'comment', None

            is_valid, new_value = self.validate_value(new_value, col_type, col_name)
            if not is_valid:
                return

        print("прошёл проверку", column_name_view, "колонка - ", column_name, "старое - ",
              old_value, "новое - ", new_value)
        edit_command = EditCommand(self.db_connection, primary_key_value, column_name, old_value, new_value,
                                   self.original_table, self.logger)
        self.command_stack.append(edit_command)

        query = QSqlQuery(self.db_connection)
        query.prepare(f"UPDATE {self.original_table} SET {column_name} = ? WHERE id = ?")
        query.addBindValue(new_value)
        query.addBindValue(primary_key_value)
        query.exec_()
        self.logger.info(f"Executed query delegate update: {query.lastQuery()}")

        self.proxy_model.sourceModel().select()
        self.table_view.viewport().update()
        self.parent_window_tool.load_data_stats()

    def validate_value(self, value, col_type, col_name):
        print("проверка", value, col_type, col_name)
        if col_type == 'str':
            if isinstance(value, str):
                if col_name in self.str_fixed_values.keys():
                    check_values = list(map(lambda x: x.lower(), self.str_fixed_values[col_name]))
                    value = value.lower()
                    if value in check_values:
                        return True, self.str_fixed_values[col_name][check_values.index(value)]

        elif col_type == 'int':
            if value.isdigit():
                return True, value
            else:
                try:
                    value = float(value.replace(',', '.'))
                except:
                    return False, None
                else:
                    return True, int(value)

        elif col_type == 'perc':
            print("perc", value)
            try:
                value = float(value.replace(',', '.').replace('%', ''))
            except:
                return False, None
            else:
                return True, value

        elif col_type == self.attribute:
            value = value.lower()

            query = QSqlQuery(self.db_connection)
            query.prepare(f"SELECT DISTINCT {self.attribute} FROM catalogue where {self.attribute} = ?")
            query.addBindValue(value)
            query.exec_()
            if query.next():
                return True, value
        else:
            return True, value

        return False, None

    def delete_rows(self, rows):
        delete_commands = []

        self.db_connection.transaction()

        for row in rows:
            source_row = self.proxy_model.mapToSource(self.proxy_model.index(row, 0)).row()

            row_selected = all(
                self.table_view.selectionModel().isSelected(
                    self.proxy_model.index(row, col)
                ) for col in range(self.proxy_model.columnCount()))

            if row_selected:
                primary_key = self.proxy_model.sourceModel().data(self.proxy_model.sourceModel().index(source_row, 0))

                query = QSqlQuery(self.db_connection)
                query.prepare(f"SELECT * FROM {self.original_table} WHERE id = ?")
                query.addBindValue(primary_key)
                query.exec_()
                self.logger.info(f"Executed query delegate to bufer: {query.lastQuery()}")
                if query.next():
                    row_data = [query.value(column) for column in range(query.record().count())]
                    delete_command = DeleteCommand(self.db_connection, primary_key, row_data, self.original_table,
                                                   self.delete_command, self.logger)
                    delete_commands.append(delete_command)
                    delete_command.delete()

        self.db_connection.commit()
        self.command_stack.append(delete_commands)

        self.proxy_model.sourceModel().database().commit()
        self.proxy_model.sourceModel().select()
        self.table_view.viewport().update()
        self.parent_window_tool.load_data_stats()

    def undo(self):
        if not self.command_stack:
            return

        last_commands = self.command_stack.pop()
        self.redo_stack.append(last_commands)
        if not isinstance(last_commands, list):
            last_commands = [last_commands]

        self.db_connection.transaction()

        for command in last_commands:
            command.undo()

        self.db_connection.commit()

        self.proxy_model.sourceModel().select()
        self.table_view.viewport().update()
        self.parent_window_tool.load_data_stats()

    def redo(self):
        if not self.redo_stack:
            return

        last_commands = self.redo_stack.pop()
        if not isinstance(last_commands, list):
            last_commands = [last_commands]

        self.db_connection.transaction()
        for command in last_commands:
            if isinstance(command, EditCommand):
                redo_edit_command = EditCommand(
                    self.db_connection,
                    command.primary_key,
                    command.column_name,
                    command.new_value,
                    command.old_value,
                    self.original_table,
                    self.logger,
                )
                redo_edit_command.undo()
            elif isinstance(command, DeleteCommand):
                command.delete()
            elif isinstance(command, AddCommand):
                command.redo()

        self.db_connection.commit()
        self.command_stack.append(last_commands)

        self.proxy_model.sourceModel().select()
        self.table_view.viewport().update()
        self.parent_window_tool.load_data_stats()

    def paint(self, painter, option, index):
        if index.column() in self.heavy_grey_cols:  # Замените 1 на индекс колонки, которую вы хотите окрасить
            painter.save()
            painter.fillRect(option.rect, QColor(192, 192, 192))  # Fill cell with red color
            painter.restore()
        elif index.column() in self.light_grey_cols:
            painter.save()
            painter.fillRect(option.rect, QColor(225, 225, 225))  # Fill cell with red color
            painter.restore()
        super(MyDelegate, self).paint(painter, option, index)

    # def flags(self, index):
    #     if self.parent_window_tool.new_row_index is not None and not self.parent_window_tool.is_ready_to_insert():
    #         if index.row() == self.parent_window_tool.new_row_index:
    #             return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable
    #         else:
    #             return Qt.ItemIsSelectable
    #     return super().flags(index)


class EditCommand:
    """
    для реализации UNDO/REDO. Объекты класса хранятся в памяти
    """
    def __init__(self, connection, primary_key, column_name, old_value, new_value, original_table, logger):
        self.connection = connection
        self.primary_key = primary_key
        self.column_name = column_name
        self.logger = logger
        self.original_table = original_table
        # print(f"editclass_old:{old_value}")
        # print(f"editclass_new:{old_value}")
        print("old_value before - ", old_value)
        if old_value == "":
            self.old_value = None
        elif old_value and isinstance(old_value, str):
            if self.column_name in ['comment_sti', 'comment_lti', 'lti_type', 'cycle_launch_frequency',
                                    'payout_frequency_within_one_cycle']:
                self.old_value = old_value
            else:
                self.old_value = old_value.replace(' ', '')
                if '%' in self.old_value:
                    self.old_value = float(self.old_value.replace('%', '')) / 100
        else:
            self.old_value = old_value
        print("old_value after - ", self.old_value)
        print("new_value before - ", new_value)
        if new_value == "":
            self.new_value = None
        elif new_value and isinstance(new_value, str):
            if self.column_name in ['comment_sti', 'comment_lti', 'lti_type', 'cycle_launch_frequency',
                                    'payout_frequency_within_one_cycle']:
                self.new_value = new_value
            else:
                self.new_value = new_value.replace(' ', '')
                if '%' in self.new_value:
                    self.new_value = float(self.new_value.replace('%', '')) / 100
        else:
            self.new_value = new_value
        print("new_value after - ", self.new_value)

    def undo(self):
        query = QSqlQuery(self.connection)
        query.prepare(f"UPDATE {self.original_table} SET {self.column_name} = ? WHERE id = ?")
        query.addBindValue(self.old_value)
        query.addBindValue(self.primary_key)
        query.exec_()
        self.logger.info(f"Executed query undo edit class: {query.lastQuery()}")


class DeleteCommand:
    """
    для реализации UNDO/REDO. Объекты класса хранятся в памяти
    """
    def __init__(self, connection, primary_key, row_data, original_table, delete_command, logger):
        self.connection = connection
        self.primary_key = primary_key
        self.row_data = row_data
        self.logger = logger
        self.original_table = original_table
        self.delete_command = delete_command

    def delete(self):
        query = QSqlQuery(self.connection)
        query.prepare(f"DELETE FROM {self.original_table} WHERE id = ?")
        query.addBindValue(self.primary_key)
        query.exec_()
        self.logger.info(f"Executed query delete delete class: {query.lastQuery()}")

    def undo(self):
        query = QSqlQuery(self.connection)
        query.prepare(self.delete_command)
        for value in self.row_data:
            print(value)
            if value == "":
                query.addBindValue(QVariant())
            else:
                query.addBindValue(value)
        query.exec_()
        self.logger.info(f"Executed query undo delete class: {query.lastQuery()}")


class AddCommand:
    """
    для реализации UNDO/REDO. Объекты класса хранятся в памяти
    """
    def __init__(self, connection, primary_key, row_data, original_table, logger):
        self.connection = connection
        self.primary_key = primary_key
        self.row_data = row_data
        self.logger = logger
        self.original_table = original_table

    def undo(self):
        query = QSqlQuery(self.connection)
        query.prepare(f"DELETE FROM {self.original_table} WHERE id = ?")
        query.addBindValue(self.primary_key)
        query.exec_()
        self.logger.info(f"Executed query undo add class: {query.lastQuery()}")

    def redo(self):
        query = QSqlQuery(self.connection)
        query.prepare(
            f"""INSERT INTO {self.original_table} (id, client_id, sti_bonus_eligibility_position, 
            sti_bonus_eligibility_person, has_lti_programs, lti_bonus_eligibility_person, base_pay)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""")
        for value in self.row_data:
            if value == "":
                query.addBindValue(QVariant())
            else:
                query.addBindValue(value)
        query.exec_()
        self.logger.info(f"Executed query redo add class: {query.lastQuery()}")
