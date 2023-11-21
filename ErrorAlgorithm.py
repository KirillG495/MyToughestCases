from PyQt5 import QtCore, QtGui, QtWidgets
from libs import *
import numpy as np
from functions_for_error_algoritm import *


class Worker_Error_Algoritm(QThread):
    """
    Алгоритм автоисправления ошибок в данных. Выполняется в отдельном потоке
    """
    progress = pyqtSignal(str)
    error = pyqtSignal(str)
    info = pyqtSignal(str)
    signal = pyqtSignal()
    alogritm_finished = pyqtSignal(object, object, object, object, object)

    def __init__(self, db_path, location_ids, sector, companies_names, interval):
        """
        :param db_path: путь к БД
        :param location_ids: список локаций (для запроса)
        :param sector: выбранный сектор (для запроса)
        :param companies_names: выбранные компании (для запроса)
        :param interval: выбранный пользователем интервал (для границ межквартильного размаха)
        """
        super().__init__()
        self.db_path = db_path
        print(self.db_path)
        self.sector = sector
        self.companies_names = companies_names
        self.location_ids = location_ids
        self.location_ids_placeholders = ','.join(['?' for _ in self.location_ids])
        self.companies_names_placeholders = ','.join(['?' for _ in self.companies_names])

        # self.number_of_deleted_rows = 0
        # self.number_of_closed_ab = 0

        self.interval = interval

        self.action_counts = {'update_code': 0, 'delete': 0, 'update_close_ab': 0, 'update_change_yes': 0}
        self.db_changes = []

        self._is_running = True

    def stop(self):
        self._is_running = False
        return

    def run(self):
        while self._is_running:
            def trace_func(statement):
                print(f"SQL: {statement}")

            """
            вспомогательные для ошибок 1 2 и 3
            """

            # Function to find the closest parent available in the dataframe
            def find_closest_parent(code):
                parent = None
                valid_parent = False

                while True:
                    # Check if the code is present in the hierarchy dataframe
                    if code in hierarchy_df['code'].values:
                        # Get parent from the hierarchy dataframe
                        parent = hierarchy_df.loc[hierarchy_df['code'] == code, 'parent'].values[0]
                    else:
                        break

                    if parent is None or pd.isna(parent):
                        break

                    if parent in result_df['code'].values:
                        valid_parent = True
                        break

                    code = parent

                return parent if valid_parent else None

            def recalculate_stats(df_rawdata):
                trimmed_df = trimming(df_rawdata)

                index = trimmed_df.code.unique().tolist()
                stats = pd.DataFrame(columns=columns, index=index)

                trimmed_df = trimmed_df[['code', 'client_id', 'AGP', 'AFP']]
                result_df = fill_cells_optimized(trimmed_df, stats)

                result_df = result_df.reset_index().rename(columns={'index': 'code'})

                return result_df

            def recalculate_stats_hier(df_rawdata, hierarchy_df):
                trimmed_df = trimming(df_rawdata)

                index = trimmed_df.code.unique().tolist()
                stats = pd.DataFrame(columns=columns, index=index)

                trimmed_df = trimmed_df[['code', 'client_id', 'AGP', 'AFP']]
                result_df = fill_cells_optimized(trimmed_df, stats)

                result_df = result_df.reset_index().rename(columns={'index': 'code'})
                result_df['L_IQR_AGP'] = result_df['AGP_25'] - self.interval * (
                        result_df['AGP_75'] - result_df['AGP_25'])
                result_df['R_IQR_AGP'] = result_df['AGP_75'] + self.interval * (
                        result_df['AGP_75'] - result_df['AGP_25'])
                result_df['L_IQR_AFP'] = result_df['AFP_25'] - self.interval * (
                        result_df['AFP_75'] - result_df['AFP_25'])
                result_df['R_IQR_AFP'] = result_df['AFP_75'] + self.interval * (
                        result_df['AFP_75'] - result_df['AFP_25'])
                result_df['parent'] = result_df['code'].apply(find_closest_parent)

                return result_df

            def append_error(error_codes, parent_code, column, description):
                if parent_code in error_codes:
                    current_value = error_codes[parent_code]
                    if isinstance(current_value, str):
                        error_codes[parent_code] = {column: description}
                    else:
                        error_codes[parent_code][column] = description
                else:
                    error_codes[parent_code] = {column: description}
                return error_codes

            def interval_variation_series(df, column_name):
                data = df[column_name]
                xmax = data.max()
                xmin = data.min()
                R = xmax - xmin
                n = len(data)
                k = math.floor(1 + 3.22 * math.log10(n))
                h = round((xmax - xmin) / k, 2)
                intervals = [xmin + i * h for i in range(k + 1)]
                if xmax > intervals[-1]:
                    intervals[-1] = xmax
                interval_counts = pd.cut(data, bins=intervals, include_lowest=True,
                                         right=True).value_counts().sort_index()

                return intervals, interval_counts

            def median_interval(intervals, counts):
                total = sum(counts)
                median_position = math.ceil(total / 2)
                cumulative_count = 0

                for i, count in enumerate(counts):
                    cumulative_count += count
                    if cumulative_count >= median_position:
                        return intervals[i], intervals[i + 1]

            self.progress.emit("Начинаем вычищать ошибки...")
            conn = sqlite3.connect(self.db_path)
            conn.set_trace_callback(trace_func)

            df = pd.read_sql_query(f"""select * from report_view where id in (select id from anketa where is_rematch = 1 
            and client_id in ({self.companies_names_placeholders}) and location_id in ({self.location_ids_placeholders}) 
            and code in (SELECT code from anketa WHERE location_id in ({self.location_ids_placeholders}) and client_id in 
            ({self.companies_names_placeholders}) and is_rematch = 1 group by code  having (COUNT(DISTINCT client_id))>2))""",
                                   conn,
                                   params=self.companies_names + self.location_ids + self.location_ids + self.companies_names)
            self.total_rows_solv = df.shape[0]

            self.total_rows = pd.read_sql_query(f"""select count(id) as id from anketa where is_rematch = 1 
            and client_id in ({self.companies_names_placeholders}) and location_id in ({self.location_ids_placeholders})""",
                                                conn, params=self.companies_names + self.location_ids).id.tolist()[0]

            if not self._is_running:
                return

            def process_group(group_data):
                (job_id, client_id), group = group_data
                if len(group) > 5:
                    percentiles = [10, 25, 50, 75, 90]
                    indicators = ['AGP', "AFP"]
                    percentiles_dict = {}

                    for indicator in indicators:
                        if group[indicator].count() > 5:
                            percentiles_dict[indicator] = nanpercentile(group[indicator], percentiles)
                        else:
                            non_nan_values = group[indicator].dropna().reset_index(drop=True)
                            missing_nans = 5 - len(non_nan_values)
                            percentiles_dict[indicator] = non_nan_values.append(
                                pd.Series([nan] * missing_nans)).reset_index(drop=True)

                    group_df = pd.DataFrame({
                        "code": [job_id] * 5,
                        "client_id": [client_id] * 5,
                        "id": [1, 2, 3, 4, 5],
                        "AGP": percentiles_dict["AGP"],
                        "AFP": percentiles_dict["AFP"],
                    })
                else:
                    group_df = group
                return group_df

            def trimming(df):
                grouped = df.groupby(["code", "client_id"])

                with ThreadPoolExecutor() as executor:
                    # Create a partial function with fixed df_columns argument
                    process_group_partial = partial(process_group)

                    # Execute process_group_partial in parallel for each group in grouped
                    results = executor.map(process_group_partial, grouped)

                # Concatenate the results and reset the index
                new_df = pd.concat(results).reset_index(drop=True)
                return new_df

            try:
                trimmed_df = trimming(df)
            except:
                trimmed_df = df

            columns = ['AGP_10', 'AGP_25', 'AGP_50', 'AGP_75', 'AGP_90', 'AGP_avg',
                       'AFP_10', 'AFP_25', 'AFP_50', 'AFP_75', 'AFP_90', 'AFP_avg']
            index = trimmed_df.code.unique().tolist()
            stats = pd.DataFrame(columns=columns, index=index)
            trimmed_df = trimmed_df[['code', 'client_id', 'AGP', 'AFP']]

            def fill_cells_optimized(trimmed_df, stats):
                grouped_df = trimmed_df.groupby('code')
                for code, group in grouped_df:
                    for attribute in trimmed_df.columns[2:]:
                        num = group[attribute].count()
                        cnt_clients = group.dropna(subset=[attribute])['client_id'].nunique()
                        if cnt_clients > 2:
                            stats.loc[code, f'{attribute}_avg'] = group[attribute].mean()
                            if 4 <= num <= 5:
                                stats.loc[code, f'{attribute}_50'] = group[attribute].quantile(q=0.5)
                            elif 6 <= num <= 7:
                                stats.loc[code, f'{attribute}_25'] = group[attribute].quantile(q=0.25)
                                stats.loc[code, f'{attribute}_50'] = group[attribute].quantile(q=0.5)
                                stats.loc[code, f'{attribute}_75'] = group[attribute].quantile(q=0.75)
                            elif num >= 8:
                                stats.loc[code, f'{attribute}_10'] = group[attribute].quantile(q=0.1)
                                stats.loc[code, f'{attribute}_25'] = group[attribute].quantile(q=0.25)
                                stats.loc[code, f'{attribute}_50'] = group[attribute].quantile(q=0.5)
                                stats.loc[code, f'{attribute}_75'] = group[attribute].quantile(q=0.75)
                                stats.loc[code, f'{attribute}_90'] = group[attribute].quantile(q=0.9)
                return stats

            fill_cells_optimized(trimmed_df, stats)

            result_df = stats.reset_index().rename(columns={'index': 'code'})
            result_df['L_IQR_AGP'] = result_df['AGP_25'] - self.interval * (result_df['AGP_75'] - result_df['AGP_25'])
            result_df['R_IQR_AGP'] = result_df['AGP_75'] + self.interval * (result_df['AGP_75'] - result_df['AGP_25'])
            result_df['L_IQR_AFP'] = result_df['AFP_25'] - self.interval * (result_df['AFP_75'] - result_df['AFP_25'])
            result_df['R_IQR_AFP'] = result_df['AFP_75'] + self.interval * (result_df['AFP_75'] - result_df['AFP_25'])
            hierarchy_df = pd.read_sql_query("SELECT * FROM hierarchy", conn)
            result_df['parent'] = result_df['code'].apply(find_closest_parent)

            agg_index = result_df.code.unique().tolist()
            agg_codes_placeholders = ','.join(['?' for _ in agg_index])
            code_info = pd.read_sql_query(f"""select code, function, division, subfunction, level as level_num
                                                      from catalogue where code in ({agg_codes_placeholders})""", conn,
                                          params=agg_index)

            # def extract_digits_or_missing(row):
            #     if pd.isna(row) or row == '' or row == 0:
            #         return np.nan
            #     else:
            #         return ''.join(re.findall('\d', row))
            #
            # code_info['level_num'] = code_info['level'].apply(extract_digits_or_missing)
            # code_info['level_num'] = code_info['level_num'].astype(float).astype('Int64')
            # code_info = code_info.drop(columns=["level"])

            result_df = pd.merge(result_df, code_info, on='code', how='left')
            result_df = result_df.sort_values(by=['level_num', 'function', 'division', 'subfunction'],
                                              ignore_index=True)
            result_df = result_df.drop(columns=['level_num', 'function', 'division', 'subfunction'])

            """
            новый алгоритм иерархии 17.07
            """

            def new_hierarchy_algoritm(df, df_stats, hierarchy_df):
                df_rawdata = df.copy()
                result_df = df_stats.copy()
                # conn.set_trace_callback(trace_func)

                columns_to_check = ['AGP_10', 'AGP_25', 'AGP_75', 'AGP_90', 'AGP_50', 'AGP_avg']

                iserror = False

                for i, row in result_df.iterrows():
                    code = result_df.iloc[i]['code']
                    parent_code = result_df.iloc[i]['parent']

                    if code == "MNG_G_1_1" or pd.isna(parent_code):  # если не 1 уровень и есть руководитель
                        continue

                    parent_stats = result_df[result_df['code'] == parent_code].iloc[0]

                    for col in columns_to_check:
                        list_of_recurring_ids = []
                        fifth_check = False
                        if not (pd.isna(parent_stats[col]) or pd.isna(result_df.loc[i, col])) and (
                                parent_stats[col] < result_df.loc[i, col]):  # если есть ошибка

                            lower_bound_parent, upper_bound_parent, median_bound_parent = count_intervals(df_rawdata,
                                                                                                          parent_code)  # рассчитываем интервалы руководителя
                            lower_bound_current_code, upper_bound_current_code, median_bound_current_code = count_intervals(
                                df_rawdata, code)  # рассчитываем интервалы текущего кода

                            step_3_2_done = True  # для переключения между циклами в 50 перцентиле и среднем

                            while True:
                                if fifth_check:
                                    if len(list_of_recurring_ids) > 4 and len(set(list_of_recurring_ids[-4:])) == 1:
                                        break
                                current_value = result_df.loc[result_df['code'] == code, col].values[0]
                                parent_value = result_df.loc[result_df['code'] == parent_code, col].values[0]

                                if (pd.isna(current_value) or pd.isna(
                                        parent_value)) or current_value < parent_value:  # делаем проверку неравенства, если ошибка исправилась то выход из проверки
                                    break  # выход

                                if col in ['AGP_10',
                                           'AGP_25']:  # в зависимости от перцентиля подбираем код для исправления
                                    sending_code = parent_code
                                    receiving_code = code
                                    min_search = True  # Минимальное значение

                                elif col in ['AGP_75', 'AGP_90']:
                                    sending_code = code
                                    receiving_code = parent_code
                                    min_search = False  # Максимальное значение

                                elif col in ['AGP_50', 'AGP_avg']:
                                    fifth_check = True
                                    if step_3_2_done:
                                        sending_code = parent_code
                                        receiving_code = code
                                        min_search = True  # Минимальное значение
                                        step_3_2_done = False
                                    else:
                                        sending_code = code
                                        receiving_code = parent_code
                                        min_search = False  # Максимальное значение
                                        step_3_2_done = True

                                df_no_nan = df_rawdata[df_rawdata['code'] == sending_code].dropna(
                                    subset=['AGP'])  # отбросить строки, в которых 'AGP' равно NaN

                                if len(df_no_nan) == 0:  # если нет AGP
                                    break  # выход

                                # Сохранить только те записи от тех клиентов, которые представлены более чем одной записью
                                df_multi_client = df_no_nan.groupby('client_id').filter(lambda x: len(x) > 1)

                                if len(df_multi_client) == 0:
                                    break  # выход

                                if min_search:  # если работа с минимальным значением
                                    sending_AGP = df_multi_client.loc[df_multi_client['AGP'].idxmin()]['AGP']
                                    sending_id = df_multi_client.loc[df_multi_client['AGP'].idxmin()]['id']
                                    edit_index = df_multi_client[df_multi_client['AGP'] == sending_AGP].index[0]
                                    print(f"{col} {receiving_code} - {current_value} > {sending_code} - {parent_value}")
                                    print(
                                        f"Проверка руководителя {sending_code}: min parent {sending_AGP} > min curr {lower_bound_current_code} and min parent {sending_AGP} < median curr {median_bound_current_code}")
                                    if sending_AGP > lower_bound_current_code and sending_AGP < median_bound_current_code:  # если значение входит в интервал
                                        action_rematch = True  # то ремэтч
                                        print(f"Входит. замена кода на current {receiving_code}")
                                    else:
                                        action_rematch = False  # иначе удаляем
                                        print("не входит. Удаление")

                                else:  # если работа с максимальным значением
                                    sending_AGP = df_multi_client.loc[df_multi_client['AGP'].idxmax()]['AGP']
                                    sending_id = df_multi_client.loc[df_multi_client['AGP'].idxmax()]['id']
                                    edit_index = df_multi_client[df_multi_client['AGP'] == sending_AGP].index[0]
                                    print(f"{col} {sending_code} - {current_value} > {receiving_code} - {parent_value}")
                                    print(
                                        f"Проверка текущего кода {sending_code}: max current {sending_AGP} > median parent {median_bound_parent} and max current {sending_AGP} < max parent {upper_bound_parent}")
                                    if sending_AGP > median_bound_parent and sending_AGP < upper_bound_parent:  # если значение входит в интервал
                                        action_rematch = True  # то ремэтч
                                        print(f"Входит. замена кода на parent {receiving_code}")
                                    else:
                                        action_rematch = False  # иначе удаляем
                                        print("не входит. Удаление")

                                if action_rematch:
                                    if fifth_check:
                                        list_of_recurring_ids.append(sending_id)
                                    self.db_changes.append(("update_code", receiving_code, int(sending_id)))
                                    print(receiving_code, int(sending_id))
                                    # query = "UPDATE anketa SET code = ? WHERE id = ?"
                                    # self.number_of_closed_ab += 1
                                    # params = (receiving_code, int(sending_id),)
                                    df_rawdata.loc[edit_index, 'code'] = receiving_code
                                    for editing_code in [sending_code, receiving_code]:
                                        updated_stats = recalculate_stats_hier(
                                            df_rawdata[df_rawdata['code'] == editing_code],
                                            hierarchy_df)
                                        index_to_replace = result_df[result_df['code'] == editing_code].index[0]
                                        result_df.loc[index_to_replace] = updated_stats.loc[0]

                                else:
                                    self.db_changes.append(("delete", None, int(sending_id)))
                                    # query = "DELETE FROM anketa WHERE id = ?"
                                    # self.number_of_deleted_rows += 1
                                    # params = (int(sending_id),)
                                    df_rawdata.drop(edit_index, inplace=True)
                                    updated_stats = recalculate_stats_hier(
                                        df_rawdata[df_rawdata['code'] == sending_code],
                                        hierarchy_df)
                                    index_to_replace = result_df[result_df['code'] == sending_code].index[0]
                                    result_df.loc[index_to_replace] = updated_stats.loc[0]

                                iserror = True

                                # cursor = conn.cursor()
                                # cursor.execute(query, params)
                                # conn.commit()

                                continue  # проверяем заново
                # print(f"удалено - {self.number_of_deleted_rows}, обновлено - {self.number_of_closed_ab}")
                return df_rawdata, result_df, iserror

            """
            Новый алгоритм 31.05
            """

            def hierarchy_errors(df, df_stats, hierarchy_df, error_codes, error_codes_after_fixing):
                df_rawdata = df.copy()
                result_df = df_stats.copy()
                warnings.filterwarnings('ignore')

                columns_to_check = ['AGP_10', 'AGP_25', 'AGP_75', 'AGP_90', 'AGP_50', 'AGP_avg',
                                    'AFP_10', 'AFP_25', 'AFP_75', 'AFP_90', 'AFP_50', 'AFP_avg']

                error_flag = True  # Инициализируем error_flag как True для входа в цикл while
                step_3_2_done = False  # для переключения между циклами. Подумать
                step_6_2_done = False  # для переключения между циклами. Подумать
                iserror = False

                while error_flag:
                    error_flag = False  # Сброс флага
                    for i, row in result_df.iterrows():
                        code = result_df.iloc[i]['code']
                        parent_code = result_df.iloc[i]['parent']

                        if code == "MNG_G_1_1" or pd.isna(parent_code):
                            continue

                        parent_stats = result_df[result_df['code'] == parent_code].iloc[0]

                        for col in columns_to_check:
                            if not (pd.isna(parent_stats[col]) or pd.isna(result_df.loc[i, col])) and (
                                    parent_stats[col] < result_df.loc[i, col]):
                                if col in ['AGP_10', 'AGP_25', 'AGP_75', 'AGP_90']:
                                    if col in ['AGP_10', 'AGP_25']:
                                        border = 'L_IQR_AGP'
                                        code_for_edit = parent_code
                                        err_rawdata = df_rawdata[
                                            (df_rawdata['code'] == code_for_edit) & (
                                                    df_rawdata['AGP'] < result_df.loc[i, border])]
                                        if err_rawdata.empty:
                                            error_codes_after_fixing = append_error(error_codes_after_fixing,
                                                                                    code_for_edit,
                                                                                    col,
                                                                                    f"В {code_for_edit} нет записей с AGP < {border} {result_df.loc[i, border]}")
                                        else:
                                            AGP_for_edit = err_rawdata['AGP'].min()
                                            print(
                                                f"{i}, {datetime.now()}. Ошибка в {code_for_edit} {col}. {parent_stats[col]}  < {result_df.loc[i, col]}, удалится - {AGP_for_edit}, порог у подчинённого {code} - {result_df.loc[i, border]}")
                                            error_codes = append_error(error_codes, code_for_edit, col, description="")
                                    else:
                                        border = 'R_IQR_AGP'
                                        code_for_edit = code
                                        err_rawdata = df_rawdata[
                                            (df_rawdata['code'] == code_for_edit) & (
                                                    df_rawdata['AGP'] > result_df.loc[i, border])]

                                        if err_rawdata.empty:
                                            error_codes_after_fixing = append_error(error_codes_after_fixing,
                                                                                    parent_code,
                                                                                    col,
                                                                                    f"В {code_for_edit} нет записей с AGP > {border} {result_df.loc[i, border]}")
                                        else:
                                            AGP_for_edit = err_rawdata['AGP'].max()
                                            print(
                                                f"{i}, {datetime.now()}. Ошибка в {parent_code} {col}. {parent_stats[col]}  < {result_df.loc[i, col]}, удалится - {AGP_for_edit}, порог у подчинённого {code_for_edit} - {result_df.loc[i, border]}")
                                            error_codes = append_error(error_codes, parent_code, col, description="")

                                    if not err_rawdata.empty:
                                        error_flag = True
                                        iserror = True
                                        first_index = err_rawdata[err_rawdata['AGP'] == AGP_for_edit].index[
                                            0]  # удаление по строке
                                        id_for_query = int(err_rawdata.loc[first_index, 'id'])
                                        self.db_changes.append(("delete", None, id_for_query))
                                        # query = "delete from anketa where id = ?"
                                        # #print(type(id_for_query))
                                        # cursor = conn.cursor()
                                        # cursor.execute(query, (id_for_query,))
                                        # conn.commit()
                                        # self.number_of_deleted_rows += 1
                                        df_rawdata = df_rawdata.drop(first_index)  # удаление по строке
                                        updated_stats = recalculate_stats_hier(
                                            df_rawdata[df_rawdata['code'] == code_for_edit],
                                            hierarchy_df)
                                        index_to_replace = result_df[result_df['code'] == code_for_edit].index[0]
                                        result_df.loc[index_to_replace] = updated_stats.loc[0]
                                        break  # Прерывает внутренний цикл for при обнаружении ошибки

                                if col in ['AGP_50', 'AGP_avg']:
                                    # проверка только если есть неравенство, если у рук-ля и подчиненного больше 3-х записей (чтобы код не закрылся) и если оба пок-ля раскрыты.
                                    if ((df_rawdata[df_rawdata['code'] == parent_code].shape[0] > 3) & (
                                            df_rawdata[df_rawdata['code'] == code].shape[0] > 3)):

                                        if not step_3_2_done:
                                            # если интервалы раскрыты то ошибочные данные берутся по интервалу
                                            if not (np.isnan(result_df.loc[i, 'L_IQR_AGP']) or np.isnan(
                                                    result_df.loc[i, 'R_IQR_AGP'])):
                                                err_rawdata = df_rawdata[(df_rawdata['code'] == parent_code) & (
                                                        df_rawdata['AGP'] < result_df.loc[i, 'L_IQR_AGP'])]
                                            # иначе просто минимальное значение в ошибочном коде
                                            else:
                                                err_rawdata = df_rawdata[df_rawdata['code'] == parent_code]

                                            if not err_rawdata.empty:
                                                error_flag = True
                                                iserror = True
                                                min_AGP = err_rawdata['AGP'].min()
                                                print(
                                                    f"{i} min, {datetime.now()}. Ошибка в {parent_code} {col}. {parent_stats[col]}  < {result_df.loc[i, col]}, удалится - {min_AGP}, порог у подчинённого {code} - {result_df.loc[i, 'L_IQR_AGP']}")  # почему i равен нулю?
                                                first_min_AGP_index = err_rawdata[err_rawdata['AGP'] == min_AGP].index[
                                                    0]  # удаление по строке
                                                id_for_query = int(err_rawdata.loc[first_min_AGP_index, 'id'])
                                                self.db_changes.append(("delete", None, id_for_query))
                                                # query = "delete from anketa where id = ?"
                                                # #print(type(id_for_query))
                                                # cursor = conn.cursor()
                                                # cursor.execute(query, (id_for_query,))
                                                # conn.commit()
                                                # self.number_of_deleted_rows += 1
                                                df_rawdata = df_rawdata.drop(first_min_AGP_index)  # удаление по строке
                                                updated_stats = recalculate_stats_hier(
                                                    df_rawdata[df_rawdata['code'] == parent_code], hierarchy_df)
                                                index_to_replace = result_df[result_df['code'] == parent_code].index[0]
                                                result_df.loc[index_to_replace] = updated_stats.loc[0]
                                                step_3_2_done = True
                                                error_codes = append_error(error_codes, parent_code, col,
                                                                           description="")
                                            else:
                                                error_codes_after_fixing = append_error(error_codes_after_fixing,
                                                                                        parent_code,
                                                                                        col,
                                                                                        f"В {parent_code} нет записей с AGP < L_IQR_AGP {result_df.loc[i, 'L_IQR_AGP']}")

                                        else:
                                            # если интервалы раскрыты то ошибочные данные берутся по интервалу
                                            if not (np.isnan(result_df.loc[i, 'L_IQR_AGP']) or np.isnan(
                                                    result_df.loc[i, 'R_IQR_AGP'])):
                                                err_rawdata = df_rawdata[(df_rawdata['code'] == code) & (
                                                        df_rawdata['AGP'] > result_df.loc[i, 'R_IQR_AGP'])]
                                            # иначе просто минимальное значение в ошибочном коде
                                            else:
                                                err_rawdata = df_rawdata[df_rawdata['code'] == code]

                                            if not err_rawdata.empty:
                                                error_flag = True
                                                iserror = True
                                                max_AGP = err_rawdata['AGP'].max()
                                                print(
                                                    f"{i} max, {datetime.now()}. Ошибка в {parent_code} {col}. {parent_stats[col]}  < {result_df.loc[i, col]}, удалится - {max_AGP}, порог у подчинённого {code} - {result_df.loc[i, 'R_IQR_AGP']}")
                                                first_max_AGP_index = err_rawdata[err_rawdata['AGP'] == max_AGP].index[
                                                    0]  # удаление по строке
                                                id_for_query = int(err_rawdata.loc[first_max_AGP_index, 'id'])
                                                self.db_changes.append(("delete", None, id_for_query))
                                                # query = "delete from anketa where id = ?"
                                                # #print(type(id_for_query))
                                                # cursor = conn.cursor()
                                                # cursor.execute(query, (id_for_query,))
                                                # conn.commit()
                                                # self.number_of_deleted_rows += 1
                                                df_rawdata = df_rawdata.drop(first_max_AGP_index)  # удаление по строке
                                                updated_stats = recalculate_stats_hier(
                                                    df_rawdata[df_rawdata['code'] == code],
                                                    hierarchy_df)
                                                index_to_replace = result_df[result_df['code'] == code].index[0]
                                                result_df.loc[index_to_replace] = updated_stats.loc[0]
                                                step_3_2_done = False
                                                error_codes = append_error(error_codes, parent_code, col,
                                                                           description="")
                                                break
                                            else:
                                                error_codes_after_fixing = append_error(error_codes_after_fixing,
                                                                                        parent_code, col,
                                                                                        f"В {code} нет записей с AGP > R_IQR_AGP {result_df.loc[i, 'R_IQR_AGP']}")
                                    else:
                                        if (df_rawdata[df_rawdata['code'] == parent_code].shape[0] <= 3):
                                            error_codes_after_fixing = append_error(error_codes_after_fixing,
                                                                                    parent_code,
                                                                                    col, f"В {parent_code} <=3 записей")
                                        else:
                                            error_codes_after_fixing = append_error(error_codes_after_fixing,
                                                                                    parent_code,
                                                                                    col, f"В {code} <=3 записей")

                                if col in ['AFP_10', 'AFP_25', 'AFP_75', 'AFP_90']:
                                    if col in ['AFP_10', 'AFP_25']:
                                        border = 'L_IQR_AFP'
                                        code_for_edit = parent_code
                                        err_rawdata = df_rawdata[
                                            (df_rawdata['code'] == code_for_edit) & (
                                                    df_rawdata['AFP'] < result_df.loc[i, border])]
                                        if err_rawdata.empty:
                                            error_codes_after_fixing = append_error(error_codes_after_fixing,
                                                                                    code_for_edit,
                                                                                    col,
                                                                                    f"В {code_for_edit} нет записей с AFP < {border} {result_df.loc[i, border]}")
                                        else:
                                            AFP_for_edit = err_rawdata['AFP'].min()
                                            print(
                                                f"{i}, {datetime.now()}. Ошибка в {code_for_edit} {col}. {parent_stats[col]}  < {result_df.loc[i, col]}, удалится - {AFP_for_edit}, порог у подчинённого {code} - {result_df.loc[i, border]}")
                                            error_codes = append_error(error_codes, code_for_edit, col, description="")

                                    else:
                                        border = 'R_IQR_AFP'
                                        code_for_edit = code
                                        err_rawdata = df_rawdata[
                                            (df_rawdata['code'] == code_for_edit) & (
                                                    df_rawdata['AFP'] > result_df.loc[i, border])]
                                        if err_rawdata.empty:
                                            error_codes_after_fixing = append_error(error_codes_after_fixing,
                                                                                    parent_code,
                                                                                    col,
                                                                                    f"В {code_for_edit} нет записей с AFP > {border} {result_df.loc[i, border]}")
                                        else:
                                            AFP_for_edit = err_rawdata['AFP'].max()
                                            print(
                                                f"{i}, {datetime.now()}. Ошибка в {parent_code} {col}. {parent_stats[col]}  < {result_df.loc[i, col]}, удалится - {AFP_for_edit}, порог у подчинённого {code_for_edit} - {result_df.loc[i, border]}")
                                            error_codes = append_error(error_codes, parent_code, col, description="")

                                    if not err_rawdata.empty:
                                        error_flag = True
                                        iserror = True
                                        first_index = err_rawdata[err_rawdata['AFP'] == AFP_for_edit].index[
                                            0]  # удаление по строке
                                        id_for_query = int(err_rawdata.loc[first_index, 'id'])
                                        self.db_changes.append(("update_close_ab", None, id_for_query))
                                        # query = "UPDATE anketa SET bonus_fact = NULL, comission_fact = NULL WHERE id = ?"
                                        # #print(type(id_for_query))
                                        # cursor = conn.cursor()
                                        # cursor.execute(query, (id_for_query,))
                                        # conn.commit()
                                        # self.number_of_closed_ab +=1
                                        df_rawdata.loc[
                                            first_index, ['bonus_fact', 'comission_fact',
                                                          'AFP']] = np.nan  # удаление по строке
                                        updated_stats = recalculate_stats_hier(
                                            df_rawdata[df_rawdata['code'] == code_for_edit],
                                            hierarchy_df)
                                        index_to_replace = result_df[result_df['code'] == code_for_edit].index[0]
                                        result_df.loc[index_to_replace] = updated_stats.loc[0]
                                        break  # Прерывает внутренний цикл for при обнаружении ошибки

                                if col in ['AFP_50', 'AFP_avg']:
                                    # проверка только если есть неравенство, если у рук-ля и подчиненного больше 3-х записей (чтобы код не закрылся) и если оба пок-ля раскрыты.
                                    if ((df_rawdata[df_rawdata['code'] == parent_code].shape[0] > 3) & (
                                            df_rawdata[df_rawdata['code'] == code].shape[0] > 3)):

                                        if not step_6_2_done:
                                            if not (np.isnan(result_df.loc[i, 'L_IQR_AFP']) or np.isnan(
                                                    result_df.loc[i, 'R_IQR_AFP'])):
                                                err_rawdata = df_rawdata[(df_rawdata['code'] == parent_code) & (
                                                        df_rawdata['AFP'] < result_df.loc[
                                                    i, 'L_IQR_AFP'])]  # если интервалы раскрыты то ошибочные данные берутся по интервалу
                                            else:
                                                err_rawdata = df_rawdata[df_rawdata[
                                                                             'code'] == parent_code]  # иначе просто минимальное значение в ошибочном коде

                                            if not err_rawdata.empty:
                                                error_flag = True
                                                iserror = True
                                                min_AFP = err_rawdata['AFP'].min()
                                                print(
                                                    f"{i} min, {datetime.now()}. Ошибка в {parent_code} {col}. {parent_stats[col]}  < {result_df.loc[i, col]}, удалится - {min_AFP}, порог у подчинённого {code} - {result_df.loc[i, 'L_IQR_AFP']}")
                                                # удаление по строке
                                                if np.isnan(min_AFP):
                                                    first_min_AFP_index = \
                                                        err_rawdata[err_rawdata['AFP'].isnull()].index[0]
                                                else:
                                                    first_min_AFP_index = \
                                                        err_rawdata[err_rawdata['AFP'] == min_AFP].index[
                                                            0]
                                                id_for_query = int(err_rawdata.loc[first_min_AFP_index, 'id'])
                                                self.db_changes.append(("update_close_ab", None, id_for_query))
                                                # query = "UPDATE anketa SET bonus_fact = NULL, comission_fact = NULL WHERE id = ?"
                                                # #print(type(id_for_query))
                                                # cursor = conn.cursor()
                                                # cursor.execute(query, (id_for_query,))
                                                # conn.commit()
                                                # self.number_of_closed_ab += 1
                                                df_rawdata.loc[
                                                    first_min_AFP_index, ['bonus_fact', 'comission_fact',
                                                                          'AFP']] = np.nan
                                                updated_stats = recalculate_stats_hier(
                                                    df_rawdata[df_rawdata['code'] == parent_code], hierarchy_df)
                                                index_to_replace = result_df[result_df['code'] == parent_code].index[0]
                                                result_df.loc[index_to_replace] = updated_stats.loc[0]

                                                step_6_2_done = True
                                                error_codes = append_error(error_codes, parent_code, col,
                                                                           description="")

                                            else:
                                                error_codes_after_fixing = append_error(error_codes_after_fixing,
                                                                                        parent_code,
                                                                                        col,
                                                                                        f"В {parent_code} нет записей с AFP < L_IQR_AFP {result_df.loc[i, 'L_IQR_AFP']}")

                                        else:
                                            # если интервалы раскрыты то ошибочные данные берутся по интервалу
                                            if not (np.isnan(result_df.loc[i, 'L_IQR_AFP']) or np.isnan(
                                                    result_df.loc[i, 'R_IQR_AFP'])):
                                                err_rawdata = df_rawdata[(df_rawdata['code'] == code) & (
                                                        df_rawdata['AFP'] > result_df.loc[i, 'R_IQR_AFP'])]
                                            # иначе просто максимальное значение в ошибочном коде
                                            else:
                                                err_rawdata = df_rawdata[df_rawdata['code'] == code]

                                            if not err_rawdata.empty:
                                                error_flag = True
                                                iserror = True
                                                max_AFP = err_rawdata['AFP'].max()
                                                print(
                                                    f"{i} max, {datetime.now()}. Ошибка в {parent_code} {col}. {parent_stats[col]}  < {result_df.loc[i, col]}, удалится - {max_AFP}, порог у подчинённого {code} - {result_df.loc[i, 'R_IQR_AFP']}")
                                                first_max_AFP_index = err_rawdata[err_rawdata['AFP'] == max_AFP].index[
                                                    0]  # удаление по строке

                                                id_for_query = int(err_rawdata.loc[first_max_AFP_index, 'id'])
                                                self.db_changes.append(("update_close_ab", None, id_for_query))
                                                # query = "UPDATE anketa SET bonus_fact = NULL, comission_fact = NULL WHERE id = ?"
                                                # #print(type(id_for_query))
                                                # cursor = conn.cursor()
                                                # cursor.execute(query, (id_for_query,))
                                                # conn.commit()
                                                # self.number_of_closed_ab += 1
                                                df_rawdata.loc[first_max_AFP_index, ['bonus_fact', 'comission_fact',
                                                                                     'AFP']] = np.nan  # удаление по строке
                                                updated_stats = recalculate_stats_hier(
                                                    df_rawdata[df_rawdata['code'] == code],
                                                    hierarchy_df)
                                                index_to_replace = result_df[result_df['code'] == code].index[0]
                                                result_df.loc[index_to_replace] = updated_stats.loc[0]
                                                step_6_2_done = False
                                                error_codes = append_error(error_codes, parent_code, col,
                                                                           description="")
                                                break
                                            else:
                                                error_codes_after_fixing = append_error(error_codes_after_fixing,
                                                                                        parent_code,
                                                                                        col,
                                                                                        f"В {code} нет записей с AFP > R_IQR_AFP {result_df.loc[i, 'R_IQR_AFP']}")
                                    else:
                                        if (df_rawdata[df_rawdata['code'] == parent_code].shape[0] <= 3):
                                            error_codes_after_fixing = append_error(error_codes_after_fixing,
                                                                                    parent_code,
                                                                                    col,
                                                                                    f"В {parent_code} <=3 записей")
                                        else:
                                            error_codes_after_fixing = append_error(error_codes_after_fixing,
                                                                                    parent_code,
                                                                                    col,
                                                                                    f"В {code} <=3 записей")
                        if error_flag:
                            break  # Прерывает внешний цикл for при обнаружении ошибки

                return df_rawdata, result_df, error_codes, error_codes_after_fixing, iserror

            """
            Ошибка 2. AGP > AFP
            """

            def AGP_BiggerThan_AFP(df, result_df, error_codes, error_codes_after_fixing):
                df_rawdata = df.copy()
                df_stats = result_df.copy()
                iserror = False

                codes = df_stats['code'].unique()

                for code in codes:
                    agp_percentiles = ['AGP_10', 'AGP_25', 'AGP_50', 'AGP_75', 'AGP_90', 'AGP_avg']
                    afp_percentiles = ['AFP_10', 'AFP_25', 'AFP_50', 'AFP_75', 'AFP_90', 'AFP_avg']

                    for agp_percentile, afp_percentile in zip(agp_percentiles, afp_percentiles):
                        agp_value = df_stats.loc[df_stats['code'] == code, agp_percentile].values[0]
                        afp_value = df_stats.loc[df_stats['code'] == code, afp_percentile].values[0]

                        if not (pd.isna(agp_value) or pd.isna(afp_value)):

                            if agp_value > afp_value:

                                print(f"{code}: {agp_percentile} {agp_value} > {afp_percentile} {afp_value}")
                                """
                                если медиана находится в самом первом интервале, то, её нулевое значение равно минимальному значению в интервале ряда. И тогда условие алгоритма в пункт 1 и 3 не выполнимо
                                Тоже самое в пункте 2 с максимальным значением
                                И уточнить про условие в варианте исправления 2:

                                Из-за особенностей алгоритма может быть такое, что из-за исправлений в шагах 2-3 при повторном запуске алгоритма появятся новые возможности в шагах выше. UPD: или потому что новые интервалы рассчитываются
                                """
                                intervals_AFP, counts_AFP = interval_variation_series(
                                    df_rawdata[df_rawdata['code'] == code], 'AFP')
                                median_interval_AFP = median_interval(intervals_AFP, counts_AFP)
                                lower_bound_AFP = intervals_AFP[0]  # МИНИМАЛЬНОЕ ЗНАЧЕНИЕ В AFP
                                upper_bound_AFP = intervals_AFP[-1]  # МАКСИМАЛЬНОЕ ЗНАЧЕНИЕ В AFP
                                if lower_bound_AFP == median_interval_AFP[0]:
                                    median_bound_AFP = median_interval_AFP[1]  # ПОРОГ МЕДИАННОГО ИНТЕРВАЛА
                                else:
                                    median_bound_AFP = median_interval_AFP[0]

                                intervals_AGP, counts_AGP = interval_variation_series(
                                    df_rawdata[df_rawdata['code'] == code], 'AGP')
                                median_interval_AGP = median_interval(intervals_AGP, counts_AGP)
                                lower_bound_AGP = intervals_AGP[0]  # МИНИМАЛЬНОЕ ЗНАЧЕНИЕ В AGP
                                upper_bound_AGP = intervals_AGP[-1]  # МАКСИМАЛЬНОЕ ЗНАЧЕНИЕ В AGP
                                if lower_bound_AGP == median_interval_AGP[0]:
                                    median_bound_AGP = median_interval_AGP[1]  # ПОРОГ МЕДИАННОГО ИНТЕРВАЛА
                                else:
                                    median_bound_AGP = median_interval_AGP[0]

                                option_1, option_2, option_3 = False, False, False
                                option_2_rows_threeshold = len(
                                    df_rawdata[df_rawdata['code'] == code])  # первоначальное кол-во строк для условия 2

                                while not (
                                        option_1 and option_2 and option_3):  # пока срабатывает хотя бы одна из трёх проверок, т.е. хотя бы одна из них False

                                    agp_value = df_stats.loc[df_stats['code'] == code, agp_percentile].values[
                                        0]  # делаем проверку неравенства, если ошибка исправилась то выход из проверки
                                    afp_value = df_stats.loc[df_stats['code'] == code, afp_percentile].values[0]

                                    if (pd.isna(agp_value) or pd.isna(afp_value)) or agp_value < afp_value:
                                        break

                                    if option_1 is False:
                                        # Option 1
                                        filtered_rawdata = df_rawdata[
                                            df_rawdata['code'] == code]  # берём записи с ошибочным кодом
                                        total_rows = len(filtered_rawdata)  # всего записей по коду
                                        filtered_rawdata = filtered_rawdata[
                                            filtered_rawdata['AFP'].notna()]  # оставляем только записи с бонусом
                                        total_AFP = len(filtered_rawdata)  # кол-во записей с бонусом
                                        min_remaining_AFP = total_AFP * 100 / total_rows  # % записей с бонусом от общего кол-ва записей по коду
                                        filtered_rawdata = filtered_rawdata.groupby('client_id').filter(
                                            lambda group: len(
                                                group) > 1)  # оставляем только те записи с бонусами у кого компаний 2 или больше

                                        condition_1 = (filtered_rawdata['AFP'] > lower_bound_AFP) & (
                                                filtered_rawdata['AFP'] <= median_bound_AFP)
                                        condition_1_1 = (filtered_rawdata['AGP'] < lower_bound_AGP) | (
                                                filtered_rawdata['AGP'] >= median_bound_AGP)
                                        option_1_rows = filtered_rawdata[
                                            condition_1 & condition_1_1]  # условие по алгоритму

                                        if not option_1_rows.empty:  # если после всех фильтров что то осталось. Иначе option_1 больше не проверяем
                                            if min_remaining_AFP >= 50:  # если записи есть, но вычищено уже много. Иначе option_1 больше не проверяем
                                                row_to_correct = option_1_rows.iloc[0]
                                                index_to_correct = int(row_to_correct['id'])
                                                print(f"option 1. {code}, {row_to_correct['id']}")
                                                self.db_changes.append(("update_close_ab", None, index_to_correct))
                                                # query = "UPDATE anketa SET bonus_fact = NULL, comission_fact = NULL WHERE id = ?"
                                                # #print(type(index_to_correct))
                                                # cursor = conn.cursor()
                                                # cursor.execute(query, (index_to_correct,))
                                                # conn.commit()
                                                # self.number_of_closed_ab += 1
                                                df_rawdata.loc[
                                                    row_to_correct.name, ['bonus_fact', 'comission_fact',
                                                                          'AFP']] = np.nan
                                                updated_stats = recalculate_stats(
                                                    df_rawdata[df_rawdata['code'] == code])
                                                index_to_replace = df_stats[df_stats['code'] == code].index[0]
                                                df_stats.loc[index_to_replace] = updated_stats.loc[0]
                                                option_1 = False
                                                error_codes = append_error(error_codes, code, "AFP",
                                                                           f"Option 1. AGP>AFP")
                                                iserror = True
                                                continue  # проверяем заново
                                            else:
                                                option_1 = True

                                        else:
                                            option_1 = True

                                    if option_2 is False:
                                        # Option 2
                                        filtered_rawdata = df_rawdata[df_rawdata['code'] == code]
                                        total_rows = len(filtered_rawdata)
                                        current_perc = total_rows / option_2_rows_threeshold * 100
                                        filtered_rawdata = filtered_rawdata.groupby('client_id').filter(
                                            lambda group: len(
                                                group) > 1)  # оставляем только те записи, у кого компаний 2 или больше
                                        condition_2 = (filtered_rawdata['AGP'] > median_bound_AGP) & (
                                                filtered_rawdata['AGP'] <= upper_bound_AGP) & filtered_rawdata[
                                                          'AFP'].isna()
                                        option_2_rows = filtered_rawdata[condition_2]  # условие по алгоритму
                                        if not option_2_rows.empty:
                                            if current_perc >= 50:  # не менее 50% от изначальных строчек
                                                row_to_delete = option_2_rows.iloc[0]
                                                index_to_delete = int(row_to_delete['id'])
                                                print(f"option 2. {code}, {row_to_delete['id']}")
                                                self.db_changes.append(("delete", None, index_to_delete))
                                                # query = "DELETE FROM anketa WHERE id = ?"
                                                # #print(type(index_to_delete))
                                                # cursor = conn.cursor()
                                                # cursor.execute(query, (index_to_delete,))
                                                # conn.commit()
                                                # self.number_of_deleted_rows += 1
                                                df_rawdata.drop(row_to_delete.name, inplace=True)
                                                updated_stats = recalculate_stats(
                                                    df_rawdata[df_rawdata['code'] == code])
                                                index_to_replace = df_stats[df_stats['code'] == code].index[0]
                                                df_stats.loc[index_to_replace] = updated_stats.loc[0]
                                                option_2 = False
                                                error_codes = append_error(error_codes, code, "AGP",
                                                                           f"Option 2. AGP>AFP")
                                                iserror = True
                                                continue  # проверяем заново
                                            else:
                                                option_2 = True
                                        else:
                                            option_2 = True

                                    if option_3 is False:
                                        # Option 3
                                        filtered_rawdata = df_rawdata[
                                            df_rawdata['code'] == code]  # берём записи с ошибочным кодом
                                        total_rows = len(filtered_rawdata)  # всего записей по коду
                                        filtered_rawdata = filtered_rawdata[
                                            filtered_rawdata['AFP'].notna()]  # оставляем только записи с бонусом
                                        total_AFP = len(filtered_rawdata)  # кол-во записей с бонусом
                                        min_remaining_AFP = total_AFP * 100 / total_rows  # % записей с бонусом от общего кол-ва записей по коду
                                        filtered_rawdata = filtered_rawdata.groupby('client_id').filter(
                                            lambda group: len(
                                                group) > 1)  # оставляем только те записи с бонусами у кого компаний 2 или больше

                                        condition_3 = (filtered_rawdata['AFP'] > lower_bound_AFP) & (
                                                filtered_rawdata['AFP'] <= median_bound_AFP)
                                        option_3_rows = filtered_rawdata[condition_3]  # условие по алгоритму

                                        if not option_3_rows.empty:  # если после всех фильтров что то осталось. Иначе option_1 больше не проверяем
                                            if min_remaining_AFP >= 50:  # если записи есть, но вычищено уже много. Иначе option_1 больше не проверяем
                                                row_to_correct = option_3_rows.iloc[0]
                                                index_to_correct = int(row_to_correct['id'])
                                                print(f"option 3. {code}, {row_to_correct['id']}")
                                                self.db_changes.append(("update_close_ab", None, index_to_correct))
                                                # query = "UPDATE anketa SET bonus_fact = NULL, comission_fact = NULL WHERE id = ?"
                                                # #print(type(index_to_correct))
                                                # cursor = conn.cursor()
                                                # cursor.execute(query, (index_to_correct,))
                                                # conn.commit()
                                                # self.number_of_closed_ab += 1
                                                df_rawdata.loc[
                                                    row_to_correct.name, ['bonus_fact', 'comission_fact',
                                                                          'AFP']] = np.nan
                                                updated_stats = recalculate_stats(
                                                    df_rawdata[df_rawdata['code'] == code])
                                                index_to_replace = df_stats[df_stats['code'] == code].index[0]
                                                df_stats.loc[index_to_replace] = updated_stats.loc[0]
                                                option_3 = False
                                                error_codes = append_error(error_codes, code, "AFP",
                                                                           f"Option 3. AGP>AFP")
                                                iserror = True
                                                continue  # проверяем заново
                                            else:
                                                option_3 = True
                                                error_codes_after_fixing = append_error(error_codes_after_fixing, code,
                                                                                        "full", f"AGP>AFP")
                                        else:
                                            option_3 = True
                                            error_codes_after_fixing = append_error(error_codes_after_fixing, code,
                                                                                    "full",
                                                                                    f"AGP>AFP")

                return df_rawdata, df_stats, error_codes, error_codes_after_fixing, iserror

            """
            Ошибка 3. Одинаковые соседние перцентили
            """

            def same_percentiles(df, result_df, error_codes, error_codes_after_fixing):
                df_rawdata = df.copy()
                df_stats = result_df.copy()
                iserror = False

                codes = df_stats['code'].unique()

                for code in codes:

                    percentiles = {'AGP_10': 'AGP_25', 'AGP_25': 'AGP_50', 'AGP_50': 'AGP_75', 'AGP_75': 'AGP_90',
                                   'AFP_10': 'AFP_25', 'AFP_25': 'AFP_50', 'AFP_50': 'AFP_75', 'AFP_75': 'AFP_90'}

                    for key in percentiles.items():
                        left_value = df_stats.loc[df_stats['code'] == code, key[0]].values[0]
                        right_value = df_stats.loc[df_stats['code'] == code, key[1]].values[0]

                        if not (pd.isna(left_value) or pd.isna(right_value)):

                            if left_value == right_value:

                                print(f"{code}: {key[0]} {left_value} = {right_value} {key[1]}")

                                option = False

                                while not option:  # пока срабатывает

                                    left_value = df_stats.loc[df_stats['code'] == code, key[0]].values[
                                        0]  # делаем проверку равенства
                                    right_value = df_stats.loc[df_stats['code'] == code, key[1]].values[
                                        0]  # делаем проверку равенства

                                    if (pd.isna(left_value) or pd.isna(right_value)) or left_value != right_value:
                                        break

                                    if option is False:
                                        filtered_rawdata = df_rawdata[(df_rawdata['code'] == code) & (df_rawdata[key[0][
                                                                                                                 :3]] == left_value)]  # берём записи с ошибочным кодом и те записи у которых в ошибочном значении пок-ль равен значению перцентиля

                                        if not filtered_rawdata.empty:  # если после всех фильтров что то осталось. Иначе ошибку не исправить
                                            filtered_rawdata = filtered_rawdata.groupby('client_id').filter(
                                                lambda group: len(
                                                    group) > 1)  # оставляем только те записи у кого компаний 2 или больше

                                            if filtered_rawdata.empty:
                                                filtered_rawdata = df_rawdata[
                                                    (df_rawdata['code'] == code) & (df_rawdata[
                                                                                        key[0][
                                                                                        :3]] == left_value)]  # если таких не осталось, то, удаляем последнюю запись от компании

                                            most_frequent_company = filtered_rawdata[
                                                'client_id'].value_counts().idxmax()
                                            row_to_correct = \
                                                filtered_rawdata[
                                                    filtered_rawdata['client_id'] == most_frequent_company].iloc[0]
                                            index_to_correct = int(row_to_correct.name)

                                            if key[0][:3] == "AGP":
                                                print(f"Delete row. {code}, {row_to_correct['id']}")
                                                self.db_changes.append(("delete", None, int(row_to_correct['id'])))
                                                # query = """DELETE FROM ANKETA WHERE id = ?"""
                                                # #print(type(index_to_correct))
                                                # cursor = conn.cursor()
                                                # cursor.execute(query, (int(row_to_correct['id']),))
                                                # conn.commit()
                                                # self.number_of_deleted_rows += 1
                                                df_rawdata.drop(index_to_correct, inplace=True)
                                            else:
                                                if row_to_correct['bonus_eligibility'] == "да":
                                                    print(f"Delete AFP. {code}, {row_to_correct['id']}")
                                                    self.db_changes.append(
                                                        ("update_close_ab", None, int(row_to_correct['id'])))
                                                    # query = "UPDATE anketa SET bonus_fact = NULL, comission_fact = NULL WHERE id = ?"
                                                    # #print(type(index_to_correct))
                                                    # cursor = conn.cursor()
                                                    # cursor.execute(query, (int(row_to_correct['id']),))
                                                    # conn.commit()
                                                    # self.number_of_closed_ab += 1
                                                    df_rawdata.loc[
                                                        index_to_correct, ['bonus_fact', 'comission_fact',
                                                                           'AFP']] = np.nan
                                                else:
                                                    print(f"Change no to yes. {code}, {row_to_correct['id']}")
                                                    self.db_changes.append(
                                                        ("update_change_yes", None, int(row_to_correct['id'])))
                                                    # query = """UPDATE anketa SET bonus_eligibility = 'да' WHERE id = ?"""
                                                    # #print(type(index_to_correct))
                                                    # cursor = conn.cursor()
                                                    # cursor.execute(query, (int(row_to_correct['id']),))
                                                    # conn.commit()
                                                    df_rawdata.at[index_to_correct, 'bonus_eligibility'] = "да"

                                            updated_stats = recalculate_stats(df_rawdata[df_rawdata['code'] == code])
                                            index_to_replace = df_stats[df_stats['code'] == code].index[0]
                                            df_stats.loc[index_to_replace] = updated_stats.loc[0]
                                            option = False
                                            error_codes = append_error(error_codes, code, key[0],
                                                                       f"{code}: {key[0]} {left_value} = {right_value} {key[1]}")
                                            iserror = True
                                            continue  # проверяем заново

                                        else:
                                            error_codes_after_fixing = append_error(error_codes_after_fixing, code,
                                                                                    key[0],
                                                                                    f"{code}: {key[0]} {left_value} = {right_value} {key[1]}")
                                            option = True

                return df_rawdata, df_stats, error_codes, error_codes_after_fixing, iserror

            def correct_errors(df_1, result_df_1):
                df = df_1.copy()
                result_df = result_df_1.copy()
                codes_with_errors_hierarchy = {}
                error_codes_after_fixing_hierarchy = {}
                codes_with_errors_AGP_BiggerThan_AFP = {}
                error_codes_after_fixing_AGP_BiggerThan_AFP = {}
                codes_with_errors_same_percentiles = {}
                error_codes_after_fixing_same_percentiles = {}
                while True:
                    # Step 1
                    print("START STEP 1")
                    self.progress.emit("Шаг 1. Проверяем и исправляем иерархию")
                    df, result_df, iserror_hierarchy_new = new_hierarchy_algoritm(df, result_df, hierarchy_df)
                    if not self._is_running:
                        break
                    # Step 2
                    print("START STEP 2")
                    self.progress.emit("Шаг 2. Проверяем и исправляем иерархию")
                    df, result_df, codes_with_errors_hierarchy, error_codes_after_fixing_hierarchy, iserror_hierarchy = hierarchy_errors(
                        df, result_df, hierarchy_df, codes_with_errors_hierarchy, error_codes_after_fixing_hierarchy)
                    if not self._is_running:
                        break
                    # Step 3
                    print("START STEP 3")
                    self.progress.emit("Шаг 3. Проверяем и исправляем Total<Guaranteed")
                    df, result_df, codes_with_errors_AGP_BiggerThan_AFP, error_codes_after_fixing_AGP_BiggerThan_AFP, iserror_AGP_BiggerThan_AFP = AGP_BiggerThan_AFP(
                        df, result_df, codes_with_errors_AGP_BiggerThan_AFP,
                        error_codes_after_fixing_AGP_BiggerThan_AFP)
                    if iserror_AGP_BiggerThan_AFP:
                        continue
                    if not self._is_running:
                        break
                    # Step 4
                    print("START STEP 4")
                    self.progress.emit("Шаг 4. Проверяем и исправляем одинаковые перцентили")
                    df, result_df, codes_with_errors_same_percentiles, error_codes_after_fixing_same_percentiles, iserror_same_percentiles = same_percentiles(
                        df, result_df, codes_with_errors_same_percentiles, error_codes_after_fixing_same_percentiles)
                    if iserror_same_percentiles:
                        continue
                    if not self._is_running:
                        break
                    if not iserror_same_percentiles and not iserror_AGP_BiggerThan_AFP and not iserror_hierarchy:
                        break

                return error_codes_after_fixing_hierarchy, error_codes_after_fixing_AGP_BiggerThan_AFP, error_codes_after_fixing_same_percentiles  # df, result_df,

            error_codes_after_fixing_hierarchy, error_codes_after_fixing_AGP_BiggerThan_AFP, error_codes_after_fixing_same_percentiles = correct_errors(
                df, result_df)  # df_clean, result_df_clean,

            self.progress.emit("Применяем изменения...")
            self.df_db_changes = pd.DataFrame(self.db_changes, columns=['action', 'code', 'id'])
            actual_counts = self.df_db_changes['action'].value_counts().to_dict()
            self.action_counts.update(actual_counts)

            # Assuming df is your DataFrame
            if self.df_db_changes.empty:
                self.info.emit("Записей для исправления не найдено")
                self._is_running = False
                return
            else:
                cursor = conn.cursor()
                for index, row in self.df_db_changes.iterrows():
                    if row['action'] == 'update_code':
                        query = "UPDATE anketa SET code = ? WHERE id = ?"
                        params = (row['code'], row['id'])
                    elif row['action'] == 'delete':
                        query = "DELETE FROM anketa WHERE id = ?"
                        params = (row['id'],)
                    elif row['action'] == 'update_close_ab':
                        query = "UPDATE anketa SET bonus_fact = NULL, comission_fact = NULL WHERE id = ?"
                        params = (row['id'],)
                    elif row['action'] == 'update_change_yes':
                        query = "UPDATE anketa SET bonus_eligibility = 'yes' WHERE id = ?"
                        params = (row['id'],)
                    else:
                        print("else")
                        continue

                    cursor.execute(query, params)
                    conn.commit()

            """
            обновляем интерфейс
            """
            self.progress.emit("Обновляем интерфейс...")
            if not self._is_running:
                return

            df_agg_pos = pd.read_sql_query(f"""select * from report_view where id in (select id from anketa where is_rematch = 1 
                        and client_id in ({self.companies_names_placeholders}) and location_id in ({self.location_ids_placeholders}))""",
                                           conn, params=self.companies_names + self.location_ids)

            temp_grouped = df_agg_pos.groupby('code').agg({'client_id': pd.Series.nunique})
            solv_codes = temp_grouped[temp_grouped['client_id'] > 2].index
            df = df_agg_pos[df_agg_pos['code'].isin(solv_codes)]

            self.total_rows_after = df_agg_pos.shape[0]
            self.total_rows_solv_after = df.shape[0]

            index = df.code.unique().tolist()
            agg_index = df_agg_pos.code.unique().tolist()
            agg_codes_placeholders = ','.join(['?' for _ in agg_index])
            code_info = pd.read_sql_query(f"""select code, function, division, subfunction, name, level as level_num
                                          from catalogue where code in ({agg_codes_placeholders})""", conn,
                                          params=agg_index)

            code_info_agg_pos = code_info[['code', 'level_num']]
            df_agg_pos = pd.merge(df_agg_pos, code_info_agg_pos, on='code', how='left')

            def level_to_suffix(level):
                if level in [2, 3]:
                    return '_TM'
                elif level in [4, 5]:
                    return '_MM'
                elif level in [6, 7, 8]:
                    return '_SP'
                else:
                    return None

            def modify_code_based_on_prefix_and_level(row):
                base_code = row['code']
                suffix = level_to_suffix(row['level_num'])

                # Если код начинается с 'AGR' или 'PHR', то берем первые два значения после split('_')
                if base_code.startswith(('AGR', 'PHR')):
                    prefix = '_'.join(base_code.split('_')[:2])
                else:
                    prefix = base_code.split('_')[0]
                # Добавляем суффикс, если он определен
                if suffix is not None:
                    return prefix + suffix
                else:
                    return None

            df_agg_pos['code'] = df_agg_pos.apply(modify_code_based_on_prefix_and_level, axis=1)

            # removing the rows where level_num is 1, 9, or 10
            df_agg_pos = df_agg_pos[~df_agg_pos['level_num'].isin([1, 9, 10])]
            # removing unsolved agg positions
            temp_grouped = df_agg_pos.groupby('code').agg({'client_id': pd.Series.nunique})
            solv_codes = temp_grouped[temp_grouped['client_id'] > 2].index
            df_agg_pos = df_agg_pos[df_agg_pos['code'].isin(solv_codes)]

            if not self._is_running:
                return

            # считаем количество раскрытых кодов
            self.solved_codes_list = index
            self.cnt_solved_codes = len(self.solved_codes_list)

            # считаем количество всех кодов
            allcodes_df = pd.read_sql_query(f"""SELECT COUNT (DISTINCT code) as cnts FROM anketa where is_rematch = 1 and client_id 
            in ({self.companies_names_placeholders}) and location_id in ({self.location_ids_placeholders}) """,
                                            conn, params=self.companies_names + self.location_ids)
            self.cnt_codes = allcodes_df['cnts'][0]

            # считаем количество НЕраскрытых кодов
            unsolved_codes_df = pd.read_sql_query(f"""select DISTINCT code from anketa where is_rematch = 1 
            and client_id in ({self.companies_names_placeholders}) and location_id in ({self.location_ids_placeholders}) 
            and code NOT IN (SELECT code from anketa WHERE location_id in ({self.location_ids_placeholders}) and client_id in 
            ({self.companies_names_placeholders}) and is_rematch = 1 group by code  having (COUNT(DISTINCT client_id))>2)""",
                                                  conn,
                                                  params=self.companies_names + self.location_ids + self.location_ids + self.companies_names)
            self.unsolved_codes = unsolved_codes_df['code'].tolist()
            self.unsolved_codes.sort()
            self.cnt_unsolved_codes = len(self.unsolved_codes)

            def process_group(group_data):
                (job_id, client_id), group = group_data
                if len(group) > 5:
                    percentiles = [10, 25, 50, 75, 90]
                    indicators = ['BP', 'AGP', "AB", "AB_perc", 'bonus_deff', 'lti', "TB_perc", "AFP", "ATP",
                                  "total_income"]
                    percentiles_dict = {}

                    for indicator in indicators:
                        if group[indicator].count() > 5:
                            percentiles_dict[indicator] = nanpercentile(group[indicator], percentiles)
                        else:
                            non_nan_values = group[indicator].dropna().reset_index(drop=True)
                            missing_nans = 5 - len(non_nan_values)
                            percentiles_dict[indicator] = non_nan_values.append(
                                pd.Series([nan] * missing_nans)).reset_index(drop=True)

                    group_df = pd.DataFrame({
                        "code": [job_id] * 5,
                        "client_id": [client_id] * 5,
                        "id": [1, 2, 3, 4, 5],
                        "BP": percentiles_dict["BP"],
                        "AGP": percentiles_dict["AGP"],
                        "AB": percentiles_dict["AB"],
                        "AB_perc": percentiles_dict["AB_perc"],
                        "bonus_deff": percentiles_dict["bonus_deff"],
                        "lti": percentiles_dict["lti"],
                        "TB_perc": percentiles_dict["TB_perc"],
                        "AFP": percentiles_dict["AFP"],
                        "ATP": percentiles_dict["ATP"],
                        "total_income": percentiles_dict["total_income"],
                    })
                else:
                    group_df = group
                return group_df

            def trimming(df):
                grouped = df.groupby(["code", "client_id"])

                with ThreadPoolExecutor() as executor:
                    # Create a partial function with fixed df_columns argument
                    process_group_partial = partial(process_group)

                    # Execute process_group_partial in parallel for each group in grouped
                    results = executor.map(process_group_partial, grouped)

                # Concatenate the results and reset the index
                new_df = pd.concat(results).reset_index(drop=True)
                return new_df

            if not self._is_running:
                return
            try:
                trimmed_df = trimming(df)
            except:
                trimmed_df = df

            try:
                trimmed_df_agg_pos = trimming(df_agg_pos)
            except:
                trimmed_df_agg_pos = df_agg_pos

            index_agg_pos = trimmed_df_agg_pos.code.unique().tolist()
            columns = ['BP_10', 'BP_25', 'BP_50', 'BP_75', 'BP_90', 'BP_avg',
                       'AGP_10', 'AGP_25', 'AGP_50', 'AGP_75', 'AGP_90', 'AGP_avg',
                       'AB_10', 'AB_25', 'AB_50', 'AB_75', 'AB_90', 'AB_avg',
                       'AB_perc_10', 'AB_perc_25', 'AB_perc_50', 'AB_perc_75', 'AB_perc_90', 'AB_perc_avg',
                       'bonus_deff_10', 'bonus_deff_25', 'bonus_deff_50', 'bonus_deff_75', 'bonus_deff_90',
                       'bonus_deff_avg',
                       'lti_10', 'lti_25', 'lti_50', 'lti_75', 'lti_90', 'lti_avg',
                       'TB_perc_10', 'TB_perc_25', 'TB_perc_50', 'TB_perc_75', 'TB_perc_90', 'TB_perc_avg',
                       'AFP_10', 'AFP_25', 'AFP_50', 'AFP_75', 'AFP_90', 'AFP_avg',
                       'ATP_10', 'ATP_25', 'ATP_50', 'ATP_75', 'ATP_90', 'ATP_avg',
                       'total_income_10', 'total_income_25', 'total_income_50', 'total_income_75', 'total_income_90',
                       'total_income_avg']

            trimmed_df = trimmed_df[['code', 'client_id', 'BP', 'AGP', 'AB', 'AB_perc',
                                     'bonus_deff', 'lti', 'TB_perc', 'AFP', 'ATP', 'total_income']]

            trimmed_df_agg_pos = trimmed_df_agg_pos[['code', 'client_id', 'BP', 'AGP', 'AB', 'AB_perc',
                                                     'bonus_deff', 'lti', 'TB_perc', 'AFP', 'ATP', 'total_income']]

            def fill_cells_optimized(trimmed_df, columns, index):
                stats = pd.DataFrame(columns=columns, index=index)
                grouped_df = trimmed_df.groupby('code')
                for code, group in grouped_df:
                    for attribute in trimmed_df.columns[2:]:
                        num = group[attribute].count()
                        cnt_clients = group.dropna(subset=[attribute])['client_id'].nunique()
                        if cnt_clients > 2:
                            stats.loc[code, f'{attribute}_avg'] = group[attribute].mean()
                            if 4 <= num <= 5:
                                stats.loc[code, f'{attribute}_50'] = group[attribute].quantile(q=0.5)
                            elif 6 <= num <= 7:
                                stats.loc[code, f'{attribute}_25'] = group[attribute].quantile(q=0.25)
                                stats.loc[code, f'{attribute}_50'] = group[attribute].quantile(q=0.5)
                                stats.loc[code, f'{attribute}_75'] = group[attribute].quantile(q=0.75)
                            elif num >= 8:
                                stats.loc[code, f'{attribute}_10'] = group[attribute].quantile(q=0.1)
                                stats.loc[code, f'{attribute}_25'] = group[attribute].quantile(q=0.25)
                                stats.loc[code, f'{attribute}_50'] = group[attribute].quantile(q=0.5)
                                stats.loc[code, f'{attribute}_75'] = group[attribute].quantile(q=0.75)
                                stats.loc[code, f'{attribute}_90'] = group[attribute].quantile(q=0.9)
                return stats

            if not self._is_running:
                return

            stats = fill_cells_optimized(trimmed_df, columns, index)
            stats_agg_pos = fill_cells_optimized(trimmed_df_agg_pos, columns, index_agg_pos)

            temp_stats_agg_pos = stats_agg_pos.reset_index()
            temp_stats_agg_pos = temp_stats_agg_pos.rename(columns={"index": "code"})

            def create_prefix(row):
                base_code = row['code']

                # Если код начинается с 'AGR' или 'PHR', то берем первые два значения после split('_')
                if base_code.startswith(('AGR', 'PHR')):
                    return '_'.join(base_code.split('_')[:2])
                else:
                    return base_code.split('_')[0]

            def count_errors_in_dict(error_dict):
                key_count = 0
                # Итерация по основному словарю
                for key in error_dict:
                    # Если значение для данного ключа является словарем
                    if isinstance(error_dict[key], dict):
                        # Увеличиваем счетчик ключей на количество ключей в подсловаре
                        key_count += len(error_dict[key])
                return key_count

            def create_agg_hierarchy(stats_agg_pos):
                stats_agg_pos['Prefix'] = stats_agg_pos.apply(create_prefix, axis=1)
                stats_agg_pos['Suffix'] = stats_agg_pos['code'].str.split('_').apply(lambda x: x[-1])
                suffix_dict = {'TM': 0, 'MM': 1, 'SP': 2}
                stats_agg_pos['Seniority'] = stats_agg_pos['Suffix'].map(suffix_dict)
                stats_agg_pos = stats_agg_pos.sort_values(['Prefix', 'Seniority'])
                stats_agg_pos['parent'] = stats_agg_pos.groupby('Prefix')['code'].shift()
                stats_agg_pos.loc[stats_agg_pos['Seniority'] == 0, 'parent'] = nan
                hierarchy_table = stats_agg_pos[['code', 'parent']]

                return hierarchy_table

            agg_hierarchy = create_agg_hierarchy(temp_stats_agg_pos[['code']])

            def count_agg_errors(stats_agg_pos, agg_hierarchy):
                temp_stats_agg_pos = pd.merge(stats_agg_pos, agg_hierarchy, on="code", how="left")

                e_1 = count_hierarchy_errors(temp_stats_agg_pos)
                e_2 = count_AGP_BiggerThan_AFP(temp_stats_agg_pos)
                e_3 = count_same_percentiles(temp_stats_agg_pos)
                e_4 = count_hierarchy_errors_e4(temp_stats_agg_pos)
                e_5 = count_avg_between_25_75(temp_stats_agg_pos)
                e_6 = count_bonus_total(temp_stats_agg_pos)

                print(f"{e_1}\n,{e_2}\n,{e_3}\n,{e_4}\n,{e_5}\n,{e_6}\n")

                agg_cnt_e_1 = count_errors_in_dict(e_1)
                agg_cnt_e_2 = count_errors_in_dict(e_2)
                agg_cnt_e_3 = count_errors_in_dict(e_3)
                agg_cnt_e_4 = count_errors_in_dict(e_4)
                agg_cnt_e_5 = count_errors_in_dict(e_5)
                agg_cnt_e_6 = count_errors_in_dict(e_6)

                errors_list = [agg_cnt_e_1, agg_cnt_e_2, agg_cnt_e_3, agg_cnt_e_4, agg_cnt_e_5, agg_cnt_e_6]

                return errors_list

            agg_errors_list = count_agg_errors(temp_stats_agg_pos, agg_hierarchy)

            """
            кусок для подсчёта ошибок
            """

            if not self._is_running:
                return

            def find_closest_parent(code):
                parent = None
                valid_parent = False

                while True:
                    # Check if the code is present in the hierarchy dataframe
                    if code in hierarchy_df['code'].values:
                        # Get parent from the hierarchy dataframe
                        parent = hierarchy_df.loc[hierarchy_df['code'] == code, 'parent'].values[0]
                    else:
                        break

                    if parent is None or pd.isna(parent):
                        break

                    if parent in result_df['code'].values:
                        valid_parent = True
                        break

                    code = parent

                return parent if valid_parent else None

            result_df = stats.reset_index().rename(columns={'index': 'code'})
            hierarchy_df = pd.read_sql_query("SELECT * FROM hierarchy", conn)
            result_df['parent'] = result_df['code'].apply(find_closest_parent)

            e_1 = count_hierarchy_errors(result_df)
            e_2 = count_AGP_BiggerThan_AFP(result_df)
            e_3 = count_same_percentiles(result_df)
            e_4 = count_hierarchy_errors_e4(result_df)
            e_5 = count_avg_between_25_75(result_df)
            e_6 = count_bonus_total(result_df)

            cnt_e_1 = count_errors_in_dict(e_1)
            cnt_e_2 = count_errors_in_dict(e_2)
            cnt_e_3 = count_errors_in_dict(e_3)
            cnt_e_4 = count_errors_in_dict(e_4)
            cnt_e_5 = count_errors_in_dict(e_5)
            cnt_e_6 = count_errors_in_dict(e_6)

            errors_list = [cnt_e_1, cnt_e_2, cnt_e_3, cnt_e_4, cnt_e_5, cnt_e_6]

            # ошибки сводных должностей
            errors_list = errors_list + agg_errors_list
            """
            """

            def format_for_output(df, format_index):
                for column in df.columns:
                    for row_index in format_index:
                        if isnan(df.loc[row_index, column]) == True:
                            pass
                        else:
                            if column.find('perc') != -1 and row_index in format_index:
                                df.loc[row_index, column] = str(round(float(df.loc[row_index, column]), 2)) + "%"
                            elif row_index in format_index:
                                df.loc[row_index, column] = "{: ,}".format(int(df.loc[row_index, column])).replace(',',
                                                                                                                   ' ')
                return df

            if not self._is_running:
                return
            stats = format_for_output(stats, index)
            stats_agg_pos = format_for_output(stats_agg_pos, index_agg_pos)

            conn.close()

            stats = stats.reset_index()
            stats = stats.rename(columns={"index": "code"})

            stats_agg_pos = stats_agg_pos.reset_index()
            stats_agg_pos = stats_agg_pos.rename(columns={"index": "code"})

            stats = pd.merge(stats, code_info, on='code', how='left')
            stats = stats.sort_values(by=['function', 'division', 'subfunction', 'level_num'], ignore_index=True)

            # Create an empty DataFrame to hold the final result
            df_final = pd.DataFrame()
            # переносим 1 уровень сразу, если он есть
            general_manager = stats['level_num'] == 1
            df_final = df_final.append(stats[general_manager], ignore_index=True)
            # и удаляем
            stats = stats[~general_manager]

            stats = stats[["code",
                           'BP_10', 'BP_25', 'BP_50', 'BP_75', 'BP_90', 'BP_avg',
                           'AGP_10', 'AGP_25', 'AGP_50', 'AGP_75', 'AGP_90', 'AGP_avg',
                           'AB_10', 'AB_25', 'AB_50', 'AB_75', 'AB_90', 'AB_avg',
                           'AB_perc_10', 'AB_perc_25', 'AB_perc_50', 'AB_perc_75', 'AB_perc_90', 'AB_perc_avg',
                           'bonus_deff_10', 'bonus_deff_25', 'bonus_deff_50', 'bonus_deff_75', 'bonus_deff_90',
                           'bonus_deff_avg',
                           'lti_10', 'lti_25', 'lti_50', 'lti_75', 'lti_90', 'lti_avg',
                           'TB_perc_10', 'TB_perc_25', 'TB_perc_50', 'TB_perc_75', 'TB_perc_90', 'TB_perc_avg',
                           'AFP_10', 'AFP_25', 'AFP_50', 'AFP_75', 'AFP_90', 'AFP_avg',
                           'ATP_10', 'ATP_25', 'ATP_50', 'ATP_75', 'ATP_90', 'ATP_avg',
                           'total_income_10', 'total_income_25', 'total_income_50', 'total_income_75',
                           'total_income_90', 'total_income_avg', "level_num"]]

            def suffix_to_level(suffix):
                if suffix == 'TM':
                    return [2, 3]
                elif suffix == 'MM':
                    return [4, 5]
                elif suffix == 'SP':
                    return [6, 7, 8]
                else:
                    return None

            # Добавляем вспомогательную колонку в 'stats_agg_pos', чтобы сохранить соответствующий 'level_num'
            stats_agg_pos['level_num'] = stats_agg_pos['code'].apply(
                lambda x: suffix_to_level(x.split('_')[-1]))

            if not stats.empty:
                stats['prefix'] = stats.apply(create_prefix, axis=1)
            if not stats_agg_pos.empty:
                stats_agg_pos['prefix'] = stats_agg_pos.apply(create_prefix, axis=1)
            stats_agg_pos['suffix'] = stats_agg_pos['code'].str.split('_').apply(lambda x: x[-1])

            """
            Перед конкатенацией добавляем все колонки
            """

            stats = pd.merge(stats, code_info[['code', 'function', 'division', 'subfunction', 'name']],
                             on='code',
                             how='left')

            code_info = code_info[['code', 'function', 'level_num']]
            # changing the code column according to the level_num
            code_info['code'] = code_info.apply(modify_code_based_on_prefix_and_level, axis=1)
            code_info = code_info[['code', 'function']]
            code_info = code_info.drop_duplicates()
            stats_agg_pos = pd.merge(stats_agg_pos, code_info[['code', 'function']], on='code', how='left')

            """
            Сортировка в нужной последовательности: сначала TM, MM, SP
            """
            # Define custom order
            custom_dict = {'TM': 0, 'MM': 1, 'SP': 2}
            stats_agg_pos['suffix'] = stats_agg_pos['suffix'].map(custom_dict)

            # Sort DataFrame based on 'Prefix' and 'Suffix'
            stats_agg_pos.sort_values(by=['function', 'suffix'], inplace=True)

            # Dropping temporary 'Prefix' and 'Suffix' columns
            stats_agg_pos.drop(['suffix'], axis=1, inplace=True)

            stats_agg_pos[['division', 'subfunction']] = 'Сводная позиция'

            level_to_name = {2: 'Топ-менеджмент', 3: 'Топ-менеджмент', 4: 'Мидл-менеджмент', 5: 'Мидл-менеджмент',
                             6: 'Специалисты', 7: 'Специалисты', 8: 'Специалисты'}

            # Define a function to apply the mapping
            def map_level_to_name(row):
                if pd.isnull(row['name']):
                    for level in row['level_num']:
                        if level in level_to_name:
                            return level_to_name[level]
                else:
                    return row['name']

            # Apply the function to the 'name' column
            stats_agg_pos['name'] = nan
            if not stats_agg_pos.empty:
                stats_agg_pos['name'] = stats_agg_pos.apply(map_level_to_name, axis=1)

            def clean_and_return(df, condition_column, condition_value):
                condtition_df = df[df[condition_column] == condition_value]
                df = df[df[condition_column] != condition_value]
                return condtition_df, df

            general_management_stats, stats = clean_and_return(stats, 'function', 'Общее руководство компании')
            general_management_stats_agg_pos, stats_agg_pos = clean_and_return(stats_agg_pos, 'function',
                                                                               'Общее руководство компании')

            def concat_final_pivot_stats(stats, stats_agg_pos, df_final):
                for _, row_agg in stats_agg_pos.iterrows():
                    # Create a mask to select the rows in 'stats' that match the current 'prefix' and 'level_num'
                    mask = (stats['prefix'] == row_agg['prefix']) & (
                        stats['level_num'].isin(row_agg['level_num']))

                    # Append the rows from 'stats' that match the current 'prefix' and 'level_num' to 'df_final'
                    df_final = df_final.append(row_agg, ignore_index=True)
                    df_final = df_final.append(stats[mask], ignore_index=True)

                    # Drop the appended rows from 'stats'
                    stats = stats[~mask]

                    if stats[(stats['level_num'] <= 8) & (stats['prefix'] == row_agg['prefix'])].empty:
                        mask = (stats['prefix'] == row_agg['prefix']) & (stats['level_num'].isin([9, 10]))
                        df_final = df_final.append(stats[mask], ignore_index=True)

                        # Drop the appended rows from 'stats'
                        stats = stats[~mask]

                if not stats.empty:
                    stats = stats.sort_values(by=['function', 'division', 'subfunction', 'level_num'],
                                              ignore_index=True)
                    df_final = df_final.append(stats, ignore_index=True)

                return df_final

            if not general_management_stats.empty and not general_management_stats_agg_pos.empty:
                df_final = concat_final_pivot_stats(general_management_stats, general_management_stats_agg_pos,
                                                    df_final)
            else:
                if general_management_stats.empty:
                    df_final = general_management_stats_agg_pos
                else:
                    df_final = general_management_stats

            if not stats.empty and not stats_agg_pos.empty:
                df_final = concat_final_pivot_stats(stats, stats_agg_pos,
                                                    df_final)
            else:
                if stats.empty:
                    df_final = stats_agg_pos
                else:
                    df_final = stats

            df_final = df_final.drop(columns=['prefix'], axis=1)

            df_final = df_final[["code", "function", "division", "subfunction", "name", "level_num",
                                 'BP_10', 'BP_25', 'BP_50', 'BP_75', 'BP_90', 'BP_avg',
                                 'AGP_10', 'AGP_25', 'AGP_50', 'AGP_75', 'AGP_90', 'AGP_avg',
                                 'AB_10', 'AB_25', 'AB_50', 'AB_75', 'AB_90', 'AB_avg',
                                 'AB_perc_10', 'AB_perc_25', 'AB_perc_50', 'AB_perc_75', 'AB_perc_90',
                                 'AB_perc_avg',
                                 'TB_perc_10', 'TB_perc_25', 'TB_perc_50', 'TB_perc_75', 'TB_perc_90',
                                 'TB_perc_avg',
                                 'bonus_deff_10', 'bonus_deff_25', 'bonus_deff_50', 'bonus_deff_75',
                                 'bonus_deff_90', 'bonus_deff_avg',
                                 'lti_10', 'lti_25', 'lti_50', 'lti_75', 'lti_90', 'lti_avg',
                                 'AFP_10', 'AFP_25', 'AFP_50', 'AFP_75', 'AFP_90', 'AFP_avg',
                                 'ATP_10', 'ATP_25', 'ATP_50', 'ATP_75', 'ATP_90', 'ATP_avg',
                                 'total_income_10', 'total_income_25', 'total_income_50', 'total_income_75',
                                 'total_income_90', 'total_income_avg']]

            df_final = df_final.fillna("")

            # отправляем в GUI
            df_final['level_num'] = df_final['level_num'].apply(
                lambda x: '-'.join(map(str, x)) if isinstance(x, list) else x)

            df_final = df_final.rename(columns={"code": "Код\n должности\n RE", "function": "Функция I уровня",
                                                "division": "Функция II уровня", "subfunction": "Функция III уровня",
                                                "name": "Название должности", "level_num": "Уровень\n должности"})

            self.info.emit(
                f"Алгоритм завершил работу!\n\nЗаписей на начало работы: {self.total_rows}\nВ раскрытых кодах: {self.total_rows_solv}\n\nПодвержено изменениям: {sum(self.action_counts.values())}\nИз них:\n  Удалено: {self.action_counts['delete']}\n  Перемэтчено: {self.action_counts['update_code']}\n  Закрыто бонусов: {self.action_counts['update_close_ab']}\n  Раскрыто бонусов: {self.action_counts['update_change_yes']}\n\nПосле работы: {self.total_rows_after}\nВ раскрытых кодах: {self.total_rows_solv_after}")
            self.alogritm_finished.emit(error_codes_after_fixing_hierarchy, error_codes_after_fixing_AGP_BiggerThan_AFP,
                                        error_codes_after_fixing_same_percentiles, df_final, errors_list)
            for obj in gc.get_objects():
                if isinstance(obj, (pd.Series, pd.DataFrame, ndarray)):
                    del obj

            gc.collect()
            return
