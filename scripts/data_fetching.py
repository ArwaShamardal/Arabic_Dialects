import sqlite3
import pandas as pd


class DBHandler:
    def __init__(self, database_path: str) -> None:
        """
        Description: DBHandler class constructor, opens database connection
        ----------------
        Parameters:
        database_path : str : path to the database file
        ----------------
        Returns:
        None
        """
        self.database_path = database_path
        self.conn = None
        self.cursor = None
        self.open_connection()

    def open_connection(self) -> None:
        """
        Description: Open a connection to the database with the provided path using sqlite3 library
        ----------------
        Parameters:
        None
        ----------------
        Returns:
        None
        """
        try:
            self.conn = sqlite3.connect(self.database_path)
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")

    def close_connection(self) -> None:
        """
        Description: Close the connection to the database
        ----------------
        Parameters:
        None
        ----------------
        Returns:
        None
        """
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def list_tables(self) -> list:
        """
        Description: Function to list all the tables in the database
        ----------------
        Parameters:
        None
        ----------------
        Returns:
        list : A list of table names in the database
        """
        table_list = []
        try:
            self.cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table';")
            tables = self.cursor.fetchall()
            print("Tables in the database:")
            for table in tables:
                table_list.append(table[0])
                print(table)
            return table_list
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return None

    def read_tables_to_dataframes(self, table_names: list) -> list:
        """
        Description: function to read tables from the database into pandas DataFrames
        ----------------
        Parameters:
        table_names : list : list of table names to read from the database
        ----------------
        Returns:
        list : A list containing DataFrames for each table
        """
        df_list = []
        try:
            for table in table_names:
                df = pd.read_sql_query(f"SELECT * FROM {table};", self.conn)
                df_list.append(df)
            return df_list

        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
            return None


# NOTE:: Use case
# database_path = './data/dialects_database.db'
# db_handler = DBHandler(database_path)

# db_table_names = db_handler.list_tables()

# if db_table_names is not None:
#     df_list = db_handler.read_tables_to_dataframes(db_table_names)
# df = pd.merge(df_list[0], df_list[1], on='id')
# db_handler.close_connection()

# print(df.shape)
# df.head()
# df.to_csv('./data/dialects_data.csv', index=False)
