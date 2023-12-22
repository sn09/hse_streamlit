"""Module with methods for downloading data."""

from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data"

COLUMNS_ORDER = [
    "ID_CLIENT",
    "AGE",
    "GENDER",
    "EDUCATION",
    "MARITAL_STATUS",
    "CHILD_TOTAL",
    "DEPENDANTS",
    "SOCSTATUS_WORK_FL",
    "SOCSTATUS_PENS_FL",
    "SOCSTATUS_WORK_DESC",
    "SOCSTATUS_PENS_DESC",
    "REG_ADDRESS_PROVINCE",
    "FACT_ADDRESS_PROVINCE",
    "POSTAL_ADDRESS_PROVINCE",
    "FL_PRESENCE_FL",
    "OWN_AUTO",
    "GEN_INDUSTRY",
    "GEN_TITLE",
    "JOB_DIR",
    "WORK_TIME",
    "FAMILY_INCOME",
    "CREDIT",
    "TERM",
    "FST_PAYMENT",
    "PERSONAL_INCOME",
    "LOAN_NUM_TOTAL",
    "LOAN_NUM_CLOSED",
    "AGREEMENT_RK",
    "TARGET",
]

BINARY_COLUMNS = [
    "GENDER",
    "SOCSTATUS_WORK_FL",
    "SOCSTATUS_PENS_FL",
    "SOCSTATUS_WORK_DESC",
    "SOCSTATUS_PENS_DESC",
    "FL_PRESENCE_FL",
    "TARGET",
]


def join_tables(data_path: str = DEFAULT_DATA_PATH, dropna: bool = False, drop_outliers: bool = False) -> pd.DataFrame:
    """Join and preprocess tables.

    :param str data_path: path to directory with data
    :param bool dropna: drop objects without target
    :param bool drop_outliers: drop outliers in WORK_TIME
    :return pd.DataFrame: joined dataframe
    """

    join_type = "inner" if dropna else "left"

    # Get clients table
    D_CLIENTS = pd.read_csv(data_path / "D_clients.csv").rename(columns={"ID": "ID_CLIENT"})

    # Get job table
    D_JOB = pd.read_csv(data_path / "D_job.csv")
    object_columns = D_JOB.select_dtypes("object").columns
    num_columns = D_JOB.columns.difference(object_columns)
    D_JOB[object_columns] = D_JOB[object_columns].fillna("unknown")
    D_JOB[num_columns] = D_JOB[num_columns].fillna(-1)

    if drop_outliers:
        D_JOB = D_JOB[D_JOB["WORK_TIME"] < 1e4]

    # Get loan table
    D_LOAN = pd.read_csv(data_path / "D_loan.csv")
    D_CLOSE_LOAN = pd.read_csv(data_path / "D_close_loan.csv")
    D_LOAN = D_LOAN.merge(D_CLOSE_LOAN, on="ID_LOAN", how="left")
    D_LOAN = D_LOAN.groupby("ID_CLIENT", as_index=False).agg(
        LOAN_NUM_TOTAL=("CLOSED_FL", "count"),
        LOAN_NUM_CLOSED=("CLOSED_FL", "sum"),
    )

    # Get work description table
    D_WORK = (
        pd.read_csv(data_path / "D_work.csv")
        .drop(columns="ID")
        .rename(columns={"FLAG": "SOCSTATUS_WORK_FL", "COMMENT": "SOCSTATUS_WORK_DESC"})
    )

    # Get pens desctiption table
    D_PENS = (
        pd.read_csv(data_path / "D_pens.csv")
        .drop(columns="ID")
        .rename(columns={"FLAG": "SOCSTATUS_PENS_FL", "COMMENT": "SOCSTATUS_PENS_DESC"})
    )

    # Get target table
    D_TARGET = pd.read_csv(data_path / "D_target.csv")

    # Get salary table
    D_SALARY = pd.read_csv(data_path / "D_salary.csv").drop_duplicates()

    # Get last credit table
    D_LAST_CREDIT = pd.read_csv(data_path / "D_last_credit.csv")

    result_df = (
        D_CLIENTS
        .merge(D_TARGET, on="ID_CLIENT", how=join_type)
        .merge(D_SALARY, on="ID_CLIENT", how=join_type)
        .merge(D_JOB, on="ID_CLIENT", how=join_type)
        .merge(D_LOAN, on="ID_CLIENT", how=join_type)
        .merge(D_LAST_CREDIT, on="ID_CLIENT", how=join_type)
        .merge(D_WORK, on="SOCSTATUS_WORK_FL", how=join_type)
        .merge(D_PENS, on="SOCSTATUS_PENS_FL", how=join_type)
    )

    result_df[BINARY_COLUMNS] = result_df[BINARY_COLUMNS].astype("category")

    if dropna:
        result_df = result_df.dropna()

    return result_df[COLUMNS_ORDER]
