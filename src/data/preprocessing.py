import pandas as pd
import numpy as np

def load_raw_data(path: str):
    """
    Load raw Telco churn dataset.
    """
    df = pd.read_csv(path)
    return df


def clean_total_charges(df: pd.DataFrame):
    """
    Convert TotalCharges to numeric and handle errors.
    """
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    return df


def clean_senior_citizen(df: pd.DataFrame):
    """
    Convert SeniorCitizen integer flag to Yes/No for consistency.
    """
    df['SeniorCitizen'] = df['SeniorCitizen'].replace({1: 'Yes', 0: 'No'})
    return df


def drop_customer_id(df: pd.DataFrame):
    """
    customerID is not useful for prediction.
    """
    df = df.drop(columns=['customerID'])
    return df


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline.
    """
    df = clean_total_charges(df)
    df = clean_senior_citizen(df)
    df = drop_customer_id(df)
    return df
