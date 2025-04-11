import os
import pytest
import pandas as pd
from src.data_loader import load_data

def test_load_data_success(tmp_path):
    # Create a temporary CSV file with sample data
    data = "price\tyear_sold\tproperty_tax\n295850\t2013\t234\n216500\t2006\t169"
    d = tmp_path / "subdir"
    d.mkdir()
    file_path = d / "test.csv"
    file_path.write_text(data)
    
    df = load_data(str(file_path))
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    assert "price" in df.columns

def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")
