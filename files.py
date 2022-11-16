import pandas as pd
from fastparquet import write, ParquetFile
from pyarrow.parquet import ParquetFile
import pyarrow.parquet as pq
import pyarrow.parquet

patient = pd.read_csv('static/Web_Patients.csv')
lc50 = pd.read_csv('static/Web_LC50_newlc50.csv')
cnv = pd.read_parquet('static/cnv_del.parquet',engine = 'pyarrow')
snv = pd.read_parquet('static/Web_snv_fillna.parquet',engine = 'pyarrow')
pid = pd.read_parquet('static/pknumbers.parquet',engine = 'pyarrow')



