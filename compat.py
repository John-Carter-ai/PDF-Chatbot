import sys
import pysqlite3
sys.modules["sqlite3"] = sys.modules["pysqlite3"]
import sqlite3
