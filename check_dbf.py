from dbfread import DBF

dbf_path = r"D:\cheque-ocr\F_23022026_010\HD010.DBF"

try:
    table = DBF(dbf_path, ignore_missing_memofile=True)
    print(f"✅ DBF has {len(table)} records. Fields are:")
    print(table.field_names)
    for i, record in enumerate(table):
        print(record)
        if i >= 4:  # just show first 5 records
            break
except Exception as e:
    print(f"❌ Error reading DBF: {e}")