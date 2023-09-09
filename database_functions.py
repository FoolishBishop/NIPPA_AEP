# notas importantes:
# conexion.execute("insert into articulos(descripcion,precio) values (?,?)", ("bananas", 25))
# conexion.commit()
# conexion.close()  <- FALTA ESE

def find_names_columns(conection, table):
    cursor = conection.cursor()
    cursor.execute(f"SELECT * FROM {table}")
    return cursor.column_names()  # tupla de los nombres


def load_values(conection, table, values_tupple):
    columns = ",".join(find_names_columns(conection, table))
    int_sign = ("?," * len(values_tupple))[:-1]
    conection.execute(f"insert into {table}({columns}) values ({int_sign}", values_tupple)
    conection.commit()
    conection.close()


def read_values(conection, table, column_value, value):
    cursor = conection.cursor()
    cursor.execute(f"SELECT * FROM {table} WHERE {column_value} = {value}")
    return cursor.fetchone()  # tuple de toda la fila
