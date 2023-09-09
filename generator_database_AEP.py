import sqlite3

conexion = sqlite3.connect("nippa_sensors.db")
conexion.execute("""create table if not exists Accelerometer_MPU6050 (
                          ID integer primary key AUTOINCREMENT,
                          time text,
                          Bx text,
                          By text,
                          Ax text,
                          Ay text,
                          Az text,
                          Gx text,
                          Gy text,
                          Gz text                          
                    )""")
conexion.execute("""create table if not exists Temperature_MS8607 (
                          ID integer primary key AUTOINCREMENT,
                          time text,
                          temp text,
                          humidity text,
                          pressure text                    
                    )""")
conexion.execute("""create table if not exists UV_table (
                          ID integer primary key AUTOINCREMENT,
                          time text,
                          UV text
                    )""")

conexion.close()

# revisar si son todos esos datos
