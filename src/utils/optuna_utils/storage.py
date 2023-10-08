from optuna.storages import RDBStorage
from sqlalchemy import create_engine, text, JSON

class CustomRDBStorage(RDBStorage):
    """
    Wrapper around Optuna's RDBStorage.
    All RDB tables from RDBStorage are preserved.
    We add extra tables, and delete them when study is deleted from storage
    """
    def __init__(self, 
                 url,
                 engine_kwargs = None,
                 skip_compatibility_check= False,
                 *,
                 heartbeat_interval= None,
                 grace_period = None,
                 failed_trial_callback = None,
                 skip_table_creation = False,
    ):
        super().__init__(url=url, 
                         engine_kwargs=engine_kwargs, 
                         skip_compatibility_check=skip_compatibility_check, 
                         heartbeat_interval=heartbeat_interval, 
                         grace_period=grace_period, 
                         failed_trial_callback=failed_trial_callback, 
                         skip_table_creation=skip_table_creation
                         )
        
        self.create_sweep_table()
    
    def create_sweep_table(self):
        # Create a sweep table if it doesn't exist
        engine = create_engine(self.url)
        with engine.connect() as conn:
            # stmt = "DROP TABLE sweeps"
            # stmt = "SELECT * FROM trial_values LIMIT 10"
            # out = conn.execute(text(stmt))
            stmt = """CREATE TABLE IF NOT EXISTS sweeps (
                sweep_id INT,
                study_id INT,
                sweep_name VARCHAR(255),
                config JSON,
                PRIMARY KEY (sweep_id, study_id, sweep_name)
            )"""
            conn.execute(text(stmt))
            conn.commit()
            stmt = """CREATE TABLE IF NOT EXISTS sweeps_intermediate_values (
                sweep_id INT,
                study_id INT,
                trial_id INT,
                step INT,
                info JSON,
                PRIMARY KEY (sweep_id, study_id, trial_id, step)
            )"""
            conn.execute(text(stmt))
            conn.commit()

    def delete_study(self, study_id):
        super().delete_study(study_id)