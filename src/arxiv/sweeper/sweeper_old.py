from sqlalchemy import create_engine, text
import friendlywords
import json

class Sweeper():
    """
    """

    def __init__(self, study, config):
        self.study = study
        self.config = config
        self.sweep_name = config['sweep_info'].get('sweep_name')
        self.storage_url = study._storage._backend.url
        if self.sweep_name is None:
            self.sweep_name = self._random_sweep_name()
        

        if not self._sweep_registered():
            self._register_sweep()

    def _sweep_registered(self):
        """
        Check if the sweep has been registered in the database
        """
        engine = create_engine(self.storage_url)
        with engine.connect() as conn:
            stmt = f"SELECT sweep_name FROM sweeps WHERE study_id={self.study._study_id}"
            sweep_names = [x[0] for x in conn.execute(text(stmt))]
            sweep_exists = self.sweep_name in sweep_names
        return sweep_exists        


    def _register_sweep(self):
        engine = create_engine(self.storage_url)
        with engine.connect() as conn:
            existing_sweep_ids = list(conn.execute(text("SELECT sweep_id FROM sweeps")))
            sweep_id = 0 if len(existing_sweep_ids) == 0 else existing_sweep_ids[-1][0] + 1

            stmnt = f"""
            INSERT INTO sweeps (sweep_id, study_id, sweep_name, config) 
            VALUES (:sweep_id, :study_id, :sweep_name, :config)
            """
            conn.execute(text(stmnt), [{"sweep_id": sweep_id,
                                        "study_id": self.study._study_id, 
                                        "sweep_name": self.sweep_name,
                                        "config": json.dumps(self.config)}])
            conn.commit()


    def _random_sweep_name(self, max_attempts=100):
        """
        Generate a random human-readable sweep name
        """
        engine = create_engine(self.storage_url)
        with engine.connect() as conn:
            stmt = "SELECT sweep_name FROM sweeps"
            out = conn.execute(text(stmt))
            sweep_name = '-'.join(friendlywords.generate('po').split(' '))
            attempts = 0
            while sweep_name in out:
                sweep_name = '-'.join(friendlywords.generate('po').split(' '))
                attempts += 1
                if attempts > max_attempts:
                    raise Exception(f"Couldn't generate a random sweep name in {attempts} attempts")

        return sweep_name