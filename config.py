import configparser
import os

class Config:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance.load_config()
        return cls._instance

    def load_config(self):
        self.config = configparser.ConfigParser()
        if os.path.exists('config.ini'):
            self.config.read('config.ini')
        else:
            raise FileNotFoundError("El archivo config.ini no existe.")
        
        # Crear directorios si no existen
        paths = self.config['PATHS']
        for key in paths:
            if key.endswith('_folder'):
                os.makedirs(paths[key], exist_ok=True)

    def get(self, section, key, fallback=None):
        return self.config.get(section, key, fallback=fallback)

    def get_bool(self, section, key, fallback=False):
        return self.config.getboolean(section, key, fallback=fallback)

    def get_float(self, section, key, fallback=0.0):
        return self.config.getfloat(section, key, fallback=fallback)

    def get_int(self, section, key, fallback=0):
        return self.config.getint(section, key, fallback=fallback)

cfg = Config()