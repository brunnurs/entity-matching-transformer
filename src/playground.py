from config import Config
from run_training import create_experiment_folder

if __name__ == "__main__":

    Config().set_arguments_to_config()
    exp_name = create_experiment_folder()
    Config().dump_config_to_json_file(exp_name)



