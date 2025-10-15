from eexp_engine import client
import eexp_config

exp_name = 'ads_exp'

if __name__ == "__main__":
    client.run(__file__, exp_name, eexp_config)
