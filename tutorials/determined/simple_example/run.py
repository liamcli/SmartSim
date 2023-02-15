import time

import determined as det
from smartsim import Experiment
from smartredis import Client
import numpy as np

def main():
    info = det.get_cluster_info()
    chief_ip = info.container_addrs[0]
    if info.container_rank==0:
        exp = Experiment("foo", launcher="local")
        db = exp.create_database(port=4200)
        exp.start(db)
        print(db.get_address())
        client = Client(address=f"{chief_ip}:4200", cluster=False)
        client.put_tensor("test", np.array([0,1,2]))

        while not client.poll_tensor("done", 1000, 5):
            pass
        print('worker sent back done')
        exit()
    else:
        success = False
        while not success:
            try:
                print("trying to initialize redis client")
                client = Client(address=f"{chief_ip}:4200", cluster=False)
                success= True
            except Exception as e:
                print(e)

        while not client.poll_tensor("test", 1000, 5):
            print("tensor not available")
            pass
        test = client.get_tensor("test")
        if test.sum()==3:
            print('received test tensor from chief')
            client.put_tensor("done", np.array([1]))


if __name__=="__main__":
    main()
