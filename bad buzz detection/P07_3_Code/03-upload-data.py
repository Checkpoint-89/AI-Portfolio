from azureml.core import Workspace
ws = Workspace.from_config()
datastore = ws.get_default_datastore()

datastore.upload(src_dir='./data',
                 target_path='datasets/data',
                 overwrite=True)

datastore.upload(src_dir='./embeddings',
                 target_path='datasets/embeddings',
                 overwrite=True)

datastore.upload(src_dir='./checkpoints',
                 target_path='datasets/checkpoints',
                 overwrite=True)