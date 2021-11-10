import netCDF4
import numpy as np

class BatchGenerator:
    def __init__(self, 
            datasets,
            box_shape=(24,128,128),
            past_steps=12,
            random_seed=None,
            batch_size=32
        ):

        self.datasets = datasets
        self.shape = self.datasets[0].get_shape()
        self.rng = np.random.RandomState(seed=random_seed)
        self.next_indices = np.array([], dtype=int)
        self.box_shape = box_shape
        self.past_steps = past_steps
        self.batch_size = batch_size

    def draw_indices(self):
        while len(self.next_indices) < self.batch_size:
            next_ind = np.arange(self.shape[0], dtype=int)
            self.rng.shuffle(next_ind)
            self.next_indices = np.hstack((self.next_indices, next_ind))

        ind = self.next_indices[:self.batch_size]
        self.next_indices = self.next_indices[self.batch_size:]

        return ind

    def random_box(self):
        t0 = self.rng.randint(self.shape[1]-self.box_shape[0])
        t1 = t0 + self.past_steps
        t2 = t0 + self.box_shape[0]
        i0 = self.rng.randint(self.shape[2]-self.box_shape[1])
        i1 = i0 + self.box_shape[1]
        j0 = self.rng.randint(self.shape[3]-self.box_shape[2])
        j1 = j0 + self.box_shape[2]

        return (
            ((t0,t1),(i0,i1),(j0,j1)),
            ((t1,t2),(i0,i1),(j0,j1)),
        )

    def augment_batches(self, inputs, outputs):
        rot = self.rng.randint(4)
        flipud = bool(self.rng.randint(2))
        fliplr = bool(self.rng.randint(2))
        for batch in [inputs, outputs]:
            for i in range(len(batch)):
                if rot > 0:
                    batch[i] = np.rot90(batch[i], k=rot, axes=(-3,-2))
                if flipud:
                    batch[i] = batch[i][:,:,::-1,:,:]
                if fliplr:
                    batch[i] = batch[i][:,:,:,::-1,:]

        return (inputs, outputs)

    def __next__(self):
        ind = self.draw_indices()
        (past_box, future_box) = self.random_box()
        return self.get_batch(ind, past_box, future_box)

    def get_batch(self, ind, past_box, future_box):
        boxes = {
            "past": past_box,
            "future": future_box,
            "target": future_box
        }

        inputs = []
        for dataset in self.datasets:            
            for group in ["past", "future"]:
                box = boxes[group]
                inp = []
                for var in dataset.variables.get(group, []):
                    inp.append(dataset.get_batch(var, ind, box))
                if inp:
                    inp = np.concatenate(inp, axis=-1)
                    inputs.append(inp)

        outputs = []
        for dataset in self.datasets:
            group = "target"
            box = boxes[group]
            for var in dataset.variables.get(group, []):
                outputs.append(dataset.get_batch(var, ind, box))

        (inputs, outputs) = self.augment_batches(inputs, outputs)

        return (inputs, outputs)


class MCHRadarDataset:
    name = "mchradar"

    def __init__(self,
        data_fn, 
        in_memory=False,
        variables=None,
    ):
        if variables is None:
            variables = {
                "past": [
                    "RZC", "CZC", "EZC-15", "EZC-20", 
                    "EZC-45", "EZC-50", "BZC", "HZC"
                ],
                "target": [
                    "THRESH-RZC-10"
                ],                
            }
        self.variables = variables
        self.in_memory = in_memory
        
        if in_memory:
            with open(data_fn, 'rb') as f:
                mem = f.read()
            self.ds = netCDF4.Dataset(None, 'r', memory=mem)
            self.dataset = self.ds.variables
            self.load_scales(self.dataset)
        else:
            self.ds = netCDF4.Dataset(data_fn, 'r')
            self.dataset = self.ds.variables
            self.load_scales(self.dataset)

    def __del__(self):        
        self.ds.close()

    def get_shape(self):
        return self.dataset[self.variables["past"][0]].shape

    def load_scales(self, dataset):
        self.scale = {}
        for v in dataset.keys():
            if v.endswith("_scale"):
                scale_var = v[:-6]
                self.scale[scale_var] = np.array(self.dataset[v][:])

    def transform_data(self, data, variable):
        data = self.scale[variable][data]
        if variable in ["RZC", "CZC"]:
            zero = data < 0.1
            data[zero] = 0.01
            data = np.log10(data)
            data -= 0.072
            data /= 0.56
        elif variable.startswith("EZC"):
            data /= 6.0
        elif variable == "BZC":
            data /= 50.0
        elif variable == "HZC":
            data = np.nan_to_num(data, copy=False, nan=0.0)
            data /= 3.6     

        return data

    def get_batch(self, variable, ind, box):
        ((t0,t1),(i0,i1),(j0,j1)) = box
        
        if variable.startswith("THRESH"):
            parts = variable.split("-")
            base_var = parts[1]
            thresh = float(parts[2])
            data = self.dataset[base_var][ind,t0:t1,i0:i1,j0:j1,:]
            data = self.scale[base_var][data]
            data = (data >= thresh).astype(np.float32)
        else:
            data = self.dataset[variable][ind,t0:t1,i0:i1,j0:j1,:]
            data = self.transform_data(data, variable)
        return data
