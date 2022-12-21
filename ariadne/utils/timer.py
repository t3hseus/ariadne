import torch


@torch.jit.script
class CudaTimer:
    def __init__(self):
        self.times : Dict[str, List[float]] = dict()
        self.cuda_events = {'sample': (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))}
        self.stream = torch.cuda.current_stream(torch.cuda.current_device())
    
    def start(self, name: str):
        if name not in self.times:
            self.times[name] : List[float] = [0.0]
            self.times[name].remove(0.0)
            self.cuda_events[name] = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        self.cuda_events[name][0].record(self.stream)

    def end(self, name: str):
        self.cuda_events[name][1].record(self.stream)
        torch.cuda.synchronize()
        self.times[name].append(self.cuda_events[name][0].elapsed_time(self.cuda_events[name][1]))
        
    def clear(self):
        for key in self.cuda_events:
            del self.cuda_events[key]
        for key in self.times:
            del self.times[key]
