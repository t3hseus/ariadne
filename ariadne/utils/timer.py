import torch


#@torch.jit.script
class CudaTimer:
    times = dict()
    
    def clear():
        for key in list(CudaTimer.times):
            del CudaTimer.times[key]

    def timeit(name=''):
        def decorator(func):
            def timer(*args, **kwargs):
                stream = torch.cuda.current_stream(torch.cuda.current_device())
                cuda_start = torch.cuda.Event(enable_timing=True)
                cuda_end = torch.cuda.Event(enable_timing=True)
                if name not in CudaTimer.times:
                    CudaTimer.times[name] = []

                cuda_start.record(stream)
                result = func(*args, **kwargs)
                cuda_end.record(stream)
                torch.cuda.synchronize()
                CudaTimer.times[name].append(cuda_start.elapsed_time(cuda_end))
                return result
            return timer
        return decorator
