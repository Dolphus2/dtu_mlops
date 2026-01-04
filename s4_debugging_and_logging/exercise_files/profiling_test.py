import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity
from torch.profiler import profile, tensorboard_trace_handler

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], on_trace_ready=tensorboard_trace_handler("./log/resnet18")) as prof:
    for i in range(100):
        model(inputs)
        prof.step()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
prof.export_chrome_trace("trace.json")